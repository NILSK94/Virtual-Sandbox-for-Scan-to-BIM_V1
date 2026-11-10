import base64
import asyncio
import copy
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from scanbim import claude, settings
from scanbim.agents import AgentRunner
from scanbim.model import SceneStore
from scanbim.model_tests import function_test


def test_provider_settings_preserve_keys_and_survive_restart(monkeypatch):
    settings.save_settings("openai-private", "openai-model", "openai")
    path=settings.settings_path()
    path.write_text(path.read_text(encoding="utf8")+'\n[extra_service]\napi_key="extra-private"\n',encoding="utf8")
    settings.save_settings("claude-private", "claude-model", "claude")
    monkeypatch.setenv("SCANBIM_PROVIDER","openai")
    assert settings.active_provider()=="claude"
    assert settings.load_credentials().api_key=="claude-private"
    assert settings.load_credentials("openai").api_key=="openai-private"
    settings.save_settings("", "second-claude-model", "claude")
    assert settings.load_credentials().api_key=="claude-private"
    assert settings._read_values(section="extra_service")["api_key"]=="extra-private"
    assert settings.settings_status("openai")["model"]=="openai-model"
    assert "private" not in json.dumps(settings.settings_status())
    settings.save_settings("", "qwen3-vl:4b", "ollama")
    assert settings.load_credentials().provider=="ollama"
    settings.save_settings("", "openai-model", "openai")
    assert settings.load_credentials().api_key=="openai-private"


@pytest.mark.parametrize("provider",["openai","claude","ollama"])
@pytest.mark.parametrize("kind",["text","vision","tools","workflow"])
def test_function_tests_use_images_and_complete_tool_roundtrip(monkeypatch,provider,kind):
    real_client=httpx.Client
    requests=[]
    def handle(request):
        body=json.loads(request.content)
        if request.url.path=="/api/show":
            return httpx.Response(200,json={"capabilities":["vision","tools"]})
        requests.append(body)
        if provider=="claude":
            assert request.headers["x-api-key"]=="test-private"
            blocks=[b for m in body["messages"] for b in m["content"]]
            images=[b["source"]["data"] for b in blocks if b["type"]=="image"]
            followup=any(b["type"]=="tool_result" for b in blocks)
        elif provider=="ollama":
            images=[image for m in body["messages"] for image in m.get("images",[])]
            followup=any(m["role"]=="tool" for m in body["messages"])
        else:
            assert request.headers["authorization"]=="Bearer test-private"
            images=[p["image_url"].split(",",1)[1] for m in body["input"]
                    if isinstance(m.get("content"),list) for p in m["content"] if p["type"]=="input_image"]
            followup=any(m.get("type")=="function_call_output" for m in body["input"])
        answer="OK"
        if images and not followup:
            with Image.open(io.BytesIO(base64.b64decode(images[0]))) as image:
                answer={(255,0,0):"RED",(0,255,0):"GREEN",(0,0,255):"BLUE"}[image.getpixel((0,0))]
        if body.get("tools"):
            if provider=="claude":
                return httpx.Response(200,json={"content":[{"type":"tool_use","id":"test-call",
                    "name":"report_test","input":{"answer":answer}}]})
            if provider=="ollama":
                return httpx.Response(200,json={"message":{"tool_calls":[{"function":{
                    "name":"report_test","arguments":{"answer":answer}}}]}})
            return httpx.Response(200,json={"output":[{"type":"function_call","call_id":"test-call",
                "name":"report_test","arguments":json.dumps({"answer":answer})}]})
        if provider=="claude":
            return httpx.Response(200,json={"content":[{"type":"text","text":answer}]})
        if provider=="ollama":
            return httpx.Response(200,json={"message":{"content":answer}})
        return httpx.Response(200,json={"output":[{"type":"message",
            "content":[{"type":"output_text","text":answer}]}]})
    monkeypatch.setattr(httpx,"Client",lambda **kwargs:real_client(transport=httpx.MockTransport(handle),**kwargs))
    assert function_test("test-private","test-model",provider,kind)["ok"]
    assert len(requests)==(2 if kind=="workflow" else 1)


def test_claude_agent_receives_scan_and_tool_results(tmp_path,monkeypatch):
    monkeypatch.setenv("SCANBIM_PROVIDER","claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY","claude-test-private")
    monkeypatch.setenv("ANTHROPIC_MODEL","test-model")
    scene=SceneStore(tmp_path).create(np.array([[0.,0.,0.],[1.,1.,1.]]),
                                     np.full((2,3),200,dtype=np.uint8),"test","test-source")
    real_client=httpx.Client
    requests=[]
    def handle(request):
        assert str(request.url)==claude.BASE_URL+"/messages"
        assert request.headers["x-api-key"]=="claude-test-private"
        body=json.loads(request.content)
        requests.append(copy.deepcopy(body))
        if len(requests)==1:
            assert any(b["type"]=="image" for m in body["messages"] for b in m["content"])
            tool="define_tool"
            args={"name":"New tool","description":"Measured scope","code":"result={}"}
        else:
            assert any(b["type"]=="tool_result" for m in body["messages"] for b in m["content"])
            tool="finish"
            args={"summary":"Claude-Probe abgeschlossen.","status":"partial"}
        return httpx.Response(200,json={"content":[{"type":"tool_use","id":f"call-{len(requests)}",
            "name":"act","input":{"tool":tool,"arguments_json":json.dumps(args),"rationale":"Probe"}}]})
    monkeypatch.setattr(httpx,"Client",lambda **kwargs:real_client(transport=httpx.MockTransport(handle),**kwargs))
    assert "Claude-Probe" in AgentRunner()._vlm(scene,4)
    assert scene.state["custom_tools"][0]["memory_id"]


def test_provider_errors_never_echo_secrets(monkeypatch):
    real_client=httpx.Client
    def handle(request):
        return httpx.Response(401,json={"error":{"message":"claude-test-private"}})
    monkeypatch.setattr(httpx,"Client",lambda **kw:real_client(transport=httpx.MockTransport(handle),**kw))
    result=function_test("claude-test-private","test-model","claude","workflow")
    assert not result["ok"] and "401" in result["message"]
    assert "claude-test-private" not in json.dumps(result)


def test_settings_routes_select_provider_and_forward_function_test(tmp_path,monkeypatch):
    import scanbim.app as app_module
    app=app_module.create_app(tmp_path/"data")
    endpoints={route.name:route.endpoint for route in app.routes if hasattr(route,"endpoint")}
    async def inline_thread(function,*args):
        return function(*args)
    monkeypatch.setattr(asyncio,"to_thread",inline_thread)
    class Request:
        def __init__(self,values):
            self.values=values
        async def body(self):
            return json.dumps(self.values).encode()
    async def exercise():
        result=await endpoints["update_settings"](Request({
            "provider":"claude","api_key":"private-route-key","model":"claude-test-model"}))
        assert json.loads(result.body)["provider"]=="claude"
        assert b"private-route-key" not in result.body
        monkeypatch.setattr(app_module,"function_test",lambda key,model,provider,kind:{
            "ok":key=="" and model=="draft-model" and provider=="openai" and kind=="workflow"})
        result=await endpoints["test_model_function"](Request({
            "provider":"openai","model":"draft-model","kind":"workflow"}))
        assert json.loads(result.body)["ok"]
        with pytest.raises(ValueError):
            await endpoints["test_model_function"](Request({"provider":42,"api_key":"secret"}))
    # These handlers only await request parsing and to_thread. Drive the
    # immediate awaitables without opening an event-loop socket in the sandbox.
    with pytest.raises(StopIteration):
        exercise().send(None)
    response=endpoints["get_settings"]("claude")
    assert json.loads(response.body)["has_api_key"]
    assert b"private-route-key" not in response.body
