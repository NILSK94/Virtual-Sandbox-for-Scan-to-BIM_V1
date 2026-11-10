"""Run with unittest as well as pytest; no network or real credentials are used."""
import base64
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from scanbim import agents, token_context as tc
from scanbim.model import SceneStore
from scanbim.settings import Credentials
from scanbim.tools import ToolRegistry


class TokenContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.scene = SimpleNamespace(folder=self.folder)

    def test_full_precision_results_and_code_are_retrievable(self):
        values = {"vertices": [[i + .123456789, 2, 3] for i in range(2000)],
                  "result": {"code": "result = 'exact'\n" * 900}, "rms_m": .000123456789}
        encoded = tc.pack_result(self.scene, values)
        self.assertLess(len(encoded), 5500)
        summary = json.loads(encoded)
        self.assertEqual(summary["preview"]["rms_m"], values["rms_m"])
        result_id = summary["result_id"]
        self.assertEqual(json.loads((self.folder / "agent-results" / (result_id + ".json")).read_text()), values)
        page = tc.read_result(self.scene, {"result_id": result_id, "pointer": "/vertices", "offset": 1200, "limit": 2})
        self.assertEqual(page["value"], values["vertices"][1200:1202])
        code = tc.read_result(self.scene, {"result_id": result_id, "pointer": "/result/code", "offset": 100, "limit": 2000})
        self.assertEqual(code["value"], values["result"]["code"][100:2100])
        with self.assertRaises(ValueError):
            tc.read_result(self.scene, {"result_id": "../../settings"})
        with self.assertRaises(ValueError):
            tc.read_result(self.scene, {"result_id": result_id, "pointer": "/does-not-exist"})

    def test_history_is_bounded_and_keeps_complete_rounds(self):
        rounds = [[{"type": "function_call", "call_id": str(i), "name": "act", "arguments": "{}"},
                   {"type": "function_call_output", "call_id": str(i), "output": "x" * 1000}] for i in range(20)]
        initial = {"role": "user", "content": "tools"}
        image = {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}]}
        messages = tc.model_input(initial, rounds, "current", image)
        self.assertEqual(len([m for m in messages if m.get("type") == "function_call"]), 3)
        ids = {m["call_id"] for m in messages if m.get("type") == "function_call"}
        self.assertEqual(ids, {m["call_id"] for m in messages if m.get("type") == "function_call_output"})
        self.assertEqual(messages[-1], image)
        self.assertEqual(sum(m.get("content", "") == "Aktueller Szenenstand: current" for m in messages), 1)

    def test_large_pages_keep_original_handle_and_bounded_preview(self):
        large = {str(i): {"code": "q" * 10000} for i in range(300)}
        result_id = json.loads(tc.pack_result(self.scene, large))["result_id"]
        page = tc.read_result(self.scene, {"result_id": result_id, "limit": 100})
        self.assertEqual(page["result_id"], result_id)
        self.assertLess(len(tc.dumps(page)), 5000)
        index = tc.read_result(self.scene, {})
        self.assertIn(result_id, [item["result_id"] for item in index["results"]])

    def test_large_scene_keeps_all_elements_locally_and_reset_removes_archive(self):
        from scanbim.reset import reset_scene
        scene = SceneStore(self.folder / "scene").create(
            np.array([[0., 0., 0.], [1., 1., 1.]]), np.zeros((2, 3), dtype=np.uint8), "probe", "hash")
        scene.state["elements"] = [{"id": str(i), "name": "Bauteil " + str(i), "kind": "mesh",
                                    "vertices": [[0, 0, 0]] * 100, "faces": [[0, 0, 0]] * 100,
                                    "uncertainty": "Messnachweise vorhanden " * 10} for i in range(100)]
        context = json.loads(tc.scene_context(scene, 20))
        result_id = context["result_id"]
        self.assertEqual(context["remaining_steps"], 20)
        self.assertEqual(len(context["current_elements"]), 100)
        self.assertEqual(len(scene.state["elements"]), 100)
        self.assertEqual(len(scene.state["elements"][99]["vertices"]), 100)
        exact = ToolRegistry()._execute(scene, "read_agent_result", {
            "result_id": result_id, "pointer": "/elements/99/name", "limit": 100}, "agent")
        self.assertEqual(exact["value"], "Bauteil 99")
        reset_scene(scene, scene.state["revision"], confirmed=True)
        self.assertFalse((scene.folder / "agent-results").exists())

    def test_small_image_default_and_explicit_high_detail_preserve_source(self):
        path = self.folder / "view.png"
        Image.new("RGB", (960, 640), (100, 120, 140)).save(path)
        original = path.read_bytes()
        view = {"file": path.name, "image_size": [960, 640]}
        low = tc.image_message(self.scene, {**view, "model_image_detail": "low"})
        high = tc.image_message(self.scene, view)
        def size(message):
            return Image.open(io.BytesIO(base64.b64decode(message["content"][1]["image_url"].split(",")[1]))).size
        self.assertEqual(size(low), (512, 341))
        self.assertEqual(size(high), (960, 640))
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(low["content"][1]["detail"], "low")

    def test_multistep_agent_for_all_providers(self):
        # Exercise pruning, image refresh, real local tools and usage accounting together.
        for provider in ("openai", "claude", "ollama"):
            with self.subTest(provider=provider):
                scene = SceneStore(self.folder / provider).create(
                    np.array([[0., 0., 0.], [1., 1., 1.], [2., 1., 2.]]),
                    np.full((3, 3), 120, dtype=np.uint8), "probe", "hash")
                requests = []
                def handle(request):
                    body = json.loads(request.content)
                    if request.url.path == "/api/show":
                        return httpx.Response(200, json={"capabilities": ["vision", "tools"]})
                    requests.append(copy.deepcopy(body))
                    step = len(requests)
                    tool = "finish" if step == 9 else "request_view" if step in (1, 5) else "remember"
                    args = ({"summary": "Probe abgeschlossen", "status": "partial"} if tool == "finish"
                            else {"label": str(step), "image_detail": "high" if step == 5 else "low"} if tool == "request_view"
                            else {"observation": "Messbefund " + str(step)})
                    action = {"tool": tool, "arguments_json": tc.dumps(args), "rationale": "Probe"}
                    if provider == "openai":
                        result = {"output": [{"type": "function_call", "name": "act", "call_id": str(step), "arguments": tc.dumps(action)}],
                                  "usage": {"input_tokens": 100, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 50}}}
                    elif provider == "claude":
                        result = {"content": [{"type": "tool_use", "name": "act", "id": str(step), "input": action}],
                                  "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50}}
                    else:
                        result = {"message": {"tool_calls": [{"function": {"name": "act", "arguments": action}}]},
                                  "prompt_eval_count": 100, "eval_count": 20}
                    return httpx.Response(200, json=result)
                real_client = httpx.Client
                def client(**kwargs):
                    return real_client(transport=httpx.MockTransport(handle), **kwargs)
                with patch.object(agents, "load_credentials", return_value=Credentials("test-key", "test-model", provider)), \
                     patch.object(agents.httpx, "Client", client):
                    summary = agents.AgentRunner()._vlm(scene, 12)
                self.assertIn("Probe abgeschlossen", summary)
                self.assertEqual(len(requests), 9)
                self.assertEqual(scene.state["agent"]["token_usage"]["input_tokens"], 900)
                self.assertEqual(scene.state["agent"]["token_usage"]["output_tokens"], 180)
                self.assertEqual(len([e for e in scene.events if e["tool"] == "model_usage"]), 9)
                for body in requests:
                    if provider == "openai":
                        messages = body["input"]
                        images = [p for m in messages if isinstance(m.get("content"), list) for p in m["content"] if p["type"] == "input_image"]
                        calls = [m for m in messages if m.get("type") == "function_call"]
                        outputs = [m for m in messages if m.get("type") == "function_call_output"]
                        self.assertEqual({m["call_id"] for m in calls}, {m["call_id"] for m in outputs})
                        self.assertLessEqual(len(calls), 3)
                    elif provider == "claude":
                        blocks = [p for m in body["messages"] for p in m["content"]]
                        images = [p for p in blocks if p["type"] == "image"]
                        self.assertEqual({p["id"] for p in blocks if p["type"] == "tool_use"},
                                         {p["tool_use_id"] for p in blocks if p["type"] == "tool_result"})
                    else:
                        images = [p for m in body["messages"] for p in m.get("images", [])]
                    self.assertEqual(len(images), 1)
                # The explicit high-detail view survives into the following request.
                if provider == "openai":
                    self.assertEqual(requests[5]["input"][-1]["content"][1]["detail"], "high")


if __name__ == "__main__":
    unittest.main()
