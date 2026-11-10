"""Loopback-only Ollama transport for the visual agent."""
from __future__ import annotations

import json
import re

import httpx

BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:4b"
UNAVAILABLE = "Ollama ist unter http://127.0.0.1:11434 nicht erreichbar. Ollama starten und erneut prüfen."


def is_local_model(name):
    return (isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9._/-]+(?::[A-Za-z0-9._-]+)?", name))
            and len(name) <= 200 and not name.lower().startswith(("sk-", "ollama.com/"))
            and not re.search(r"(?:^|[:/-])cloud(?:$|[-/])", name, re.I))


def model_status(client, model):
    try:
        response = client.post(BASE_URL + "/api/show", json={"model": model})
        if response.status_code == 404:
            return {"ok": False, "message": f"{model} ist in Ollama nicht installiert. Unter Verfügbare Modelle laden eine installierte Modell-ID auswählen oder im Modellfeld eintragen und erneut testen."}
        response.raise_for_status()
        details = response.json()
        if details.get("remote_host") or details.get("remote_model"):
            return {"ok": False, "message": "Bitte ein lokal installiertes Modell wählen."}
        if not {"vision", "tools"}.issubset(details.get("capabilities") or []):
            return {"ok": False, "message": "Das Modell muss Bildinput und Werkzeugaufrufe unterstützen."}
        return {"ok": True, "message": "Lokales Modell mit Bildinput und Werkzeugaufrufen verfügbar."}
    except (httpx.HTTPError, ValueError, AttributeError):
        return {"ok": False, "message": UNAVAILABLE}


def available_models():
    try:
        with httpx.Client(timeout=3., trust_env=False) as client:
            response = client.get(BASE_URL + "/api/tags")
            response.raise_for_status()
            models = sorted({item["name"] for item in response.json().get("models", [])
                             if is_local_model(item.get("name")) and item.get("size", 0) > 0
                             and not (item.get("remote_host") or item.get("remote_model"))})
        message = ("Installierte Modelle geladen. Bildinput und Werkzeugaufrufe über Modell prüfen bestätigen."
                   if models else "Keine lokalen Modelle installiert. Standardmodell mit Start.bat -InstallLocalModel laden.")
        return {"ok": True, "models": models, "message": message}
    except (httpx.HTTPError, ValueError, AttributeError):
        return {"ok": False, "models": [], "message": UNAVAILABLE}


def chat_payload(payload):
    messages = [{"role": "system", "content": payload["instructions"] + " Call act exactly once per response."}]
    tool_names = {}
    for item in payload["input"]:
        if item.get("type") == "function_call":
            tool_names[item["call_id"]] = item["name"]
            messages.append({"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": item["name"], "arguments": json.loads(item["arguments"])}}]})
        elif item.get("type") == "function_call_output":
            messages.append({"role": "tool", "tool_name": tool_names[item["call_id"]], "content": item["output"]})
        elif isinstance(item["content"], list):
            messages.append({"role": "user",
                "content": "\n".join(part["text"] for part in item["content"] if part["type"] == "input_text"),
                "images": [part["image_url"].split(",", 1)[1] for part in item["content"] if part["type"] == "input_image"]})
        else:
            messages.append(item)
    tools = [{"type": "function", "function": {key: tool[key] for key in ("name", "description", "parameters")}}
             for tool in payload["tools"]]
    return {"model": payload["model"], "messages": messages, "tools": tools, "stream": False,
            "keep_alive": "10m", "options": {"num_ctx": 8192, "num_predict": 2048}}


def chat_output(response, step):
    calls = response.get("message", {}).get("tool_calls") or []
    return [{"type": "function_call", "name": call["function"]["name"], "call_id": f"ollama-{step}-{index}",
             "arguments": json.dumps(call["function"]["arguments"], ensure_ascii=False)}
            for index, call in enumerate(calls)]
