"""Small, explicit end-to-end probes with synthetic inputs."""
import base64
import io
import json
import secrets

import httpx
from PIL import Image

from . import claude, ollama
from .settings import resolve_draft, local_credentials, resolve_provider


def _send(client, credentials, payload):
    if credentials.provider == "claude":
        response = client.post(claude.BASE_URL + "/messages",
                               headers=claude.headers(credentials.api_key), json=claude.chat_payload(payload))
    elif credentials.provider == "ollama":
        response = client.post(ollama.BASE_URL + "/api/chat", json=ollama.chat_payload(payload))
    else:
        response = client.post("https://api.openai.com/v1/responses",
                               headers={"Authorization": "Bearer " + credentials.api_key}, json=payload)
    response.raise_for_status()
    data = response.json()
    if credentials.provider == "claude":
        calls = claude.chat_output(data)
        text = " ".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    elif credentials.provider == "ollama":
        calls = ollama.chat_output(data, 0)
        text = data.get("message", {}).get("content", "")
    else:
        calls = [b for b in data.get("output", []) if b.get("type") == "function_call"]
        text = " ".join(part.get("text", "") for b in data.get("output", [])
                        if b.get("type") == "message" for part in b.get("content", [])
                        if part.get("type") == "output_text")
    return calls, text


def function_test(api_key, model, provider=None, kind="text"):
    provider = resolve_provider(provider)
    if kind not in ("text", "vision", "tools", "workflow"):
        raise ValueError("Test Text, Bild, Werkzeug oder Workflow auswählen.")
    credentials = local_credentials(model.strip()) if provider == "ollama" else resolve_draft(api_key, model, provider)
    if not credentials.model:
        raise ValueError("Bitte eine Modell-ID eintragen.")
    use_image = kind in ("vision", "workflow")
    use_tool = kind in ("tools", "workflow")
    colors = {"RED": (255,0,0), "GREEN": (0,255,0), "BLUE": (0,0,255)}
    expected = secrets.choice(list(colors)) if use_image else "OK"
    instruction = ("Identify the dominant colour of the supplied image. Answer with an uppercase colour name."
                   if use_image else "This is a minimal function test. The answer is OK.")
    instruction += " Call report_test with that answer." if use_tool else " Reply with only the answer."
    content = [{"type": "input_text", "text": instruction}]
    if use_image:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), colors[expected]).save(buffer, format="PNG")
        content.append({"type": "input_image", "image_url": "data:image/png;base64," +
                        base64.b64encode(buffer.getvalue()).decode()})
    tool = {"type": "function", "name": "report_test", "description": "Report the test answer.",
            "strict": True, "parameters": {"type": "object", "properties": {
                "answer": {"type": "string", "enum": ["OK", "RED", "GREEN", "BLUE"]}},
                "required": ["answer"], "additionalProperties": False}}
    payload = {"model": credentials.model, "instructions": "Follow the synthetic test instructions.",
               "input": [{"role": "user", "content": content}],
               "max_output_tokens": 512, "store": False, "tools": [tool] if use_tool else []}
    if use_tool:
        payload.update(tool_choice="required", parallel_tool_calls=False)
    try:
        with httpx.Client(timeout=httpx.Timeout(50., connect=5.), trust_env=provider != "ollama") as client:
            if provider == "ollama":
                status = ollama.model_status(client, model)
                if not status["ok"]:
                    return status
            calls, text = _send(client, credentials, payload)
            if use_tool:
                valid = (len(calls) == 1 and calls[0].get("name") == "report_test"
                         and json.loads(calls[0]["arguments"]) == {"answer": expected})
            else:
                valid = text.strip().upper().strip(".!") == expected
            if not valid:
                return {"ok": False, "message": "Testantwort entspricht nicht der erwarteten Funktion. "
                        "Bildverständnis, Werkzeugformat oder Ausgabelimit prüfen."}
            if kind == "workflow":
                payload["input"].extend([calls[0], {"type": "function_call_output",
                    "call_id": calls[0]["call_id"], "output": "Test tool executed successfully. Reply exactly OK."}])
                payload["instructions"] = "The test tool has executed. Confirm its result with exactly OK."
                payload.update(tools=[])
                payload.pop("tool_choice", None)
                payload.pop("parallel_tool_calls", None)
                next_calls, text = _send(client, credentials, payload)
                if next_calls or text.strip().upper().strip(".!") != "OK":
                    return {"ok": False, "message": "Bild und Werkzeugaufruf erfolgreich. "
                            "Die Fortsetzung nach dem Werkzeugergebnis ist fehlgeschlagen."}
        messages = {"text": "Textantwort erfolgreich.",
                    "vision": "Farbe des Testbilds korrekt erkannt.",
                    "tools": "Strukturierter Werkzeugaufruf korrekt.",
                    "workflow": "Bild, Werkzeugaufruf und Fortsetzung nach dem Werkzeugergebnis erfolgreich."}
        return {"ok": True, "message": messages[kind], "provider": provider, "model": model}
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        hints = {400: "Modell-ID und unterstützte Funktionen prüfen.",
                 401: "API-Schlüssel abgelehnt.", 402: "API-Guthaben prüfen.",
                 403: "Zugriff verweigert.", 404: "Modell nicht gefunden.",
                 429: "Budget, Guthaben oder Anfragelimit prüfen."}
        return {"ok": False, "message": f"{provider} meldet HTTP {status}. " +
                hints.get(status, "Anbieter konnte den Test nicht ausführen.")}
    except httpx.RequestError:
        return {"ok": False, "message": f"{provider} ist nicht erreichbar. Verbindung prüfen."}
    except (ValueError, TypeError, KeyError, AttributeError):
        return {"ok": False, "message": "Die Testantwort konnte nicht ausgewertet werden."}
