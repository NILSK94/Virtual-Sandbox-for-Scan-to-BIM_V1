"""Bounded model context; complete results and renderings stay in the scene folder."""
import base64
import io
import json
import re
import uuid
import hashlib

from PIL import Image


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact(value, depth=0):
    if isinstance(value, dict):
        if depth >= 5:
            return {"omitted_keys": list(value)[:30]}
        return {key: ({"omitted_items": len(item)} if key in
                     {"vertices", "faces", "points", "colors", "source_indices", "overlays", "distances_m", "sample_ids", "worst_points"}
                     and isinstance(item, list) else compact(item, depth + 1))
                for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > 8:
            return {"total": len(value), "preview": [compact(item, depth + 1) for item in value[:8]],
                    "truncated": True}
        return [compact(item, depth + 1) for item in value]
    if isinstance(value, str) and len(value) > 800:
        return {"preview": value[:800], "characters": len(value), "truncated": True}
    return value


def pack_result(scene, value, limit=5000, label="tool_result"):
    encoded = dumps(value)
    if len(encoded) <= limit:
        return encoded
    folder = scene.folder / "agent-results"
    folder.mkdir(exist_ok=True)
    result_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
    (folder / (result_id + ".json")).write_text(encoded, encoding="utf-8")
    if hasattr(scene, "state"):
        refs = scene.state["agent"].setdefault("result_refs", [])
        refs[:] = [ref for ref in refs if ref["result_id"] != result_id]
        refs.append({"result_id": result_id, "source": label})
        del refs[:-8]
    preview = compact(value)
    if len(dumps(preview)) > limit - 500:
        # Keep valid JSON and an exact retrieval handle, never a cut JSON fragment.
        preview = {"keys": list(value)[:60]} if isinstance(value, dict) else {"type": type(value).__name__}
    return dumps({"truncated": True, "result_id": result_id, "preview": preview,
                  "retrieve": "read_agent_result: result_id, pointer (JSON Pointer), offset, limit. Full data retained locally."})


def read_result(scene, args):
    result_id = args.get("result_id", "")
    if not result_id:
        offset, limit = int(args.get("offset", 0)), int(args.get("limit", 8))
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("offset >= 0 und limit zwischen 1 und 100 verwenden.")
        paths = sorted((scene.folder / "agent-results").glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        return {"total": len(paths), "results": [{"result_id": p.stem, "bytes": p.stat().st_size}
                                                  for p in paths[offset:offset + limit]]}
    if not isinstance(result_id, str) or not re.fullmatch(r"[a-f0-9]{32}", result_id):
        raise ValueError("Ungueltige Ergebnis-ID.")
    offset, limit = int(args.get("offset", 0)), int(args.get("limit", 8))
    if offset < 0 or not 1 <= limit <= 4000:
        raise ValueError("offset >= 0 und limit zwischen 1 und 4000 verwenden.")
    pointer = args.get("pointer", "")
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("JSON Pointer muss leer sein oder mit / beginnen.")
    path = scene.folder / "agent-results" / (result_id + ".json")
    if not path.is_file():
        raise ValueError("Gespeichertes Ergebnis nicht gefunden.")
    value = json.loads(path.read_text(encoding="utf-8"))
    try:
        for part in pointer.split("/")[1:] if pointer else []:
            key = part.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
    except (KeyError, IndexError, TypeError, ValueError):
        raise ValueError("JSON Pointer wurde nicht gefunden.") from None
    if isinstance(value, (dict, list)) and limit > 100:
        raise ValueError("Bei Listen/Objekten hoechstens 100 Eintraege je Seite lesen.")
    if isinstance(value, dict):
        keys = list(value)
        page = {key: value[key] for key in keys[offset:offset + limit]}
        total = len(keys)
    elif isinstance(value, (list, str)):
        total = len(value)
        page = value[offset:offset + limit]
    else:
        total, page = 1, value
    # Large pages return a preview plus the SAME handle, so finer paths remain accessible.
    if len(dumps(page)) > 4000:
        preview = compact(page)
        if len(dumps(preview)) > 3500:
            preview = {"keys": list(page)[:60]} if isinstance(page, dict) else {"items": len(page)}
        return {"result_id": result_id, "pointer": pointer, "total": total,
                "truncated": True, "preview": preview,
                "hint": "Use a narrower pointer or smaller limit; string offsets count characters."}
    return {"result_id": result_id, "pointer": pointer, "offset": offset, "total": total,
            "next_offset": offset + limit if offset + limit < total else None, "value": page}


def scene_context(scene, remaining):
    state = scene.state
    elements = [{key: value for key, value in element.items()
                 if key not in {"vertices", "faces", "openings"}} for element in state["elements"]]
    context = {"remaining_steps": remaining, "elements": elements, "region": state["region"],
               "ifc": state["ifc"], "coordination": state["coordination"],
               "questions": state["questions"][-4:], "observations": state["observations"][-4:],
               "uncertainties": state.get("uncertainties", [])[-4:],
               "decisions": state["decisions"][-2:], "review": state["review"],
               "recent_results": state["agent"].get("result_memory", []),
               "experiments": [{k:e.get(k) for k in ("id","name","ok")} for e in state["experiments"][-6:]],
               "recent_result_refs": state["agent"].get("result_refs", []),
               "custom_tools": [{k: v for k, v in tool.items() if k != "code"}
                                for tool in state["custom_tools"]],
               "recent_fit_reports": [{k: v for k, v in report.items() if k != "worst_points"}
                                      for report in state["fit_reports"][-3:]]}
    packed = json.loads(pack_result(scene, context, limit=6500, label="scene_context"))
    # The complete step index is never cut by the preview or recent-round limits.
    packed["working_memory"] = state.get("working_memory", [])
    # Control information must remain immediately visible even for a large archived state.
    packed["remaining_steps"] = remaining
    packed["user_guidance"] = state.get("user_guidance", [])
    packed["reconstruction_plan"] = state.get("reconstruction_plan", {"overview":"","items":[]})
    packed["current_elements"] = [{"id": e["id"], "name": e.get("name"),
                                    "semantic": e.get("semantic", e.get("kind")),
                                    "locked": e.get("locked", False)} for e in state["elements"]]
    packed["workflow_feedback"] = state["agent"].get("workflow_feedback", {})
    packed["model_revision"] = state["model_revision"]
    # Keep local comparison evidence visible even when the general state is archived.
    packed["latest_comparisons"] = [{k: report.get(k) for k in
        ("id", "element_id", "element_signature", "ifc_geometry", "metrics", "region", "sampled")}
        for report in state["fit_reports"][-3:]]
    packed["continuation"] = "Continue investigating unmodeled scan regions; prior partial conclusions are revisable."
    packed["memory_note"] = "Every step is retained. Use read_agent_result with its result_id; /arguments and /result contain exact data. inspect_scene retrieves all decisions, observations and experiments."
    return dumps(packed)


def tool_index(catalog):
    """Expose the complete tool contracts, including arguments, on every model turn."""
    return dict(catalog)


def restore_working_memory(scene, catalog):
    folder = scene.folder / "agent-results"
    folder.mkdir(exist_ok=True)
    ledger = []
    for event in scene.events:
        if event["tool"] not in catalog:
            continue
        archive = {"tool": event["tool"], "arguments": event.get("args", {}), "result": event.get("result")}
        content = dumps(archive)
        result_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
        path = folder / (result_id + ".json")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        ledger.append({"step": len(ledger)+1,"tool":event["tool"],"result_id":result_id})
    scene.state["working_memory"] = ledger
    scene.save()


def remember_result(scene, name, args, result, encoded):
    archive = {"tool": name, "arguments": args, "result": result}
    content = dumps(archive)
    result_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
    folder = scene.folder / "agent-results"
    folder.mkdir(exist_ok=True)
    (folder / (result_id + ".json")).write_text(content, encoding="utf-8")
    ledger = scene.state.setdefault("working_memory", [])
    entry = {"step": len(ledger)+1, "tool": name, "result_id": result_id}
    if isinstance(result, dict):
        for key in ("id", "name", "ok", "error"):
            if key in result:
                entry[key] = str(result[key])[:120] if isinstance(result[key],str) else result[key]
    ledger.append(entry)
    scene.save()
    if name in {"tool_help", "request_view", "record_decision", "remember", "finish"}:
        return
    receipt = {"tool": name, "args": {k:v for k,v in args.items() if k != "code"}}
    summary = json.loads(encoded)
    if isinstance(summary, dict):
        if summary.get("result_id"):
            receipt["result_id"] = summary["result_id"]
        receipt["summary"] = compact(summary.get("preview", summary))
    if len(dumps(receipt)) > 1400:
        receipt.pop("summary", None)
        if isinstance(result, dict):
            receipt["result"] = {k: result[k] for k in ("id", "name", "ok", "error") if k in result}
    memory = scene.state["agent"].setdefault("result_memory", [])
    memory[:] = [item for item in memory if not (item["tool"] == name and item["args"] == receipt["args"])]
    memory.append(receipt)
    while len(memory) > 8 or (len(memory) > 1 and len(dumps(memory)) > 3500):
        memory.pop(0)


def image_message(scene, view):
    detail = view.get("model_image_detail", "high")
    with Image.open(scene.folder / view["file"]) as original:
        image = original.convert("RGB")
        image.thumbnail((512, 512) if detail == "low" else (1280, 1280))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_size = list(image.size)
    metadata = {k: v for k, v in view.items() if k not in {"source_hash"}}
    metadata["model_image_size"] = image_size
    metadata["note"] = "Original camera calibration refers to image_size; model_image_size is the resized transport image."
    return {"role": "user", "content": [
        {"type": "input_text", "text": "Aktuelle Ansicht: " + dumps(metadata)},
        {"type": "input_image", "detail": detail,
         "image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}]}


def model_input(initial, rounds, context, image):
    # Whole call/result rounds remain paired for OpenAI, Claude and Ollama.
    recent = list(rounds[-3:])
    while len(recent) > 1 and sum(len(dumps(r)) for r in recent) > 24000:
        recent.pop(0)
    return [initial, *(item for turn in recent for item in turn),
            {"role": "user", "content": "Aktueller Szenenstand: " + context}, image]


def record_usage(scene, data, provider, model, step):
    usage = data.get("usage") or {}
    if provider == "ollama":
        values = {"input_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")}
    else:
        values = {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                  "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens")}
        if provider == "claude":
            values["cached_input_tokens"] = usage.get("cache_read_input_tokens")
            values["cache_creation_input_tokens"] = usage.get("cache_creation_input_tokens")
    values = {key: value for key, value in values.items() if isinstance(value, int) and value >= 0}
    if not values:
        return
    with scene.lock:
        totals = scene.state["agent"].setdefault("token_usage", {"calls": 0})
        totals["calls"] += 1
        for key, value in values.items():
            totals[key] = totals.get(key, 0) + value
        scene.event("model_usage", {"provider": provider, "model": model, "step": step}, values, actor="system")
