"""Anthropic Messages adapter for the studio's common tool conversation."""
import json

BASE_URL = "https://api.anthropic.com/v1"


def headers(key):
    return {"x-api-key": key, "anthropic-version": "2023-06-01"}


def chat_payload(payload):
    messages = []
    for item in payload["input"]:
        if item.get("type") == "function_call":
            role, content = "assistant", [{
                "type": "tool_use", "id": item["call_id"], "name": item["name"],
                "input": json.loads(item["arguments"]),
            }]
        elif item.get("type") == "function_call_output":
            role, content = "user", [{
                "type": "tool_result", "tool_use_id": item["call_id"], "content": item["output"],
            }]
        elif item.get("role") in ("user", "assistant"):
            role = item["role"]
            if isinstance(item["content"], str):
                content = [{"type": "text", "text": item["content"]}]
            else:
                content = []
                for part in item["content"]:
                    if part["type"] == "input_text":
                        content.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "input_image":
                        prefix, data = part["image_url"].split(",", 1)
                        content.append({"type": "image", "source": {
                            "type": "base64", "media_type": prefix[5:].split(";")[0], "data": data,
                        }})
        else:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(content)
        else:
            messages.append({"role": role, "content": content})
    result = {"model": payload["model"], "system": payload["instructions"],
              "messages": messages, "max_tokens": payload.get("max_output_tokens", 6000)}
    if payload.get("tools"):
        result["tools"] = [{"name": tool["name"], "description": tool["description"],
                            "input_schema": tool["parameters"]} for tool in payload["tools"]]
        result["tool_choice"] = {"type": "any", "disable_parallel_tool_use": True}
    return result


def chat_output(response):
    return [{"type": "function_call", "name": block["name"], "call_id": block["id"],
             "arguments": json.dumps(block["input"], ensure_ascii=False)}
            for block in response.get("content", []) if block.get("type") == "tool_use"]
