"""Runs episodes with the local model as concierge and collects training data."""
# Import unsloth FIRST before any trl/transformers to avoid import-order warnings
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import json
import os
import re
import time

import requests
import torch

from model_loader import get_model_and_tokenizer

MAX_STEPS_PER_EPISODE = 15

FALLBACK_REQUESTS = [
    "I need a Vegan meal under $40 with a flexible refund policy.",
    "Get me something Halal, budget is $50, must be refundable.",
    "I want to order food, no dietary restrictions, cheapest option possible.",
    "I need a Keto meal under $30 — also I have a dog so pet-friendly is a must.",
    "Looking for Gluten-Free options, budget $100, strictly refundable please.",
    "Order me anything under $20, No Restrictions on diet.",
    "I want a Vegan and Nut-Free meal, budget is $80, flexible returns preferred.",
]

CONCIERGE_SYSTEM_PROMPT = (
    "You are an E-Commerce AI Concierge. Fulfill the user's food ordering request by calling tools.\n\n"
    "TOOL FORMAT — Output ONLY a JSON object (nothing else) to call a tool:\n"
    '- List merchants:  {"tool": "getMerchant"}\n'
    '- Check merchant:  {"tool": "ask_watchdog", "merchant_name": "NAME"}\n'
    '- Place order:     {"tool": "place_order", "merchant_name": "NAME", "payload": {"field": "value"}}\n\n'
    "RULES:\n"
    "1. Always call ask_watchdog BEFORE place_order for any merchant.\n"
    "2. The place_order payload must contain EXACTLY the fields in ask_watchdog's required_fields list.\n"
    "3. Check price / refund / diet policies match the user's constraints.\n"
    "4. Invent any missing details (name, address, contact) — do NOT ask the user.\n"
    "5. If place_order fails with a field error, fix the payload and retry immediately.\n"
    "6. When the order is placed or all options are exhausted, write a plain text summary (not JSON)."
)

# Model tool name → environment server tool name
_TOOL_NAME_MAP = {
    "getMerchant": "get_merchants",
    "check_merchant": "ask_watchdog",
    "ask_watchdog": "ask_watchdog",
    "place_order": "place_order",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_prompt(messages: list[dict]) -> str:
    """Format a message list as a chat string ending with the assistant turn opener."""
    _, tokenizer = get_model_and_tokenizer()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


def _generate(prompt: str, max_new_tokens: int = 256) -> str:
    """Run inference with the local model and return the new text only."""
    model, tokenizer = get_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _parse_tool_call(text: str) -> dict | None:
    """Extract the first balanced JSON object from model output.

    Uses brace counting instead of a flat regex so nested dicts (place_order
    payloads) are parsed correctly.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_reward(obs: dict, response_json: dict) -> float:
    """Extract reward from either observation or top-level response payload."""
    raw = obs.get("reward")
    if raw is None:
        raw = response_json.get("reward")
    if raw is None:
        return 0.0
    return float(raw)


def _execute_tool(tool_call: dict, server_base_url: str) -> tuple[float, str, bool]:
    """Execute a parsed tool call via the environment server.

    Maps model-facing tool names to environment-server tool names,
    then POSTs to /step and returns (reward, observation_data, episode_done).
    """
    model_tool = tool_call.get("tool", "")
    env_tool = _TOOL_NAME_MAP.get(model_tool, model_tool)  # fix getMerchant → get_merchants

    merchant_name = tool_call.get("merchant_name")
    if merchant_name is None and isinstance(tool_call.get("merchant_names"), list):
        merchant_list = tool_call.get("merchant_names") or []
        merchant_name = merchant_list[0] if merchant_list else None

    action = {
        "tool": env_tool,
        "merchant_name": (
            "directory"                                  # getMerchant placeholder
            if model_tool == "getMerchant"
            else (merchant_name or "unknown")
        ),
        "payload": tool_call.get("payload") or {},
    }

    try:
        resp = requests.post(
            f"{server_base_url}step",
            json={"action": action},
            timeout=15,
        )
        payload = resp.json()
        obs = payload.get("observation", {})
        reward = _extract_reward(obs, payload)
        data = obs.get("data", "")
        done = bool(obs.get("done", False))
        return reward, data, done
    except Exception as e:
        return -5.0, f"[HTTP ERROR] {e}", False


def _get_persona_request(server_base_url: str, fallback_idx: int) -> str:
    """Get a persona request via OpenRouter, or use a hardcoded fallback."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()

    if api_key and model_name:
        try:
            from personaAgent import persona_node
            result = persona_node({"messages": []})
            for msg in result.get("messages", []):
                if hasattr(msg, "content") and msg.content:
                    return msg.content
        except Exception as e:
            print(f"[ROLLOUT] Persona node failed ({e}), using fallback.", flush=True)
    else:
        if not api_key:
            print("[ROLLOUT] OPENROUTER_API_KEY not set — using fallback persona.", flush=True)
        if not model_name:
            print("[ROLLOUT] MODEL_NAME not set — using fallback persona.", flush=True)

    return FALLBACK_REQUESTS[fallback_idx % len(FALLBACK_REQUESTS)]


# ── main public function ──────────────────────────────────────────────────────

def collect_rollouts(episodes: int, server_base_url: str) -> list[dict]:
    """Run episodes and collect (prompt, completion, reward) per concierge step."""
    rollout_buffer: list[dict] = []

    for ep in range(episodes):
        print(f"[ROLLOUT] Episode {ep + 1}/{episodes}", flush=True)

        try:
            requests.post(f"{server_base_url}reset", timeout=10)
        except Exception as e:
            print(f"[ROLLOUT] Reset failed: {e}", flush=True)
            continue

        persona_request = _get_persona_request(server_base_url, ep)
        print(f"[ROLLOUT]   Persona: {persona_request[:80]}", flush=True)

        messages = [
            {"role": "system", "content": CONCIERGE_SYSTEM_PROMPT},
            {"role": "user", "content": persona_request},
        ]

        episode_reward = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
            prompt = _build_prompt(messages)
            completion = _generate(prompt)

            # Always log what the model actually said (trimmed)
            print(f"[ROLLOUT]   step {step + 1} model output: {completion[:120]!r}", flush=True)

            tool_call = _parse_tool_call(completion)
            if tool_call is None:
                print(f"[ROLLOUT]   step {step + 1}: no tool call → episode end (reward 0)", flush=True)
                rollout_buffer.append({"prompt": prompt, "completion": completion, "reward": 0.0})
                break

            reward, obs_data, done = _execute_tool(tool_call, server_base_url)
            episode_reward += reward
            print(
                f"[ROLLOUT]   step {step + 1}: tool={tool_call.get('tool')!r} → "
                f"reward={reward:.1f} done={done}",
                flush=True,
            )

            rollout_buffer.append({"prompt": prompt, "completion": completion, "reward": reward})

            messages.append({"role": "assistant", "content": completion})
            messages.append({"role": "user", "content": f"[Tool Result]: {obs_data}"})

            if done:
                break

        print(f"[ROLLOUT] Episode {ep + 1} total reward: {episode_reward:.1f}", flush=True)

    return rollout_buffer
