"""Self-check against a fake gateway that answers in LiteLLM's response format.
python test_llmkit.py"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import llmkit

# Reply shape as returned by LiteLLM in JSON mode (Groq upstream), ids replaced.
REAL = {
    "id": "chatcmpl-00000000-0000-0000-0000-000000000000", "created": 1794668435,
    "model": "fallback-chat", "object": "chat.completion", "system_fingerprint": "fp_0000000000",
    "choices": [{"finish_reason": "stop", "index": 0, "message": {
        "content": "{\"kcal\":67,\"protein_g\":12}", "role": "assistant",
        "reasoning": "We need to output JSON with fields kcal and protein_g. Provide estimates: "
                     "100g low-fat quark ~ 67 kcal, protein ~ 12g. Provide JSON."}}],
    "usage": {"completion_tokens": 57, "prompt_tokens": 133, "total_tokens": 190,
              "completion_tokens_details": {"reasoning_tokens": 37}},
    "x_groq": {"id": "req_00000000000000000000000000", "seed": 0}, "service_tier": "auto",
}

replies, requests_seen = [], []


def reply(content, **extra):
    """REAL with the message content (and other message fields) swapped."""
    r = json.loads(json.dumps(REAL))
    msg = r["choices"][0]["message"]
    msg["content"] = content
    msg.pop("reasoning")
    msg.update(extra)
    return r


class Gateway(BaseHTTPRequestHandler):
    """Fake LiteLLM: records each request body and answers with the next queued reply."""

    def do_POST(self):
        """Serve one chat completion."""
        requests_seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        body = json.dumps(replies.pop(0)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Keep the test output quiet."""


server = HTTPServer(("127.0.0.1", 0), Gateway)
threading.Thread(target=server.serve_forever, daemon=True).start()
os.environ["LITELLM_URL"] = f"http://127.0.0.1:{server.server_port}/"
os.environ["LITELLM_MODEL"] = "fallback-chat"

# --- real reply: JSON mode on, default model from the environment
replies.append(REAL)
assert llmkit.chat_json("100 g low-fat quark", system="JSON only.") == {"kcal": 67, "protein_g": 12}
body = requests_seen[-1]
assert body["model"] == "fallback-chat" and body["response_format"] == {"type": "json_object"}, body
assert body["messages"][0] == {"role": "system", "content": "JSON only."}, body

# --- thinking block, code fence and chatter around the object
replies.append(reply("<think>work it out first</think>Here:\n```json\n{\"kcal\": 67}\n```\nDone."))
assert llmkit.chat_json("x") == {"kcal": 67}

# --- broken JSON: asked again with the parser error, second answer counts
replies += [reply("{\"kcal\": 67,"), reply("{\"kcal\": 67}")]
assert llmkit.chat_json("x") == {"kcal": 67}
retry = requests_seen[-1]["messages"]
assert retry[-2] == {"role": "assistant", "content": "{\"kcal\": 67,"}, retry
assert retry[-1]["role"] == "user" and "not valid JSON" in retry[-1]["content"], retry

# --- small slips are repaired without spending a retry
before = len(requests_seen)
replies.append(reply('{"title": "Jeans", "price_eur": NaN, "brand": , "tags": ["denim", "blue",],}'))
assert llmkit.chat_json("x") == {"title": "Jeans", "price_eur": None, "brand": None,
                                 "tags": ["denim", "blue"]}
assert len(requests_seen) == before + 1

# --- NaN on its own is valid for Python's parser, it still comes back as None
replies.append(reply('{"price_eur": NaN}'))
assert llmkit.chat_json("x") == {"price_eur": None}

# --- a valid reply is never touched by the repair
replies.append(reply('{"note": "a, } and NaN stay as they are"}'))
assert llmkit.chat_json("x") == {"note": "a, } and NaN stay as they are"}

# --- broken twice: gives up with LLMError instead of a bare JSONDecodeError
replies += [reply("not json"), reply("still not json")]
try:
    llmkit.chat_json("x")
    raise AssertionError("expected LLMError")
except llmkit.LLMError as e:
    assert "2 attempts" in str(e), e

# --- no object at all (bare number) counts as broken and is asked again
replies += [reply("42"), reply("{\"kcal\": 42}")]
assert llmkit.chat_json("x") == {"kcal": 42}

# --- empty content, answer only in the reasoning channel
replies.append(reply(None, reasoning_content="{\"kcal\": 67}"))
assert llmkit.chat_json("x") == {"kcal": 67}

# --- chat: plain text, explicit model, images as content parts, no JSON mode
replies.append(reply("<think>hm</think>  Three sentences of coaching.  "))
assert llmkit.chat("Plan?", model="vision", images=["QUJD"]) == "Three sentences of coaching."
body = requests_seen[-1]
assert body["model"] == "vision" and "response_format" not in body, body
assert body["messages"][0]["content"][1] == {
    "type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}, body

# --- chat: thinking cut off by the length limit is dropped up to the end
replies.append(reply("Short: <think>the user wants {\"a\": 1"))
assert llmkit.chat("x") == "Short:"

# --- chat: empty answer is an error, not an empty string
replies.append(reply(None))
try:
    llmkit.chat("x")
    raise AssertionError("expected LLMError")
except llmkit.LLMError:
    pass

# --- missing configuration is reported before any request
del os.environ["LITELLM_MODEL"]
try:
    llmkit.chat("x")
    raise AssertionError("expected LLMError")
except llmkit.LLMError as e:
    assert "LITELLM_MODEL" in str(e), e

assert not replies, replies
server.shutdown()
print("ok")
