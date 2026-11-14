# llmkit

Small client for an OpenAI-compatible [LiteLLM](https://github.com/BerriAI/litellm) gateway.
It covers the two calls our apps make over and over: a plain chat answer and a JSON object.

## Install

Pin a tag in `requirements.txt`:

```
llmkit @ https://github.com/food/llmkit/archive/refs/tags/v0.3.0.tar.gz
```

## Configuration

| Variable | Meaning |
|---|---|
| `LITELLM_URL` | Gateway base URL, e.g. `http://litellm:4000` |
| `LITELLM_MODEL` | Default model or lane; `model=` overrides it per call |

Both are read on every call.

## Usage

```python
import llmkit

text = llmkit.chat("Summarise this week in three sentences.", system="You are a coach.")
data = llmkit.chat_json("Estimate kcal and protein for 100 g quark.", system="Reply in JSON.")
tags = llmkit.chat_json("Describe the item.", model="vision", images=[jpeg_base64])
plan = llmkit.chat_json(prompt, system=rules, check=Plan.model_validate, max_tokens=800)
```

`chat_json` turns on JSON mode and copes with what models actually send back:
`<think>` blocks, code fences, text around the object, trailing commas, empty values,
`NaN`, and an empty answer with the JSON in the reasoning channel. If the reply still
does not parse, the model gets the parser error and one more try (`retries=1`).

`check` receives the parsed object and returns what the caller gets instead. A
`ValueError` from it, such as Pydantic's `ValidationError`, is fed back to the model
like a parser error.

Both functions raise `llmkit.LLMError` when no usable answer comes back, including
network errors and HTTP errors from the gateway.

## Tests

```
python test_llmkit.py
```
