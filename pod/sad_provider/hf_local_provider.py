"""evalugator provider for local Hugging Face models (docs/behavioural-anchor-registration.md).

SAD runs only models its providers host, and its README prescribes a provider module in the clone's
`providers/` directory for any other model. This is that module and nothing more: it turns
evalugator's GetTextRequest (a list of role/content messages, temperature, max_tokens) into text.
SAD's items, templates, answer parsing and scoring are untouched.

- Model ids are the bare names of SAD's models.yaml entries (for example `qwen3-0.6b`). SAD looks the
  answer-assist setting up by the id, which must therefore equal the name; the repo and revision
  come from MODELS below, the same weights as the stored probe arrays.
- Rendering: the request's messages through the model's own chat template (Qwen3 and Qwen3.5 with
  enable_thinking=False, the date pinned as in the probe pipeline). Trailing assistant messages, such
  as evalugator's "Answer: (" prefix, are appended verbatim after the generation prompt as a prefill.
- Decoding: greedy (the stages templates ask for temperature 0), at most request.max_tokens new
  tokens, bf16 weights, no quantisation. A request with temperature > 0 is refused rather than
  silently sampled.
"""

from __future__ import annotations

import os

from evalugator.api.requests import GetTextRequest, GetTextResponse

PINNED_DATE = "01 Jan 2025"
# name -> (repo, revision), the revisions recorded in the stored probe arrays
MODELS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "qwen3-4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "qwen3-14b": ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18"),
    "qwen3-32b": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "qwen3.5-2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "qwen3.5-27b": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"),
    "olmo-2-0425-1b-instruct": ("allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e"),
    "olmo-2-1124-7b-instruct": ("allenai/OLMo-2-1124-7B-Instruct", "470b1fba1ae01581f270116362ee4aa1b97f4c84"),
    "olmo-2-1124-13b-instruct": ("allenai/OLMo-2-1124-13B-Instruct", "3a5c85baefbb1896a54d56fe2e76c0395627ddf4"),
    "olmo-2-0325-32b-instruct": ("allenai/OLMo-2-0325-32B-Instruct", "b96024342a77a69aa0dda815c3454a671f477463"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
_LOADED: dict = {}


def template_kwargs(name: str) -> dict:
    """The non-thinking rendering of the probe pipeline for the hybrid families."""
    kw = {"date_string": PINNED_DATE}
    if name.startswith(("qwen3-", "qwen3.5-")):
        kw["enable_thinking"] = False
    return kw


def load(name: str):
    """-> (tokenizer, model, device); cached, so one load serves every request of a run."""
    if name not in _LOADED:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        repo, rev = MODELS[name]
        cpu = os.environ.get("SAD_PROVIDER_DEVICE") == "cpu" or not torch.cuda.is_available()
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=None if cpu else "cuda").eval()
        _LOADED[name] = (tok, model, "cpu" if cpu else "cuda")
    return _LOADED[name]


def render(name: str, tok, messages) -> str:
    """Chat-template the messages; trailing assistant messages become a verbatim prefill."""
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    prefill = ""
    while msgs and msgs[-1]["role"] == "assistant":
        prefill = msgs.pop()["content"] + prefill
    kw = template_kwargs(name)
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text.replace("\r\n", "\n") + prefill


def provides_model(model_id: str) -> bool:
    return model_id in MODELS


def execute(model_id: str, request):
    if not isinstance(request, GetTextRequest):
        raise NotImplementedError("stages_oversight issues text requests only")
    if request.temperature and request.temperature > 0:
        raise NotImplementedError("only greedy decoding (temperature 0) is implemented")
    import torch
    tok, model, device = load(model_id)
    text = render(model_id, tok, request.prompt)
    enc = tok(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**enc, do_sample=False, max_new_tokens=int(request.max_tokens),
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return GetTextResponse(model_id=model_id, request=request, txt=txt, raw_responses=[],
                           context=None)
