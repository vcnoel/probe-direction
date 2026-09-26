"""Prompt construction: wrap identical tasks in different situational frames.

The whole experiment hinges on holding the *content* constant and varying only
the *frame*. So a prompt is always ``frame(system) + frame(user_template).format(task)``,
rendered through the model's own chat template. Two conditions applied to the same
task differ by exactly the framing tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    task: str


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    system: str
    user_template: str

    def render_messages(self, task_text: str, user_prefix: str = "") -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_prefix + self.user_template.format(task=task_text)},
        ]


@dataclass(frozen=True)
class Prompt:
    task_id: str
    category: str
    condition: str
    text: str  # fully chat-templated string, ready for analyze_text
    task_text: str  # the byte-identical base task, for subgraph span location


def load_tasks(path: Optional[Path] = None) -> List[Task]:
    path = Path(path) if path else _DATA_DIR / "tasks.jsonl"
    tasks: List[Task] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tasks.append(Task(id=d["id"], category=d["category"], task=d["task"]))
    return tasks


def load_framings(path: Optional[Path] = None) -> Dict[str, Condition]:
    path = Path(path) if path else _DATA_DIR / "framings.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    conditions: Dict[str, Condition] = {}
    for key, spec in raw.items():
        if key.startswith("_"):
            continue
        conditions[key] = Condition(
            key=key,
            label=spec["label"],
            system=spec["system"],
            user_template=spec["user_template"],
        )
    return conditions


def apply_chat_template(tokenizer, messages: List[Dict[str, str]],
                        template_kwargs: Optional[Dict] = None) -> str:
    """Render messages to a raw string via the model's chat template.

    Some models (e.g. gemma) reject a separate ``system`` role. We fold the
    system content into the first user turn in that case so the frame still
    reaches the model.

    ``template_kwargs`` are passed to the template itself, for the two things a
    template can vary that are not the messages: a pinned ``date_string`` (some
    templates interpolate today's date, so rendering is otherwise not
    reproducible) and per-family switches such as Qwen3's ``enable_thinking``.
    A kwarg a template does not accept is dropped rather than raising, so one
    call site can serve every family.
    """
    kw = dict(template_kwargs or {})

    def render(msgs):
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **kw
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    try:
        return render(messages)
    except Exception:
        merged = messages
        if messages and messages[0]["role"] == "system":
            sys_txt = messages[0]["content"]
            rest = messages[1:]
            if rest:
                rest = [dict(rest[0]) for _ in [0]] + rest[1:]
                rest[0] = {
                    "role": rest[0]["role"],
                    "content": f"{sys_txt}\n\n{rest[0]['content']}",
                }
            merged = rest
        return render(merged)


# Neutral filler used to position-match frames. A CYCLE of low-content function
# words (not a single repeated token) so the padding carries ~no evaluative or
# deployment semantics AND does not create a degenerate block of identical tokens
# (which makes the full-prompt graph Laplacian singular). A paper should verify
# robustness to this choice with a second filler set.
_FILLER_WORDS = ["the", "a", "of", "and", "to", "in", "that", "it",
                 "is", "for", "as", "with", "on", "by", "at", "or"]


def _preamble_start(tokenizer, cond: Condition, user_prefix: str, probe_task: str,
                    max_length: int) -> int:
    """Token index at which the task begins — depends only on the preamble, not the
    task content, so any probe_task gives the right answer."""
    from .spans import locate_task_span, SpanError
    text = apply_chat_template(tokenizer, cond.render_messages(probe_task, user_prefix))
    try:
        return locate_task_span(tokenizer, text, probe_task, max_length).start
    except SpanError:
        return -1


def _position_match_prefixes(conditions: Dict[str, Condition], keys, tokenizer,
                             max_length: int, margin: int) -> Dict[str, str]:
    """Per-condition user_prefix (neutral function-word filler) so every condition's
    task starts at the SAME token index — removing absolute-position (RoPE) as a
    confound. Note: this trades position for a (small) filler-amount difference; it
    is cleanest for the length-matched eval-vs-deploy contrast (near-equal filler)."""
    probe = "PLACEHOLDER TASK CONTENT FOR ALIGNMENT."
    starts0 = {k: _preamble_start(tokenizer, conditions[k], "", probe, max_length) for k in keys}
    target = max(starts0.values()) + margin
    prefixes = {}
    for k in keys:
        words: List[str] = []
        while (_preamble_start(tokenizer, conditions[k], " ".join(words) + " ", probe, max_length)
               < target and len(words) < 8000):
            words.append(_FILLER_WORDS[len(words) % len(_FILLER_WORDS)])
        prefixes[k] = (" ".join(words) + " ") if words else ""
    return prefixes


def build_prompts(
    tasks: List[Task],
    conditions: Dict[str, Condition],
    tokenizer,
    only_conditions: Optional[List[str]] = None,
    position_match: bool = False,
    pos_margin: int = 2,
    template_kwargs: Optional[Dict] = None,
) -> List[Prompt]:
    """Produce one chat-templated Prompt per (task, condition).

    If ``position_match`` is set, each condition's preamble is padded with neutral
    function-word filler so the (byte-identical) task tokens begin at the same
    absolute token index in every frame — closing the RoPE/position shortcut a probe
    could exploit. Under position-matching the ``full``-scope graph is dominated by
    filler and not meaningful; analyze the ``task_subgraph`` cell.
    """
    keys = only_conditions or list(conditions.keys())
    prefixes = ({k: "" for k in keys} if not position_match
                else _position_match_prefixes(conditions, keys, tokenizer,
                                              max_length=512, margin=pos_margin))
    prompts: List[Prompt] = []
    for task in tasks:
        for key in keys:
            cond = conditions[key]
            messages = cond.render_messages(task.task, user_prefix=prefixes[key])
            text = apply_chat_template(tokenizer, messages, template_kwargs)
            prompts.append(
                Prompt(
                    task_id=task.id,
                    category=task.category,
                    condition=key,
                    text=text,
                    task_text=task.task,
                )
            )
    return prompts
