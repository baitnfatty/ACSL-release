"""Loaders/adapters for public safety benchmarks (interfaces only).

Each line of an input ``.jsonl`` is an object with at least a ``prompt`` field;
optional ``id``, ``label``/``category``. Labels: ``0`` == benign/harmless, ``>=1``
== a harmful category index. These loaders never synthesize jailbreaks or
harmful text — they only map an existing benchmark file to ``Example`` objects.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class Example:
    id: str
    prompt: str
    label: int = 0                                # intent axis: 0=benign, 1=malicious
    severity: int = -1                             # severity axis: 0=inert, 1=moderate, 2=dangerous; -1=unlabeled
    split: str = "default"
    wrapper_family: Optional[str] = None           # wrapper family tag for held-out splits
    input_ids: Optional[list] = field(default=None)
    attention_mask: Optional[list] = field(default=None)
    meta: dict = field(default_factory=dict)


def _read_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlPromptLoader:
    """Iterate ``Example`` objects from a jsonl prompt file.

    Parameters
    ----------
    path : str | None
        Path to the benchmark jsonl. ``None`` => the loader is empty (raises on
        iteration) so a missing optional dataset fails loudly, not silently.
    split : str
        Split tag stamped on every yielded Example (used as the cache key).
    default_label : int
        Label used when a record carries neither ``label`` nor ``category``.
    limit : int | None
        Optional cap on number of examples (after deterministic file order).
    """

    def __init__(self, path: Optional[str], split: str, default_label: int = 0, limit: Optional[int] = None):
        self.path = path
        self.split = split
        self.default_label = int(default_label)
        self.limit = limit

    def __iter__(self) -> Iterator[Example]:
        if self.path is None:
            raise FileNotFoundError(
                f"no path configured for split {self.split!r}; supply a public benchmark jsonl"
            )
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"dataset not found: {self.path!r}")
        for i, rec in enumerate(_read_jsonl(self.path)):
            if self.limit is not None and i >= self.limit:
                break
            if "prompt" not in rec:
                raise ValueError(f"record {i} in {self.path!r} has no 'prompt' field")
            label = rec.get("label", rec.get("category", self.default_label))
            yield Example(
                id=str(rec.get("id", i)),
                prompt=str(rec["prompt"]),
                label=int(label),
                severity=int(rec.get("severity", -1)),
                split=self.split,
                wrapper_family=rec.get("wrapper_family"),
                meta={k: v for k, v in rec.items()
                      if k not in ("id", "prompt", "label", "category", "severity", "wrapper_family")},
            )


class HarmHarmlessLoader:
    """Yields harmful (label>=1) and matched harmless (label 0) prompts.

    Used to (a) build the diff-of-means caution direction and (b) train the head.
    """

    def __init__(self, harm_path, harmless_path, split: str = "train", limit: Optional[int] = None):
        self.harm = JsonlPromptLoader(harm_path, split, default_label=1, limit=limit)
        self.harmless = JsonlPromptLoader(harmless_path, split, default_label=0, limit=limit)

    def __iter__(self) -> Iterator[Example]:
        yield from self.harm
        yield from self.harmless


class JailbreakWrappedLoader(JsonlPromptLoader):
    """Jailbreak-wrapped harmful prompts (existing public benchmark, user-supplied).

    The repo does NOT generate these wrappers; it consumes a file of them to test
    whether the internal signal is invariant to surface-form wrapping.
    """

    def __init__(self, wrapped_path, split: str = "wrapped", limit: Optional[int] = None):
        super().__init__(wrapped_path, split, default_label=1, limit=limit)


class BenignSensitiveLoader(JsonlPromptLoader):
    """Benign-but-sensitive prompts, the over-refusal probe (label 0)."""

    def __init__(self, benign_sensitive_path, split: str = "benign", limit: Optional[int] = None):
        super().__init__(benign_sensitive_path, split, default_label=0, limit=limit)


def condition_examples(
    harm_path=None,
    harmless_path=None,
    wrapped_path=None,
    benign_sensitive_path=None,
    limit: Optional[int] = None,
    harmless_wrapped_only: bool = False,
) -> Iterator[Example]:
    """Yield Examples across the four eval conditions from user-supplied files.

    Splits: ``clean`` (harmful label>=1 + harmless label 0), ``wrapped``
    (jailbreak-wrapped harmful label>=1 + the SAME harmless negatives, so the
    kill test has two classes), ``benign`` (over-refusal probe). Conditions whose
    path is ``None`` are skipped.
    """
    if harm_path:
        yield from JsonlPromptLoader(harm_path, "clean", default_label=1, limit=limit)
    if harmless_path:
        for ex in JsonlPromptLoader(harmless_path, "clean", default_label=0, limit=limit):
            if not harmless_wrapped_only:
                yield ex
            yield Example(id=ex.id, prompt=ex.prompt, label=0, split="wrapped")
    if wrapped_path:
        yield from JailbreakWrappedLoader(wrapped_path, "wrapped", limit=limit)
    if benign_sensitive_path:
        yield from BenignSensitiveLoader(benign_sensitive_path, "benign", limit=limit)
