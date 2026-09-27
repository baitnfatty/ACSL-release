"""Dataset adapters for public safety benchmarks.

No harmful content is bundled. Loaders read prompts the user supplies/obtains
(as jsonl) and the eval only classifies model responses as refusal vs compliance.
A tiny harmless synthetic set is provided for smoke tests.
"""

from .loaders import (
    Example,
    JsonlPromptLoader,
    HarmHarmlessLoader,
    JailbreakWrappedLoader,
    BenignSensitiveLoader,
    condition_examples,
)
from .synthetic_smoke import (
    synthetic_smoke_examples,
    synthetic_conditions,
    synthetic_benign_sensitive,
)

__all__ = [
    "Example",
    "JsonlPromptLoader",
    "HarmHarmlessLoader",
    "JailbreakWrappedLoader",
    "BenignSensitiveLoader",
    "condition_examples",
    "synthetic_smoke_examples",
    "synthetic_conditions",
    "synthetic_benign_sensitive",
]
