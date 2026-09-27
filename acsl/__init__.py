"""ACSL — Attention-Coupled Security Loop.

A defensive safety-evaluation harness. The base LLM is *frozen*; only a small
SecurityHead is trained. The harness measures refusal-vs-compliance on existing
public safety benchmarks the user supplies. It does not generate jailbreaks or
harmful content.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
