#!/usr/bin/env python
"""05_prereg - freeze the prediction, decision rule, and kill criterion.

Writes prereg.json (with a content hash of the eval sets and a self-hash) BEFORE
the held-out eval is run. Run this after 04 and before 06/07.

    python scripts/05_prereg.py --config configs/default.yaml
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acsl.prereg import freeze_prereg
from acsl.runtime import build_parser, resolve, write_manifest


def main() -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args()
    cfg, out = resolve(args)

    eval_paths = [
        cfg.get_path("data.harm_path"), cfg.get_path("data.harmless_path"),
        cfg.get_path("data.wrapped_path"), cfg.get_path("data.benign_sensitive_path"),
    ]
    prereg_path = os.path.join(out, "prereg.json")
    prereg = freeze_prereg(cfg, out_path=prereg_path, eval_paths=eval_paths)

    print(f"[prereg] prediction:\n  {prereg['prediction']}")
    print(f"[prereg] decision rule: {json.dumps(prereg['decision_rule'])}")
    print(f"[prereg] kill criterion: kill_delta_max={prereg['kill_criterion']['kill_delta_max']}")
    print(f"[prereg] sha256={prereg['prereg_sha256'][:16]}... -> {prereg_path}")
    write_manifest(out, cfg, eval_paths=eval_paths, extra={"prereg_sha256": prereg["prereg_sha256"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
