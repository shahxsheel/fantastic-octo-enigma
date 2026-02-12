#!/usr/bin/env python3
"""
Quick replay evaluator for JSONL infer results.

Input: one JSON object per line shaped like InferResult.to_dict().
Usage: python scripts/replay_eval.py --input results.jsonl
"""

import argparse
import json
from collections import Counter


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to JSONL infer results dump.")
    args = p.parse_args()

    total = 0
    locked = 0
    warn = 0
    alert = 0
    reason_counts: Counter[str] = Counter()
    score_sum = 0.0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            total += 1
            driver = d.get("driver") or {}
            risk = d.get("risk") or {}
            alerts = d.get("alerts") or {}

            if driver.get("locked"):
                locked += 1
            if alerts.get("warn"):
                warn += 1
            if alerts.get("alert"):
                alert += 1
            score_sum += float(risk.get("score", 0.0))
            for r in (risk.get("reason_codes") or []):
                reason_counts[str(r)] += 1

    if total == 0:
        print("No valid rows found.")
        return 1

    print(f"rows={total}")
    print(f"lock_ratio={locked/total:.4f}")
    print(f"warn_ratio={warn/total:.4f}")
    print(f"alert_ratio={alert/total:.4f}")
    print(f"mean_risk_score={score_sum/total:.3f}")
    print("top_reasons:")
    for reason, cnt in reason_counts.most_common(8):
        print(f"  {reason}: {cnt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
