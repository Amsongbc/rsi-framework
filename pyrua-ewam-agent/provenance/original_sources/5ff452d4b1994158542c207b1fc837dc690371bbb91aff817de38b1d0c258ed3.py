#!/usr/bin/env python3
"""Summarize only saved real simulator verdicts and matched experiment pairs."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(root):
    groups = defaultdict(list)
    matched = defaultdict(dict)
    for path in sorted(root.glob("*/result.json")):
        row = json.loads(path.read_text())
        if row.get("benchmark_episode") is False or row["mode"] == "smoke":
            continue
        row["result_path"] = str(path)
        groups[(row["task"], row["mode"])].append(row)
        if row["valid"]:
            key = (row["task"], row["eval_seed"], row["layout"])
            if row["mode"] in matched[key]:
                raise ValueError(f"Duplicate valid run for {key} {row['mode']}; do not select outcomes")
            matched[key][row["mode"]] = row
    aggregate = []
    for (task, mode), attempted in sorted(groups.items()):
        valid = [r for r in attempted if r["valid"]]
        means = {}
        for field in ["client_score", "ewam_calls", "primitive_calls", "native_actions_issued",
                      "startup_s", "control_wall_s", "wall_s"]:
            values = [r[field] for r in valid if r.get(field) is not None]
            means[field] = statistics.mean(values) if values else None
        aggregate.append({"task": task, "mode": mode, "attempted": len(attempted),
                          "valid": len(valid), "invalid": len(attempted) - len(valid),
                          "successes": sum(r["success"] for r in valid),
                          "success_rate": statistics.mean(r["success"] for r in valid) if valid else None,
                          "means": means})
    pairs = []
    for (task, seed, layout), modes in sorted(matched.items()):
        if set(modes) != {"baseline", "hybrid"}:
            continue
        base, hybrid = modes["baseline"], modes["hybrid"]
        pairs.append({"task": task, "eval_seed": seed, "layout": layout,
                      "baseline_success": base["success"], "hybrid_success": hybrid["success"],
                      "baseline_score": base["client_score"], "hybrid_score": hybrid["client_score"],
                      "baseline_native_steps": base["client_verdict"]["step"],
                      "hybrid_native_steps": hybrid["client_verdict"]["step"],
                      "baseline_control_wall_s": base["control_wall_s"],
                      "hybrid_control_wall_s": hybrid["control_wall_s"],
                      "baseline_ewam_calls": base["ewam_calls"], "hybrid_ewam_calls": hybrid["ewam_calls"],
                      "hybrid_primitive_calls": hybrid["primitive_calls"],
                      "baseline_result": base["result_path"], "hybrid_result": hybrid["result_path"]})
    return {"groups": aggregate, "matched_pairs": pairs,
            "paired_hybrid_only_success": sum(p["hybrid_success"] and not p["baseline_success"] for p in pairs),
            "paired_baseline_only_success": sum(p["baseline_success"] and not p["hybrid_success"] for p in pairs),
            "notes": ["Infrastructure errors are reported as invalid and are not task failures.",
                      "Current-window planner API token usage is unavailable.",
                      "Small matched samples do not establish a benchmark-wide success rate."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    result = summarize(args.runs)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text)
    if args.csv and result['matched_pairs']:
        with args.csv.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(result['matched_pairs'][0]))
            writer.writeheader()
            writer.writerows(result['matched_pairs'])
    print(text)


if __name__ == "__main__":
    main()
