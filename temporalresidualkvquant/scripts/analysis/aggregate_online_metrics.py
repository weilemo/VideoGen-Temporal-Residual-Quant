#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate E0 per-config summaries")
    parser.add_argument("--analysis", action="append", required=True, metavar="CONFIG=DIR")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for value in args.analysis:
        if "=" not in value:
            parser.error(f"invalid --analysis {value!r}; expected CONFIG=DIR")
        config, directory = value.split("=", 1)
        summary = json.loads((Path(directory).expanduser() / "summary.json").read_text(encoding="utf-8"))
        row = {"config": config, "pairs": summary["pairs"], "prompts": summary["prompts"]}
        for field, values in summary["absolute_metrics"].items():
            row[f"median_{field}"] = values["prompt_median"]
            row[f"{field}_ci_low"] = values["bootstrap_95_ci"][0]
            row[f"{field}_ci_high"] = values["bootstrap_95_ci"][1]
        boundary = summary.get("boundary_jump")
        row["median_boundary_jump"] = boundary["prompt_median"] if boundary else None
        rows.append(row)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated {len(rows)} E0 configs: {output}")


if __name__ == "__main__":
    main()
