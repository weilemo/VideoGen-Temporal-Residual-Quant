#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

TASKS = [
    ("subject_consistency", "subject consistency"),
    ("background_consistency", "background consistency"),
    ("motion_smoothness", "motion smoothness"),
    ("dynamic_degree", "dynamic degree"),
    ("aesthetic_quality", "aesthetic quality"),
    ("imaging_quality", "imaging quality"),
    ("overall_consistency", "overall consistency"),
    ("clip_score", "clip score"),
]

NORMALIZE = {
    "subject consistency": (0.1462, 1.0),
    "background consistency": (0.2615, 1.0),
    "motion smoothness": (0.706, 0.9975),
    "dynamic degree": (0.0, 1.0),
    "aesthetic quality": (0.0, 1.0),
    "imaging quality": (0.0, 1.0),
    "overall consistency": (0.0, 0.364),
    "clip score": (0.0, 0.3557),
}

DIM_WEIGHT = {
    "subject consistency": 1.0,
    "background consistency": 1.0,
    "motion smoothness": 1.0,
    "dynamic degree": 0.5,
    "aesthetic quality": 1.0,
    "imaging quality": 1.0,
    "overall consistency": 1.0,
    "clip score": 1.0,
}


def scalar(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        return float(value[0])
    raise ValueError(f"cannot extract scalar from {type(value)}")


def norm(name, value):
    lo, hi = NORMALIZE[name]
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def weighted_mean(raw, dims):
    num = 0.0
    den = 0.0
    for dim in dims:
        num += norm(dim, raw[dim]) * DIM_WEIGHT[dim]
        den += DIM_WEIGHT[dim]
    return num / den


def load_result(input_dir, label):
    raw = {}
    for key, name in TASKS:
        fp = input_dir / f"{label}_{key}_eval_results.json"
        if not fp.exists():
            raise FileNotFoundError(fp)
        obj = json.loads(fp.read_text())
        if key not in obj:
            raise KeyError(f"{key} not in {fp}")
        raw[name] = scalar(obj[key])

    temporal = weighted_mean(
        raw,
        [
            "subject consistency",
            "background consistency",
            "motion smoothness",
            "dynamic degree",
        ],
    )
    frame = weighted_mean(raw, ["aesthetic quality", "imaging quality"])
    text = weighted_mean(raw, ["overall consistency", "clip score"])
    final = (2 * temporal + 2 * frame + text) / 5
    return {
        "label": label,
        "raw_scores": raw,
        "normalized_scores": {k: norm(k, v) for k, v in raw.items()},
        "temporal_quality": temporal,
        "frame_wise_quality": frame,
        "text_alignment": text,
        "final_score": final,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    rows = [load_result(input_dir, label) for label in args.label]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": rows}, indent=2, ensure_ascii=False))

    dims = [name for _, name in TASKS]
    headers = [
        "method",
        "subject",
        "background",
        "motion",
        "dynamic",
        "aesthetic",
        "imaging",
        "overall",
        "clip",
        "temporal",
        "frame",
        "text",
        "final",
    ]
    md = ["# MovieGenBench-128 VBench Summary", "", "| " + " | ".join(headers) + " |"]
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        vals = [f"{row['raw_scores'][d]:.4f}" for d in dims]
        vals.extend(
            [
                f"{row['temporal_quality']:.4f}",
                f"{row['frame_wise_quality']:.4f}",
                f"{row['text_alignment']:.4f}",
                f"{row['final_score']:.4f}",
            ]
        )
        md.append(f"| {row['label']} | " + " | ".join(vals) + " |")
    Path(args.markdown).write_text("\n".join(md) + "\n")
    print(json.dumps({r["label"]: r["final_score"] for r in rows}, indent=2))


if __name__ == "__main__":
    main()
