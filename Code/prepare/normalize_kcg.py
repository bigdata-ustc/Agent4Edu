import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.relation_graph import normalize_kcg_pairs
from utils import load_json, save_json


def load_edge_pairs(path: Path) -> list[list[int]]:
    """Load edge pairs from JSON or RCD-style TSV/TXT files."""
    if path.suffix.lower() in {".json", ".jsonl"}:
        raw = load_json(path)
        if isinstance(raw, dict):
            for key in ("edges", "kcg", "pairs"):
                if key in raw:
                    raw = raw[key]
                    break
        return normalize_kcg_pairs(raw)

    pairs: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if len(parts) < 2:
                continue
            pairs.append([int(parts[0]), int(parts[1])])
    return pairs


def main():
    p = argparse.ArgumentParser(description="Normalize an RCD/KCG relation-pair file into Agent4Edu kcg.json format.")
    p.add_argument("--input", type=Path, required=True, help="input JSON, knowledgeGraph.txt, K_Directed.txt, or K_Undirected.txt")
    p.add_argument("--output", type=Path, required=True, help="output kcg.json")
    args = p.parse_args()
    pairs = load_edge_pairs(args.input)
    save_json(args.output, pairs)
    print(f"Saved {len(pairs)} KCG edges to {args.output}")


if __name__ == "__main__":
    main()
