"""Build the Agent4Edu KCG artifact with the RCD concept-map rule.

Agent4Edu does not train the RCD model.  It only uses the graph-construction
utility released with RCD, namely the code under ``data/ASSIST/graph`` that
constructs a concept dependency map from student response logs.

This file refactors that utility into a reusable Agent4Edu tool:

1. read Agent4Edu/RCD-style response logs;
2. count transitions between correctly answered consecutive concepts;
3. normalize transition strengths with the original RCD threshold rule;
4. split reciprocal edges into similarity edges and one-way edges into
   prerequisite/dependency edges;
5. export ``kcg.json`` for Agent4Edu memory reinforcement.

The original RCD script assumes dense, 1-based concept ids and fixed dataset
sizes.  This version keeps the same construction logic but uses sparse Python
structures and preserves Agent4Edu's raw concept ids by default.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

RelationScope = Literal["all", "directed", "undirected"]


def _load_json(path: str | Path) -> Any:
    path = Path(path)
    last_error: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read JSON file {path}: {last_error}")


def _save_json(path: str | Path, data: Any, *, indent: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def _write_edge_txt(path: str | Path, edges: Iterable[tuple[int, int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for src, dst in sorted(edges):
            f.write(f"{src}\t{dst}\n")


def _as_knowledge_list(value: Any, *, id_offset: int = 0) -> list[int]:
    """Normalize one log's knowledge_code field to a list of int ids."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value + id_offset]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Support "1", "1,2", "1 2", or "[1, 2]" from different datasets.
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                return _as_knowledge_list(parsed, id_offset=id_offset)
            except json.JSONDecodeError:
                pass
        chunks = text.replace(",", " ").split()
        return [int(x) + id_offset for x in chunks]
    if isinstance(value, Sequence):
        out: list[int] = []
        for item in value:
            out.extend(_as_knowledge_list(item, id_offset=id_offset))
        # Preserve order while removing duplicates inside one exercise.
        return list(dict.fromkeys(out))
    raise TypeError(f"Unsupported knowledge_code value: {value!r}")


def _score_is_correct(log: Mapping[str, Any]) -> bool:
    for key in ("score", "correct", "is_correct", "label", "answer_correct"):
        if key in log:
            try:
                return int(float(log[key])) == 1
            except (TypeError, ValueError):
                return str(log[key]).strip().lower() in {"true", "yes", "correct"}
    return False


def iter_log_sequences(rows: Sequence[Mapping[str, Any]]) -> Iterator[list[Mapping[str, Any]]]:
    """Yield each student's ordered practice sequence.

    Supported inputs:
    - Agent4Edu: [{"user_id": ..., "logs": [...]}, ...]
    - RCD ASSIST: [{"log_num": ..., "logs": [...]}, ...]
    """
    for row in rows:
        logs = row.get("logs", [])
        if not isinstance(logs, list):
            continue
        if "log_num" in row:
            try:
                logs = logs[: int(row["log_num"])]
            except (TypeError, ValueError):
                pass
        if logs:
            yield logs


def count_correct_transitions(
    rows: Sequence[Mapping[str, Any]],
    *,
    knowledge_id_offset: int = 0,
) -> tuple[Counter[tuple[int, int]], Counter[int]]:
    """Count RCD-style concept transitions.

    For two consecutive logs, a transition from concept i to concept j is counted
    only when both exercises are answered correctly and i != j.  The denominator
    follows the original RCD implementation: every observed outgoing pair also
    increments the source concept's denominator.
    """
    pair_counts: Counter[tuple[int, int]] = Counter()
    source_denom: Counter[int] = Counter()

    for logs in iter_log_sequences(rows):
        if len(logs) < 2:
            continue
        for left, right in zip(logs[:-1], logs[1:]):
            if not (_score_is_correct(left) and _score_is_correct(right)):
                continue
            left_codes = _as_knowledge_list(left.get("knowledge_code"), id_offset=knowledge_id_offset)
            right_codes = _as_knowledge_list(right.get("knowledge_code"), id_offset=knowledge_id_offset)
            for src in left_codes:
                for dst in right_codes:
                    if src == dst:
                        continue
                    pair_counts[(src, dst)] += 1
                    source_denom[src] += 1
    return pair_counts, source_denom


def _rcd_threshold(strengths: Sequence[float]) -> float:
    """RCD threshold: min-max normalize, average, then power four."""
    if not strengths:
        return 1.0
    min_v = min(strengths)
    max_v = max(strengths)
    if max_v == min_v:
        return 0.0
    normalized = [(v - min_v) / (max_v - min_v) for v in strengths]
    avg = sum(normalized) / len(normalized)
    return avg**4


def build_concept_graph(
    rows: Sequence[Mapping[str, Any]],
    *,
    knowledge_id_offset: int = 0,
    min_transition_count: int = 1,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Build raw RCD concept-map edges from logs.

    Returns ``knowledgeGraph``-style edges before reciprocal-edge splitting.
    """
    pair_counts, source_denom = count_correct_transitions(
        rows,
        knowledge_id_offset=knowledge_id_offset,
    )
    strengths: dict[tuple[int, int], float] = {}
    for edge, count in pair_counts.items():
        if count < min_transition_count:
            continue
        src, _ = edge
        denom = source_denom[src]
        if denom > 0:
            strengths[edge] = float(count) / float(denom)

    threshold = _rcd_threshold(list(strengths.values()))

    if strengths:
        min_v = min(strengths.values())
        max_v = max(strengths.values())
    else:
        min_v = max_v = 0.0

    selected: list[tuple[int, int]] = []
    for edge, value in strengths.items():
        if max_v == min_v:
            normalized = 1.0
        else:
            normalized = (value - min_v) / (max_v - min_v)
        if normalized >= threshold:
            selected.append(edge)

    concept_ids = sorted({k for edge in pair_counts for k in edge})
    stats = {
        "concept_count": len(concept_ids),
        "transition_pair_count": len(pair_counts),
        "candidate_edge_count": len(strengths),
        "selected_edge_count": len(selected),
        "min_transition_count": min_transition_count,
        "threshold": threshold,
        "min_strength": min_v,
        "max_strength": max_v,
        "knowledge_id_offset": knowledge_id_offset,
    }
    return sorted(selected), stats


def split_directed_undirected(edges: Iterable[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split RCD raw graph into prerequisite and similarity edges.

    Reciprocal pairs are written once as undirected/similarity edges.  One-way
    pairs are written as directed/dependency edges.  This mirrors RCD's
    ``process_edge.py`` behavior while making the output deterministic.
    """
    edge_set = {tuple(map(int, e)) for e in edges}
    visited: set[tuple[int, int]] = set()
    directed: list[tuple[int, int]] = []
    undirected: list[tuple[int, int]] = []

    for src, dst in sorted(edge_set):
        if (src, dst) in visited:
            continue
        if (dst, src) in edge_set:
            undirected.append((src, dst))
            visited.add((src, dst))
            visited.add((dst, src))
        else:
            directed.append((src, dst))
            visited.add((src, dst))
    return directed, undirected


def select_agent_edges(
    raw_edges: Sequence[tuple[int, int]],
    directed_edges: Sequence[tuple[int, int]],
    undirected_edges: Sequence[tuple[int, int]],
    *,
    relation_scope: RelationScope,
) -> list[tuple[int, int]]:
    if relation_scope == "all":
        return sorted(set(raw_edges))
    if relation_scope == "directed":
        return sorted(set(directed_edges))
    if relation_scope == "undirected":
        return sorted(set(undirected_edges))
    raise ValueError(f"Unknown relation_scope: {relation_scope}")


def export_graph_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    agent_kcg_path: str | Path | None = None,
    relation_scope: RelationScope = "all",
    knowledge_id_offset: int = 0,
    min_transition_count: int = 1,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    raw_edges, stats = build_concept_graph(
        rows,
        knowledge_id_offset=knowledge_id_offset,
        min_transition_count=min_transition_count,
    )
    directed_edges, undirected_edges = split_directed_undirected(raw_edges)
    agent_edges = select_agent_edges(raw_edges, directed_edges, undirected_edges, relation_scope=relation_scope)

    _write_edge_txt(output_dir / "knowledgeGraph.txt", raw_edges)
    _write_edge_txt(output_dir / "K_Directed.txt", directed_edges)
    _write_edge_txt(output_dir / "K_Undirected.txt", undirected_edges)
    _save_json(output_dir / "kcg.json", [[a, b] for a, b in agent_edges])

    if agent_kcg_path is not None:
        _save_json(agent_kcg_path, [[a, b] for a, b in agent_edges])

    stats = {
        **stats,
        "directed_edge_count": len(directed_edges),
        "undirected_edge_count": len(undirected_edges),
        "agent_edge_count": len(agent_edges),
        "agent_relation_scope": relation_scope,
    }
    _save_json(output_dir / "graph_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Agent4Edu kcg.json using the RCD data/ASSIST/graph concept-map construction rule."
    )
    parser.add_argument("--logs", type=Path, required=True, help="Agent4Edu stu_logs.json or RCD log_data_all.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for knowledgeGraph/K_Directed/K_Undirected/kcg")
    parser.add_argument("--agent-kcg", type=Path, default=None, help="optional final Agent4Edu data/*/kcg.json path")
    parser.add_argument(
        "--relation-scope",
        choices=["all", "directed", "undirected"],
        default="all",
        help="which RCD relation set to export to Agent4Edu kcg.json; default keeps the current Agent4Edu behavior",
    )
    parser.add_argument(
        "--knowledge-id-offset",
        type=int,
        default=0,
        help="add this value to every knowledge_code. Use -1 to reproduce the original RCD ASSIST 1-based-to-0-based scripts.",
    )
    parser.add_argument(
        "--min-transition-count",
        type=int,
        default=1,
        help="drop concept-pair candidates observed fewer than this number before thresholding",
    )
    args = parser.parse_args()

    rows = _load_json(args.logs)
    if not isinstance(rows, list):
        raise ValueError("The log file must be a JSON list of student records.")

    stats = export_graph_artifacts(
        rows,
        output_dir=args.output_dir,
        agent_kcg_path=args.agent_kcg,
        relation_scope=args.relation_scope,
        knowledge_id_offset=args.knowledge_id_offset,
        min_transition_count=args.min_transition_count,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
