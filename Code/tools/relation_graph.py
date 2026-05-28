from pathlib import Path
from typing import Any, Iterable, Sequence

from utils import load_json, normalize_concept, pick_distractors


class RelationGraph:
    """Knowledge Concept Graph (KCG) wrapper.

    The paper uses the KCG/RCD-derived relation graph to judge whether two
    concepts are related when reinforcing memory. The graph can be built with
    ``Code/tools/rcd_graph/build_kcg.py``, which refactors the RCD
    ``data/ASSIST/graph`` concept-map utility into an Agent4Edu tool.
    """

    def __init__(self, data_path: str | Path):
        data_path = Path(data_path)
        self.kcg_pairs = {tuple(pair) for pair in load_json(data_path / "kcg.json")}
        self.know_name = {normalize_concept(k): int(v) for k, v in load_json(data_path / "know_name_list.json").items()}
        self.know_course = {normalize_concept(k): str(v) for k, v in load_json(data_path / "know_course_list.json").items()}
        self.concepts = list(self.know_name.keys())

    def concept_id(self, concept: str) -> int | None:
        return self.know_name.get(normalize_concept(concept))

    def course(self, concept: str) -> str | None:
        return self.know_course.get(normalize_concept(concept))

    def is_related(self, left: str, right: str) -> bool:
        left_id = self.concept_id(left)
        right_id = self.concept_id(right)
        if left_id is None or right_id is None:
            return False
        return (left_id, right_id) in self.kcg_pairs or (right_id, left_id) in self.kcg_pairs

    def same_course(self, left: str, right: str) -> bool:
        return self.course(left) is not None and self.course(left) == self.course(right)

    def sample_distractors(self, true_concept: str, k: int = 2, seed: int | None = None) -> list[str]:
        return pick_distractors(true_concept, self.concepts, k=k, seed=seed)


def normalize_kcg_pairs(raw_pairs: Iterable[Sequence[Any]]) -> list[list[int]]:
    out: list[list[int]] = []
    for pair in raw_pairs:
        if len(pair) != 2:
            continue
        out.append([int(pair[0]), int(pair[1])])
    return out
