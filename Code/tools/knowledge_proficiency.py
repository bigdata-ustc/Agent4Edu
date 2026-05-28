from pathlib import Path
from typing import Any

from utils import load_json, normalize_concept


class KnowledgeProficiency:
    """DNeuralCDM-produced dynamic knowledge proficiency loader.

    Expected JSON shape is compatible with the old script:
    ``{student_id: [[[k_0, k_1, ...], ...]]}``, where the outer one-element list
    stores the sequence for the student.
    """

    def __init__(self, data_path: str | Path):
        data_path = Path(data_path)
        self.file = data_path / "stu_know_proficiency.json"
        self.data: dict[str, Any] = {}
        if self.file.exists():
            self.data = load_json(self.file)
        know_file = data_path / "know_name_list.json"
        self.know_name = {normalize_concept(k): int(v) for k, v in load_json(know_file).items()} if know_file.exists() else {}

    def available(self) -> bool:
        return bool(self.data)

    def _sequence(self, student_id: int) -> list[list[float]] | None:
        value = self.data.get(str(student_id))
        if value is None:
            return None
        if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
            return value[0]
        return value

    def value(self, student_id: int, concept: str, time_step: int) -> float | None:
        seq = self._sequence(student_id)
        if not seq:
            return None
        concept_id = self.know_name.get(normalize_concept(concept))
        if concept_id is None:
            return None
        index = max(0, min(len(seq) - 1, time_step - 1))
        if concept_id >= len(seq[index]):
            return None
        return float(seq[index][concept_id])

    @staticmethod
    def tier(value: float | None) -> str:
        if value is None:
            return "unknown"
        if value > 0.66:
            return "good"
        if value > 0.33:
            return "common"
        return "poor"
