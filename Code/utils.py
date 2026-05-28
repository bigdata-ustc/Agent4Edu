import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence


ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "gbk")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: os.PathLike | str) -> Any:
    path = Path(path)
    last_error = None
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError as exc:
            last_error = exc
        except json.JSONDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read JSON file {path}: {last_error}")


def save_json(path: os.PathLike | str, data: Any, *, indent: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_config_numbers(config_path: os.PathLike | str) -> tuple[int, int, int]:
    """Read a two-line config file: header, then student_n,exercise_n,knowledge_n."""
    path = Path(config_path)
    text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if len(text) < 2:
        raise ValueError(f"Config file must contain two lines: {path}")
    parts = [int(x.strip()) for x in text[1].split(",")]
    if len(parts) != 3:
        raise ValueError(f"Config line must be 'stu,exer,know': {text[1]}")
    return parts[0], parts[1], parts[2]


def write_config_numbers(config_path: os.PathLike | str, student_n: int, exercise_n: int, knowledge_n: int) -> None:
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config_path).write_text(
        "# Number of Students, Number of Exercises, Number of Knowledge Concepts\n"
        f"{student_n},{exercise_n},{knowledge_n}\n",
        encoding="utf-8",
    )


def normalize_concept(value: Any) -> str:
    return str(value).replace('"', '').strip().lower()


def iter_student_logs(students: Sequence[Mapping[str, Any]]) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    for row in students:
        yield int(row["user_id"]), list(row.get("logs", []))


def infer_dataset_shape(students: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
    max_user = 0
    max_exer = 0
    max_know = 0
    for user_id, logs in iter_student_logs(students):
        max_user = max(max_user, user_id)
        for log in logs:
            max_exer = max(max_exer, int(log.get("exer_id", 0)))
            max_know = max(max_know, int(log.get("knowledge_code", 0)))
    return max_user + 1, max_exer + 1, max_know + 1


def pick_distractors(true_concept: str, candidates: Sequence[str], k: int = 2, seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    true_norm = normalize_concept(true_concept)
    pool = [c for c in candidates if normalize_concept(c) != true_norm]
    rng.shuffle(pool)
    return pool[:k]
