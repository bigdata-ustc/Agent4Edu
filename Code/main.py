import argparse
from pathlib import Path

from action import Action
from config import DATA_PATH, RESULT_PATH
from llm_client import LLMClient
from memory import Memory
from profile import Profile
from utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agent4Edu learner simulation.")
    parser.add_argument("--data-dir", type=Path, default=DATA_PATH, help="Directory containing stu_logs/profile/kcg files.")
    parser.add_argument("--result-dir", type=Path, default=RESULT_PATH, help="Directory to save simulation outputs.")
    parser.add_argument("--students", type=str, default="0", help="Comma-separated row indices in stu_logs.json. Example: 0,1,2")
    parser.add_argument("--student-ids", action="store_true", help="Treat --students as actual user_id values instead of row indices.")
    parser.add_argument("--max-steps", type=int, default=0, help="Max exercises per student. 0 means all logs.")
    parser.add_argument("--llm-provider", type=str, default=None, choices=["mock", "openai"], help="Override AGENT4EDU_LLM_PROVIDER.")
    return parser.parse_args()


def select_students(all_students: list[dict], selector: str, by_id: bool) -> list[dict]:
    values = [int(x.strip()) for x in selector.split(",") if x.strip()]
    if by_id:
        lookup = {int(row["user_id"]): row for row in all_students}
        missing = [v for v in values if v not in lookup]
        if missing:
            raise KeyError(f"Student ids not found: {missing}")
        return [lookup[v] for v in values]
    return [all_students[i] for i in values]


def run_student(row: dict, data_dir: Path, result_dir: Path, action: Action, max_steps: int = 0) -> dict:
    student_id = int(row["user_id"])
    profile = Profile(student_id, data_dir)
    memory = Memory(student_id, data_dir)
    logs = list(row.get("logs", []))
    if max_steps > 0:
        logs = logs[:max_steps]

    outputs = []
    for step, practice in enumerate(logs, start=1):
        print(f"Simulating student={student_id}, step={step}, exercise={practice.get('exer_id')}")
        outputs.append(action.simulate_practice(profile, memory, practice, step))

    student_dir = result_dir / f"student_{student_id}"
    save_json(student_dir / "actions.json", outputs)
    save_json(student_dir / "memory_factual.json", memory.factual)
    save_json(student_dir / "memory_short.json", memory.short)
    save_json(student_dir / "memory_long.json", memory.long)
    summary = {
        "student_id": student_id,
        "num_steps": len(outputs),
        "task1_accuracy": _avg([o["task_scores"]["task1"] for o in outputs]),
        "task2_accuracy": _avg([o["task_scores"]["task2"] for o in outputs]),
        "task4_accuracy": _avg([o["task_scores"]["task4"] for o in outputs]),
        "has_dneuralcdm_proficiency": memory.knowledge_proficiency.available(),
    }
    save_json(student_dir / "summary.json", summary)
    return summary


def _avg(xs: list[int]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    args = parse_args()
    students = load_json(args.data_dir / "stu_logs.json")
    selected = select_students(students, args.students, args.student_ids)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    llm = LLMClient(provider=args.llm_provider)
    action = Action(llm)
    summaries = [run_student(row, args.data_dir, args.result_dir, action, args.max_steps) for row in selected]
    save_json(args.result_dir / "summary.json", summaries)
    print(f"Finished. Results saved to: {args.result_dir}")


if __name__ == "__main__":
    main()
