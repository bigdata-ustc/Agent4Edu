import argparse
import random
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils import infer_dataset_shape, load_json, save_json, write_config_numbers


def parse_args():
    p = argparse.ArgumentParser(description="Split stu_logs.json into IRT train/val/test files.")
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2025)
    return p.parse_args()


def flatten(students):
    out = []
    for row in students:
        uid = int(row["user_id"])
        for log in row.get("logs", []):
            out.append({
                "user_id": uid,
                "exer_id": int(log["exer_id"]),
                "knowledge_code": int(log["knowledge_code"]),
                "score": int(log["score"]),
            })
    return out


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    students = load_json(args.logs)
    student_n, exercise_n, knowledge_n = infer_dataset_shape(students)
    records = flatten(students)
    rng.shuffle(records)
    n = len(records)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "train_set.json", train)
    save_json(args.output_dir / "val_set.json", val)
    save_json(args.output_dir / "test_set.json", test)
    write_config_numbers(args.config, student_n, exercise_n, knowledge_n)
    print(f"Saved splits to {args.output_dir}: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Saved config to {args.config}: {student_n},{exercise_n},{knowledge_n}")


if __name__ == "__main__":
    main()
