import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import infer_dataset_shape, load_json, normalize_concept, read_config_numbers, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Build Agent4Edu profile.json from logs and IRT ability.")
    p.add_argument("--logs", type=Path, required=True, help="stu_logs.json")
    p.add_argument("--output", type=Path, required=True, help="output profile.json")
    p.add_argument("--config", type=Path, default=None, help="config.txt with stu,exer,know")
    p.add_argument("--irt-ability", type=Path, default=None, help="IRT ability npy, e.g. epoch_10_stu_ability.npy")
    p.add_argument("--ability-index-offset", type=int, default=0, help="array index = user_id + offset; use -1 if your ability file is strictly 1-indexed")
    p.add_argument("--default-ability", type=float, default=0.5, help="ability value used if --irt-ability is missing")
    return p.parse_args()


def build_profiles(logs_path: Path, config_path: Path | None, ability_path: Path | None, ability_index_offset: int, default_ability: float):
    students = load_json(logs_path)
    if config_path:
        _, exer_n, know_n = read_config_numbers(config_path)
    else:
        _, exer_n, know_n = infer_dataset_shape(students)
    ability = np.load(ability_path) if ability_path else None
    profiles = {}
    for row in students:
        user_id = int(row["user_id"])
        logs = list(row.get("logs", []))
        log_num = len(logs)
        activity = log_num / max(exer_n, 1)
        concepts = [int(log.get("knowledge_code", 0)) for log in logs]
        diversity = len(set(concepts)) / max(know_n, 1)
        names = [normalize_concept(log.get("know_name", log.get("knowledge_code", ""))) for log in logs]
        preference = Counter(names).most_common(1)[0][0] if names else ""
        success_rate = sum(int(log.get("score", 0)) for log in logs) / max(log_num, 1)
        ability_value = default_ability
        if ability is not None:
            idx = user_id + ability_index_offset
            if 0 <= idx < len(ability):
                ability_value = float(np.asarray(ability[idx]).reshape(-1)[0])
            elif 0 <= user_id - 1 < len(ability):
                ability_value = float(np.asarray(ability[user_id - 1]).reshape(-1)[0])
        profiles[str(user_id)] = f"{user_id}\t{activity}\t{diversity}\t{preference}\t{success_rate}\t{ability_value}\n"
    return profiles


def main():
    args = parse_args()
    profiles = build_profiles(args.logs, args.config, args.irt_ability, args.ability_index_offset, args.default_ability)
    save_json(args.output, profiles)
    print(f"Saved {len(profiles)} learner profiles to {args.output}")


if __name__ == "__main__":
    main()
