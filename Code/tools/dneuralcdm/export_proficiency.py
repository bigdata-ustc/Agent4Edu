import argparse
from pathlib import Path

import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils import save_json
from data import StudentSequenceDataset, make_loader
from model import DNeuralCDM


def parse_args():
    p = argparse.ArgumentParser(description="Export dynamic student knowledge proficiency from a trained DNeuralCDM checkpoint.")
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="stu_know_proficiency.json")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    dataset = StudentSequenceDataset(args.logs, checkpoint["num_exercises"], checkpoint["num_know"])
    _, _, _, all_data = dataset.split_by_student()
    loader = make_loader(all_data, batch_size=1, shuffle=False)
    model = DNeuralCDM(
        checkpoint["num_exercises"],
        checkpoint["num_know"],
        checkpoint["embedding_dim"],
        checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    out = {}
    with torch.no_grad():
        for sequences, masks, exe_ids, labels, user_ids, lengths in loader:
            _, _, stu_emb = model(sequences[:, :-1, :], exe_ids[:, 1:, :], masks[:, 1:, :])
            sid = str(int(user_ids[0][0]))
            out[sid] = stu_emb.cpu().tolist()
    save_json(args.output, out)
    print(f"Saved knowledge proficiency for {len(out)} students to {args.output}")


if __name__ == "__main__":
    main()
