import argparse
from pathlib import Path

import numpy as np
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils import load_json, read_config_numbers
from model import Net


def parse_args():
    p = argparse.ArgumentParser(description="Train the IRT/CD model and export student ability.")
    p.add_argument("--data-dir", type=Path, required=True, help="directory with train_set/val_set/test_set.json")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.002)
    return p.parse_args()


def make_loader(path: Path, batch_size: int, shuffle: bool):
    rows = load_json(path)
    user_ids = torch.tensor([int(r["user_id"]) for r in rows], dtype=torch.long)
    exer_ids = torch.tensor([int(r["exer_id"]) for r in rows], dtype=torch.long)
    scores = torch.tensor([float(r["score"]) for r in rows], dtype=torch.float32)
    return DataLoader(TensorDataset(user_ids, exer_ids, scores), batch_size=batch_size, shuffle=shuffle)


def evaluate(net, loader, device):
    net.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for user_ids, exer_ids, scores in loader:
            user_ids, exer_ids, scores = user_ids.to(device), exer_ids.to(device), scores.to(device)
            preds = net(user_ids, exer_ids).view(-1)
            loss = criterion(preds, scores)
            total_loss += loss.item() * len(scores)
            correct += ((preds >= 0.5).float() == scores).sum().item()
            total += len(scores)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ability").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model").mkdir(parents=True, exist_ok=True)
    student_n, exercise_n, knowledge_n = read_config_numbers(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(student_n, exercise_n, knowledge_n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    train_loader = make_loader(args.data_dir / "train_set.json", args.batch_size, True)
    val_loader = make_loader(args.data_dir / "val_set.json", args.batch_size, False)
    for epoch in range(1, args.epochs + 1):
        net.train()
        total_loss = 0.0
        total = 0
        for user_ids, exer_ids, scores in train_loader:
            user_ids, exer_ids, scores = user_ids.to(device), exer_ids.to(device), scores.to(device)
            optimizer.zero_grad()
            preds = net(user_ids, exer_ids).view(-1)
            loss = criterion(preds, scores)
            loss.backward()
            optimizer.step()
            net.apply_clipper()
            total_loss += loss.item() * len(scores)
            total += len(scores)
        val_loss, val_acc = evaluate(net, val_loader, device)
        torch.save(net.state_dict(), args.output_dir / "model" / f"model_epoch{epoch}.pt")
        student_ids = torch.arange(0, student_n, dtype=torch.long, device=device)
        ability = net.get_knowledge_status(student_ids).detach().cpu().numpy()
        np.save(args.output_dir / "ability" / f"epoch_{epoch}_stu_ability.npy", ability)
        print(f"Epoch {epoch}/{args.epochs}, train_loss={total_loss/max(total,1):.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
