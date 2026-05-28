import argparse
from pathlib import Path

import torch
from torch import nn

from data import StudentSequenceDataset, make_loader
from model import DNeuralCDM


def parse_args():
    p = argparse.ArgumentParser(description="Train DNeuralCDM and save a checkpoint.")
    p.add_argument("--logs", type=Path, required=True, help="stu_logs.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-exercises", type=int, default=0)
    p.add_argument("--num-know", type=int, default=0)
    p.add_argument("--embedding-dim", type=int, default=0, help="0 means 2*num_exercises, matching the original DNeuralCDM LSTM input size")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.001)
    return p.parse_args()


def step_loss(model, batch, criterion, device):
    sequences, masks, exe_ids, labels, user_ids, lengths = batch
    sequences = sequences.to(device)
    masks = masks.to(device)
    exe_ids = exe_ids.to(device)
    labels = labels.to(device)
    outputs, _, _ = model(sequences[:, :-1, :], exe_ids[:, 1:, :], masks[:, 1:, :])
    target = labels[:, 1:]
    return criterion(outputs, target), outputs, target


def main():
    args = parse_args()
    dataset = StudentSequenceDataset(args.logs, args.num_exercises or None, args.num_know or None)
    train_data, val_data, _, all_data = dataset.split_by_student()
    train_loader = make_loader(train_data, args.batch_size, True)
    val_loader = make_loader(val_data, args.batch_size, False)
    embedding_dim = args.embedding_dim or (2 * dataset.num_exercises)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNeuralCDM(dataset.num_exercises, dataset.num_know, embedding_dim, args.hidden_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = None
    best_val = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _, _ = step_loss(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
            model.apply_clipper()
            train_loss += loss.item()
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss, outputs, target = step_loss(model, batch, criterion, device)
                val_loss += loss.item()
                pred = (outputs >= 0.5).float()
                correct += (pred == target).sum().item()
                total += target.numel()
        val_acc = correct / max(total, 1)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = model.state_dict()
            torch.save({
                "model_state_dict": best_state,
                "num_exercises": dataset.num_exercises,
                "num_know": dataset.num_know,
                "embedding_dim": embedding_dim,
                "hidden_dim": args.hidden_dim,
            }, args.output_dir / "best_model.pt")
        print(f"Epoch {epoch}/{args.epochs}, train_loss={train_loss/max(len(train_loader),1):.4f}, val_loss={val_loss/max(len(val_loader),1):.4f}, val_acc={val_acc:.4f}")
    print(f"Saved best checkpoint to {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
