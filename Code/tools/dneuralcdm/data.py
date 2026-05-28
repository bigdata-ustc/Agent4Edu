from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils import infer_dataset_shape, load_json


class StudentSequenceDataset:
    """Encode logs in the same form used by the original DNeuralCDM code."""

    def __init__(self, logs_path: Path, num_exercises: int | None = None, num_know: int | None = None):
        self.raw_data = load_json(logs_path)
        inferred_student, inferred_exer, inferred_know = infer_dataset_shape(self.raw_data)
        self.num_exercises = num_exercises or inferred_exer
        self.num_know = num_know or inferred_know
        self.samples = []
        for student in self.raw_data:
            sequence = []
            label = []
            mask = []
            exer_id_list = []
            user_id_list = []
            for log in student.get("logs", []):
                exercise_id = int(log["exer_id"])
                user_id = int(student["user_id"])
                know_id = int(log["knowledge_code"])
                score = int(log["score"])
                if exercise_id >= self.num_exercises or know_id >= self.num_know:
                    continue
                mask_vec = [0] * self.num_know
                exer_v = [0] * (2 * self.num_know)
                exer_id_vec = [0] * self.num_exercises
                if score == 1:
                    exer_v[2 * know_id] = 1.0
                else:
                    exer_v[2 * know_id + 1] = 1.0
                mask_vec[know_id] = 1
                exer_id_vec[exercise_id] = 1
                sequence.append(exer_v)
                label.append(score)
                mask.append(mask_vec)
                exer_id_list.append(exer_id_vec)
                user_id_list.append(user_id)
            if len(sequence) >= 2:
                self.samples.append((sequence, mask, exer_id_list, label, user_id_list))

    def split_by_student(self, train_ratio: float = 0.8, val_ratio: float = 0.2):
        train_data, val_data, test_data, all_data = [], [], [], []
        for seq, mask, exeid, lbl, uid in self.samples:
            split1 = int(len(seq) * train_ratio)
            split2 = int(split1 * (1 - val_ratio))
            train_data.append((seq[:split2], mask[:split2], exeid[:split2], lbl[:split2], uid[:split2]))
            val_data.append((seq[split2:split1], mask[split2:split1], exeid[split2:split1], lbl[split2:split1], uid[split2:split1]))
            test_data.append((seq[split1:], mask[split1:], exeid[split1:], lbl[split1:], uid[split1:]))
            all_data.append((seq, mask, exeid, lbl, uid))
        return train_data, val_data, test_data, all_data


def make_loader(data, batch_size: int, shuffle: bool = True):
    filtered = [x for x in data if len(x[0]) >= 2]
    return DataLoader(filtered, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def collate_fn(batch):
    sequences, masks, exe_ids, labels, user_ids = zip(*batch)
    max_length = max(len(seq) for seq in sequences)
    padded_seqs = torch.zeros(len(batch), max_length, len(sequences[0][0]), dtype=torch.float32)
    padded_masks = torch.zeros(len(batch), max_length, len(masks[0][0]), dtype=torch.float32)
    padded_exe_ids = torch.zeros(len(batch), max_length, len(exe_ids[0][0]), dtype=torch.float32)
    padded_labels = torch.zeros(len(batch), max_length, dtype=torch.float32)
    padded_userids = torch.zeros(len(batch), max_length, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, (seq, mask, exe_id, lbl, uid) in enumerate(zip(sequences, masks, exe_ids, labels, user_ids)):
        length = len(seq)
        lengths[i] = length
        padded_seqs[i, :length] = torch.tensor(seq, dtype=torch.float32)
        padded_masks[i, :length] = torch.tensor(mask, dtype=torch.float32)
        padded_exe_ids[i, :length] = torch.tensor(exe_id, dtype=torch.float32)
        padded_labels[i, :length] = torch.tensor(lbl, dtype=torch.float32)
        padded_userids[i, :length] = torch.tensor(uid, dtype=torch.long)
    return padded_seqs, padded_masks, padded_exe_ids, padded_labels, padded_userids, lengths
