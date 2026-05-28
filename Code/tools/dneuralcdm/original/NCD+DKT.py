import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from rouge import Rouge  # 用于计算ROUGE指标
from rouge_score import rouge_scorer

# num_exercises = len(dataset.exercise_map)
# with open('../data/iflytek_sample/config.txt') as i_f:
#     i_f.readline()
#     num_stu, num_exercises, knowledge_n = i_f.readline().split(',')
num_exercises = 2074+1
num_know = 710

# 1. 数据预处理
class StudentDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)

        self.sequences = []
        self.labels = []
        self.masks = []
        # self.exercise_map = {}
        self.knowledge_list = {}

        # 为题目编码
        # exercise_id = 0
        for student in raw_data:
            sequence = []
            label = []
            mask = []
            # score_vec = []
            for log in student['logs']:
                know_id = log['knowledge_code']
                score_vec = [0]*num_know
                mask_vec = [0]*num_know
                exer_v = [0.0] * (2*num_know)
                if log['score'] == 1:
                    exer_v[2*(know_id)] = 1.0
                else:
                    exer_v[2*(know_id)+1] = 1.0

                # if exer not in self.exercise_map:
                #     self.exercise_map[exer] = exercise_id
                #     exercise_id += 1

                # 将题目与得分打包为序列
                # sequence.append(self.exercise_map[exer])
                sequence.append(exer_v)
                score_vec[know_id] = log['score']
                mask_vec[know_id] = 1

                label.append(log['score'])
                # label.append(score_vec)
                mask.append(mask_vec)

            self.sequences.append(sequence)
            self.labels.append(label)
            self.masks.append(mask)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.masks[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

    def train_val_test_split(self, train_ratio=0.8, val_ratio=0.2):
        """
        Split each student's sequence into training, validation, and test sets.
        - Train: first 80% of data (last 10% of this for validation).
        - Test: last 20% of data.
        """
        train_data, val_data, test_data, all_data = [], [], [], []

        for seq, mask, lbl in zip(self.sequences, self.masks, self.labels):
            split1 = int(len(seq) * train_ratio)
            split2 = int(split1 * (1 - val_ratio))

            train_data.append((seq[:split2], mask[:split2], lbl[:split2]))
            val_data.append((seq[split2:split1], mask[split2:split1], lbl[split2:split1]))
            test_data.append((seq[split1:], mask[split1:], lbl[split1:]))
            all_data.append((seq, mask, lbl))

        return train_data, val_data, test_data, all_data


# 2. 定义 LSTM 模型
class DKTModel(nn.Module):
    def __init__(self, num_exercises, embedding_dim, hidden_dim):
        super(DKTModel, self).__init__()
        self.embedding = nn.Linear(2*num_exercises, embedding_dim)  # 嵌入层: 将题目编码映射为稠密向量
        self.lstm = nn.LSTM(2*num_exercises, hidden_dim, batch_first=True)  # LSTM层: 处理时间序列数据
        self.fc0 = nn.Linear(hidden_dim, hidden_dim)  # 全连接层: 输出单个数值
        self.fc = nn.Linear(hidden_dim, num_exercises)  # 全连接层: 输出单个数值
        self.sigmoid = nn.Sigmoid()  # 激活函数: 输出概率值 (0~1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x, mask):
        """
        前向传播函数
        输入: x (题目序列, [batch_size, seq_len])
        输出: probs (预测正确率, [batch_size, seq_len, 1]), hidden_state (LSTM的隐藏状态)
        """
        # 1. 嵌入层映射
        # print(x.shape)
        x_reshaped = x.view(-1, x.shape[2]).float()
        # print(x_reshaped.shape)

        x_reshaped = self.embedding(x_reshaped)
        # print(x_reshaped.shape)
        #
        x_reshaped = x_reshaped.view(x.shape[0], x.shape[1], -1)
        # print(x_reshaped.shape)

        # 2. LSTM 处理时间序列
        lstm_out, (h_n, c_n) = self.lstm(x_reshaped)
        # print (lstm_out.shape)
        # print (h_n.shape)
        # print (c_n.shape)

        # print (torch.tanh(lstm_out))

        # y = torch.tanh(self.fc0(lstm_out))

        # 3. 全连接层映射到单值
        logits = self.fc(lstm_out)

        # 4. 激活函数输出概率
        # print (logits)
        probs = self.sigmoid(logits) * mask

        # print (probs.shape)

        probs = torch.sum(probs, dim=2)
        # print (probs.shape)
        # print ('sss')

        # 返回预测概率和LSTM的最终隐藏状态
        return probs, lstm_out


# 3. 数据准备与训练逻辑
def prepare_dataloader(data, batch_size):
    sequences, masks, labels = zip(*data)
    sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    masks = [torch.tensor(mask, dtype=torch.float32) for mask in masks]
    labels = [torch.tensor(lbl, dtype=torch.float32) for lbl in labels]
    dataset = [(seq, mask, lbl) for seq, mask, lbl in zip(sequences, masks, labels)]
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def collate_fn(batch):
    sequences, masks, labels = zip(*batch)
    seq_lengths = [len(seq) for seq in sequences]
    max_length = max(seq_lengths)

    # print(sequences[0].shape)

    padded_seqs = torch.zeros(len(batch), max_length, sequences[0][0].shape[0], dtype=torch.long)
    padded_masks = torch.zeros(len(batch), max_length, masks[0][0].shape[0], dtype=torch.long)
    # padded_labels = torch.zeros(len(batch), max_length, labels[0][0].shape[0], dtype=torch.long)
    padded_labels = torch.zeros(len(batch), max_length, dtype=torch.float32)

    for i, (seq, mask, lbl) in enumerate(zip(sequences, masks, labels)):
        # print(seq.shape)
        # print(padded_seqs[i].shape)

        padded_seqs[i, :len(seq)] = seq
        padded_masks[i, :len(mask)] = mask
        padded_labels[i, :len(lbl)] = lbl

    return padded_seqs, padded_masks, padded_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_acc = 0
    best_val_loss = 100
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for sequences, masks, labels in train_loader:
            optimizer.zero_grad()
            # print(sequences.shape)
            # print(masks.shape)

            outputs, lstm_out = model(sequences[:,:-1,:], masks[:,1:,:])  # (batch_size, seq_len, 1)
            # print(outputs.shape) # batch seq
            # print(labels[:,1:].shape)

            # 选择对应的输出
            # preds = outputs.squeeze(-1)
            # mask = labels != 0
            loss = criterion(outputs, labels[:,1:])
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sequences, masks, labels in val_loader:
                # print (sequences.shape)
                # print (sequences[:,:-1,:].shape)

                outputs, lstm_out = model(sequences[:,:-1,:], masks[:,1:,:])
                # outputs = outputs.squeeze(-1)
                # print (outputs.shape)
                # print (labels[:,1:,:].shape)
                predictions = (outputs >= 0.5).float()
                # mask = labels != 0
                val_correct += (predictions == labels[:,1:]).sum().item()
                val_total += predictions.shape[1] #mask.sum().item()
                loss = criterion(outputs, labels[:,1:])
                val_loss += loss.item()

        val_acc = val_correct / val_total
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_acc:.4f}")

        # # 保存最好的模型
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        # 保存最好的模型
        # if val_loss / len(val_loader) < best_val_loss:
        #     best_val_loss = val_loss / len(val_loader)
        #     best_model_state = model.state_dict()

    # 加载最好的模型
    model.load_state_dict(best_model_state)
    return model


def test_model(model, test_loader):
    """
    测试模型性能，计算准确率、F1、AUC和ROUGE-2。
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_test_sequences = []
    all_test_labels = []

    # 存储从训练集和验证集获得的隐状态
    all_hidden_states = []

    with torch.no_grad():
        # 使用训练集和验证集来获取模型的隐状态
        # for loader in [train_loader, val_loader, test_loader]:
        #     for sequences, labels in loader:
        #         # 获取模型的输出和隐状态
        #         probs, lstm_out = model(sequences)
        #
        #         # 预测结果
        #         # all_preds.append(probs.squeeze(-1).cpu().numpy())  # 预测概率
        #         # all_labels.append(labels.cpu().numpy())  # 实际标签
        #
        #         # 收集LSTM的输出（隐藏状态）
        #         all_hidden_states.append(lstm_out.cpu().numpy())  # 存储每个时间步的输出

        # 使用测试集进行最终预测
        # print (len(train_loader))
        # print (len(val_loader))
        for sequences, masks, labels in test_loader:
            probs, lstm_out = model(sequences[:,:-1,:], masks[:,1:,:])  # 获取模型的输出和隐状态
            # print (probs.shape)
            labels = labels[:,1:]

            all_preds.append(probs.cpu().numpy()[0][int(probs.shape[1]*0.8):])  # 预测概率
            all_labels.append(labels.cpu().numpy()[0][int(probs.shape[1]*0.8):])  # 实际标签
            # print (all_preds.cpu().numpy()[0].shape)
            # print (probs.squeeze(-1).cpu().numpy()[0].shape)

            # all_test_sequences.append(sequences.cpu().numpy()[int(probs.shape[1]*0.8):])  # 存储测试序列
            # all_test_labels.append(labels.cpu().numpy()[int(probs.shape[1]*0.8):])  # 存储测试标签

    # 转换为NumPy数组
    # print (all_preds.shape)
    # all_labels = np.concatenate(all_labels, axis=0)  # 连接所有标签
    # print (all_labels.shape)


    # 转换为NumPy数组
    all_preds = np.concatenate(all_preds, axis=0)  # 连接所有预测
    all_labels = np.concatenate(all_labels, axis=0)  # 连接所有标签

    # 将预测和真实值保存到本地txt文件
    with open('predictions.txt', 'w') as f:
        for pred, label in zip(all_preds, all_labels):
            f.write(f"{pred}\t{label}\n")  # 每行写入一个预测值和真实标签，使用tab分隔

    # 计算准确率
    accuracy = accuracy_score(all_labels, (all_preds > 0.5))  # 二分类，预测值大于0.5认为是1

    # 计算F1分数
    f1 = f1_score(all_labels, (all_preds > 0.5))

    # 计算AUC
    auc_score = roc_auc_score(all_labels, all_preds)


    # 计算ROUGE-2
    # scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    # rouge_2_scores = []
    # for pred, label in zip(all_preds, all_labels):
    #     # 将预测结果和标签转换为文本
    #     pred_text = " ".join([str(p) for p in pred])  # 假设pred为tokenized text
    #     label_text = " ".join([str(l) for l in label])  # 假设label为tokenized text
    #     rouge_2_scores.append(scorer.score(label_text, pred_text)['rouge2'].fmeasure)

    # rouge_2_avg = np.mean(rouge_2_scores)

    # 打印结果
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc_score:.4f}")
    # print(f"ROUGE-2: {rouge_2_avg:.4f}")
    rouge_2_avg=0

    return accuracy, f1, auc_score, rouge_2_avg, all_hidden_states

if __name__ == "__main__":
    # 参数设置
    DATA_PATH = "../data/iflytek_sample/stu_logs.json"
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    BATCH_SIZE = 1
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001

    # 加载数据
    dataset = StudentDataset(DATA_PATH)
    train_data, val_data, test_data, all_data = dataset.train_val_test_split()

    train_loader = prepare_dataloader(train_data, BATCH_SIZE)
    val_loader = prepare_dataloader(val_data, BATCH_SIZE)
    test_loader = prepare_dataloader(all_data, 1)


    # model = DKTModel(num_exercises, EMBEDDING_DIM, HIDDEN_DIM)
    # EMBEDDING_DIM = int(knowledge_n)
    # HIDDEN_DIM = int(knowledge_n)
    model = DKTModel(int(num_know), EMBEDDING_DIM, HIDDEN_DIM)

    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)

    # 测试模型
    test_model(model, test_loader)
