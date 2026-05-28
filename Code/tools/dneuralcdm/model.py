import torch
import torch.nn as nn


class DNeuralCDM(nn.Module):
    def __init__(self, num_exercises, num_know, embedding_dim, hidden_dim):
        super(DNeuralCDM, self).__init__()
        self.embedding = nn.Linear(2*num_know, embedding_dim)
        self.exer_diff = nn.Linear(1*num_exercises, num_know)
        self.exer_disc = nn.Linear(1*num_exercises, 1)
        self.lstm = nn.LSTM(2*num_exercises, hidden_dim, batch_first=True)
        self.fc0 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_know)
        self.pre1 = nn.Linear(num_know, 512)
        self.pre2 = nn.Linear(512, 256)
        self.pre3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.pre1.apply(clipper)
        self.pre2.apply(clipper)
        self.pre3.apply(clipper)

    def forward(self, x, exercise_id, mask):
        x_reshaped = x.view(-1, x.shape[2]).float()
        x_reshaped = self.embedding(x_reshaped)
        x_reshaped = x_reshaped.view(x.shape[0], x.shape[1], -1)

        lstm_out, (h_n, c_n) = self.lstm(x_reshaped)

        exercise_id_reshaped = exercise_id.view(-1, exercise_id.shape[2]).float()
        exer_diff = torch.sigmoid(self.exer_diff(exercise_id_reshaped)).view(exercise_id.shape[0], exercise_id.shape[1], -1)
        exer_disc = (self.exer_disc(exercise_id_reshaped)).view(exercise_id.shape[0], exercise_id.shape[1], -1)*1

        stu_emb = torch.sigmoid(self.fc(lstm_out))

        input_x = (stu_emb - exer_diff) * mask * exer_disc
        input_x = torch.tanh(self.pre1(input_x)).squeeze(-1)
        input_x = torch.tanh(self.pre2(input_x)).squeeze(-1)
        probs = self.sigmoid(self.pre3(input_x)).squeeze(-1)
        return probs, lstm_out, stu_emb


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
