import torch
import torch.nn as nn


class Net(nn.Module):
    """
    IRT / 2PL / optional 3PL model.

    2PL:
        P(Y_ui = 1) = sigmoid(a_i * (theta_u - b_i))

    3PL:
        P(Y_ui = 1) = c_i + (1 - c_i) * sigmoid(a_i * (theta_u - b_i))
    """

    def __init__(self, student_n, exer_n, knowledge_n=None, use_guess=False):
        super(Net, self).__init__()

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.use_guess = use_guess

        # student ability theta_u
        self.student_emb = nn.Embedding(self.emb_num, 1)

        # item difficulty b_i
        self.k_difficulty = nn.Embedding(self.exer_n, 1)

        # item discrimination a_i
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        # guessing parameter c_i, only used when use_guess=True
        self.e_guess = nn.Embedding(self.exer_n, 1)

        # initialization
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id):
        """
        :param stu_id: LongTensor, student ids
        :param exer_id: LongTensor, exercise ids
        :return: FloatTensor, probability of answering correctly
        """

        # theta_u: student ability
        # Here sigmoid constrains ability into (0, 1), following your original design.
        stu_emb = torch.sigmoid(self.student_emb(stu_id))

        # b_i: item difficulty
        # Here sigmoid constrains difficulty into (0, 1), following your original design.
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))

        # a_i: item discrimination, constrained to positive values.
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10

        # Correct 2PL IRT:
        # P = sigmoid(a_i * (theta_u - b_i))
        irt_output = torch.sigmoid(
            e_discrimination * (stu_emb - k_difficulty)
        )

        # Optional 3PL IRT:
        # P = c_i + (1 - c_i) * sigmoid(a_i * (theta_u - b_i))
        if self.use_guess:
            e_guess = torch.sigmoid(self.e_guess(exer_id))
            output = e_guess + (1 - e_guess) * irt_output
        else:
            output = irt_output

        return output

    def apply_clipper(self):
        """
        Apply non-negative clipping if needed.
        In this implementation, discrimination is already positive because of sigmoid,
        so this function is usually unnecessary.
        """
        self.apply(NoneNegClipper())

    def get_knowledge_status(self, stu_id):
        """
        Return student ability theta_u.
        """
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        """
        Return item difficulty b_i and discrimination a_i.
        """
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
