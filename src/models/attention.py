import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    """
    Additive attention mechanism (equations 10–12).
    """
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden]
        scores = self.v(torch.tanh(self.W(lstm_out)))  # [batch, seq_len, 1]
        weights = F.softmax(scores.squeeze(-1), dim=-1)  # [batch, seq_len]
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # [batch, hidden]
        return context, weights
