import torch
import torch.nn as nn
from .attention import AdditiveAttention

class CVNet(nn.Module):
    def __init__(self, input_size, lstm_hidden, dropout, attention, batch_norm, dense_units, num_classes):
        super(CVNet, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden[0], num_layers=len(lstm_hidden), batch_first=True, dropout=dropout if len(lstm_hidden) > 1 else 0)
        self.batch_norm = nn.BatchNorm1d(lstm_hidden[-1]) if batch_norm else None
        self.attention = AdditiveAttention(lstm_hidden[-1]) if attention == 'additive' else None
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(lstm_hidden[-1], dense_units)
        self.output = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.batch_norm:
            lstm_out = self.batch_norm(lstm_out.transpose(1, 2)).transpose(1, 2)
        if self.attention:
            context, weights = self.attention(lstm_out)
            out = context
        else:
            out = lstm_out[:, -1, :]  # Last timestep
        out = self.dropout(out)
        out = torch.relu(self.dense(out))
        out = self.output(out)
        return out
