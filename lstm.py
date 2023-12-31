import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, d_model, features, n_layer):
        super(LSTM, self).__init__()

        self.encode = nn.Linear(features, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layer)
        self.decode = nn.Linear(d_model, features)

    def forward(self, input_ids):
        x = self.encode(input_ids)
        x, _ = self.lstm(x)
        x = self.decode(x)
        return x
    
