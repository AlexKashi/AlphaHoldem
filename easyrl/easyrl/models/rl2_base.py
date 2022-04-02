import torch.nn as nn


class RL2Base(nn.Module):
    def __init__(self,
                 body_net,
                 rnn_features=128,
                 in_features=128,
                 rnn_layers=1,
                 ):
        super().__init__()
        self.body = body_net
        self.gru = nn.GRU(input_size=in_features,
                          hidden_size=rnn_features,
                          num_layers=rnn_layers,
                          batch_first=True)
        self.fcs = nn.Linear(rnn_features, rnn_features)
        self.fcs = nn.Sequential(
            nn.ELU(),
            nn.Linear(in_features=rnn_features, out_features=rnn_features),
            nn.ELU()
        )

    def forward(self, x=None, hidden_state=None):
        b = x.shape[0]
        t = x.shape[1]
        x = x.view(b * t, *x.shape[2:])
        obs_feature = self.body(x)
        obs_feature = obs_feature.view(b, t, *obs_feature.shape[1:])
        rnn_features, hidden_state = self.gru(obs_feature,
                                              hidden_state)
        out = self.fcs(rnn_features)
        return out, hidden_state
