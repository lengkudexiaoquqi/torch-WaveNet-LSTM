from torch import nn


class LSTM(nn.Module):
    def __init__(self,hidden_size,predict_step,input_dim):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=input_dim,
            hidden_size=hidden_size,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_size, predict_step)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(r_out[:, -1, :])
        return out


class RNN(nn.Module):
    def __init__(self,hidden_size,predict_step,input_dim):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(         # if use nn.RNN(), it hardly learns
            input_size=input_dim,
            hidden_size=hidden_size,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_size, predict_step)

    def forward(self, x):
        r_out ,h = self.rnn(x, None)
        out = self.out(r_out[:,-1,:])
        return out


class BPNet(nn.Module):

    def __init__(self,input_size,hidden_size,predict_step):
        super(BPNet, self).__init__()
        self.Flatten=nn.Flatten()
        self.Linear1=nn.Linear(input_size,hidden_size)
        self.Sigmoid=nn.Sigmoid()
        self.Linear2=nn.Linear(hidden_size,predict_step)

    def forward(self, x):
        x=self.Flatten(x)
        x=self.Sigmoid(self.Linear1(x))
        y_pre=self.Linear2(x)
        return y_pre

