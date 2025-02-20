import torch.nn as nn
import torch.nn.functional as F
import torch

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class ResidualLayer(nn.Module):
    def __init__(self, residual_size, skip_size, dilation):
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_size, residual_size, kernel_size=2, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_size, residual_size, kernel_size=2, dilation=dilation)
        self.resconv1_1 = nn.Conv1d(residual_size, residual_size, kernel_size=1)
        self.skipconv1_1 = nn.Conv1d(residual_size, skip_size, kernel_size=1)

    def forward(self, x):
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)
        fx = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
        fx = self.resconv1_1(fx)
        skip = self.skipconv1_1(fx)
        residual = fx + x
        return skip, residual


class DilatedStack(nn.Module):
    def __init__(self, residual_size, skip_size, dilation_depth):
        super(DilatedStack, self).__init__()
        residual_stack = [ResidualLayer(residual_size, skip_size, 2 ** layer)
                          for layer in range(dilation_depth)]
        self.residual_stack = nn.ModuleList(residual_stack)

    def forward(self, x):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
            # skip =[1,batch,skip_size,seq_len]
        return torch.cat(skips, dim=0), x  # [layers,batch,skip_size,seq_len]


class WaveNet_LSTM(nn.Module):
    def __init__(self, input_size, residual_size, skip_size, dilation_depth, hidden_size, predict_step=1,
                 dilation_cycles=1):
        super(WaveNet_LSTM, self).__init__()
        self.input_conv = CausalConv1d(input_size, residual_size, kernel_size=2)
        self.dilated_stacks = nn.ModuleList(
            [DilatedStack(residual_size, skip_size, dilation_depth)
             for cycle in range(dilation_cycles)]
        )
        self.lstm = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=skip_size,
            hidden_size=hidden_size,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, predict_step)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # input is [batch_size,seq_len,input_feature]
        x = x.permute(0, 2, 1)  # [batch_size,input_feature_dim,seq_len]
        x = self.input_conv(x)  # [batch_size,residual_size,seq_len]  first_causal_conv
        skip_connections = []
        for cycle in self.dilated_stacks:
            skips, x = cycle(x)
            skip_connections.append(skips)

        # skip_connection =[total_layers,batch_size,skip_size,seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)
        # gather all output skip connections to generate output,discard last residual output
        out = skip_connections.sum(dim=0)  # [batch,skip_size,seq_len]
        #         in order to input to lstm, out should permute
        out = out.permute(0, 2, 1)  # [batch_size, seq_len, skip_size]
        r_out, (h_n, h_c) = self.lstm(out, None)
        res = self.out(r_out[:, -1, :])
        return res
