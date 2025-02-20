import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    """
    Input and output sizes will be the same.
    """
    def __init__(self, in_size, out_size, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., :-self.pad]
        return x

input = torch.randn(10, 5, 5)
# batch_size x max_sent_len x embedding_size -> batch_size x embedding_size x max_sent_len
# input = input.permute(0, 2, 1)
print("input:", input.size())
cc1d = CausalConv1d(5, 5, kernel_size=3, dilation=1)
output = cc1d(input)
print("output:", output.size())
