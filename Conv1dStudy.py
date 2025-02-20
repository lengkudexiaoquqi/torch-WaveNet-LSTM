import torch
import torch.nn as nn

#源自：https://zhuanlan.zhihu.com/p/296306578

# 原始输入大小为(50, 35, 300)，经过permute(0, 2, 1)操作后，输入的大小变为(50, 300, 35)；
# 使用1个window_size为3的卷积核进行卷积，因为一维卷积是在最后维度上扫的，最后output的大小即为：50*100*（35-3+1）=50*100*33
# output经过最大池化操作后，得到了数据维度为：(50,100,1)
# 经过（输入特征=100，输出特征=2）的全连接层，数据维度就变为了：(50，2)
# 再经过softmax函数就得到了属于两个类别的概率值

# 一般来说，一维卷积nn.Conv1d()用于文本数据，只对宽度进行卷积，对高度不卷积。
# 通常，输入大小为word_embedding_dim * max_sent_length，
# 其中，word_embedding_dim为词向量的维度，max_sent_length为句子的最大长度。
# 卷积核窗口在句子长度的方向上滑动，进行卷积操作。

# 输入：批大小为50，句子的最大长度为35，词向量维度为300

# 目标：句子分类，共2类

# max_sent_len=35, batch_size=50, embedding_size=300
conv1 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3)
input = torch.randn(50, 35, 300)
# batch_size x max_sent_len x embedding_size -> batch_size x embedding_size x max_sent_len
input = input.permute(0, 2, 1)
print("input:", input.size())
output = conv1(input)
print("output:", output.size())
# 最大池化
pool1d = nn.MaxPool1d(kernel_size=35-3+1)
pool1d_value = pool1d(output)
print("最大池化输出：", pool1d_value.size())
# 全连接
fc = nn.Linear(in_features=100, out_features=2)
fc_inp = pool1d_value.view(-1, pool1d_value.size(1))
print("全连接输入：", fc_inp.size())
fc_outp = fc(fc_inp)
print("全连接输出：", fc_outp.size())
# softmax
m = nn.Softmax()
out = m(fc_outp)
print("输出结果值：", out)