import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim_input, dim_q, dim_v):
        '''
        参数说明：
        dim_input: 输入数据x中每一个样本的向量维度
        dim_q:     Q矩阵的列向维度, 在运算时dim_q要和dim_k保持一致;
                   因为需要进行: K^T*Q运算, 结果为：[dim_input, dim_input]方阵
        dim_v:     V矩阵的列项维度,维度数决定了输出数据attention的列向维度
        '''
        super(SelfAttention, self).__init__()

        # dim_k = dim_q
        self.dim_input = dim_input
        self.dim_q = dim_q
        self.dim_k = dim_q
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(self.dim_input, self.dim_q, bias=False)
        self.linear_k = nn.Linear(self.dim_input, self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_input, self.dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(self.dim_k)

    def forward(self, x):
        batch, n, dim_q = x.shape

        q = self.linear_q(x)  # Q: batch_size * seq_len * dim_k
        k = self.linear_k(x)  # K: batch_size * seq_len * dim_k
        v = self.linear_v(x)  # V: batch_size * seq_len * dim_v
        print(f'x.shape:{x.shape} \n  Q.shape:{q.shape} \n  K.shape: {k.shape} \n  V.shape:{v.shape}')
        # K^T*Q
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        # 归一化获得attention的相关系数：A
        dist = torch.softmax(dist, dim=-1)
        print('attention matrix: ', dist.shape)
        # socre与v相乘，获得最终的输出
        att = torch.bmm(dist, v)
        print('attention output: ', att.shape)
        return att


if __name__ == '__main__':
    batch_size = 2  # 批量数
    dim_input = 5  # 句子中每个单词的向量维度，也就是每个最小样本x的维度
    seq_len = 3  # 句子的长度，样本的数量
    x = torch.randn(batch_size, seq_len, dim_input)
    self_attention = SelfAttention(dim_input, 10, 12)
    print(x)
    print('=' * 50)
    attention = self_attention(x)
    print('=' * 50)
    print(attention)

