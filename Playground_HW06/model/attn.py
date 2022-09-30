import math
from torch import nn
import torch


class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()
        self.word_embeddings = nn.Linear(vocabs, 4)  # word embedding size for 4
        self.d = 3  # embedding size 3 for q,k,v
        self.q_w = nn.Linear(4, self.d)
        self.k_w = nn.Linear(4, self.d)
        self.v_w = nn.Linear(4, self.d)

    def forward(self, x, attention_mask):
        x = x.to(torch.float)
        x = self.word_embeddings(x)

        Q = self.q_w(x)
        K = self.k_w(x)
        V = self.v_w(x)

        score = torch.matmul(Q, torch.transpose(K, 1, 2))  # batch_dot Q*K^trnas
        score = score / math.sqrt(self.d)
        print('Q*K before apply attention_mask\n', score)

        attention_mask = attention_mask
        attention_mask = attention_mask.unsqueeze(1).repeat(1, score.shape[1], 1)
        score = score * attention_mask  # apply mask to score
        print('Q*K after apply attention_mask\n', score)

        score = score - torch.where(attention_mask > 0, torch.zeros_like(score), torch.ones_like(score) * float(
            'inf'))  # apply mask to softmax for thoese value is `0`

        print('Q*K prepare for mask_softmax\n', score)
        softmax = torch.nn.Softmax(dim=-1)
        atten_prob = softmax(score)
        print('Atten prob\n', atten_prob)
        atten_score = torch.matmul(atten_prob, V)
        print('Atten score\n', atten_score)

        return {'atten_prob': atten_prob, 'atten_score': atten_score}


if __name__ == "__main__":
    vocabs = 6
    model = Attn()
    x = torch.randn(1, 4, vocabs)
    mask = torch.torch.LongTensor([[1, 1, 1, 1]])
    print(mask)
    print(model(x, mask))
