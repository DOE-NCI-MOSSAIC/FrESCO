import math

import torch
from torch import nn
import torch.nn.functional as F


class CaseLevelContext(nn.Module):
    def __init__(self,
                 num_classes,
                 doc_embed_size=400,
                 att_dim_per_head=50,
                 att_heads=8,
                 att_dropout=0.1,
                 forward_mask=True,
                 device='cuda'
                 ):

        super().__init__()
        self.doc_embed_size = doc_embed_size
        self.att_dim_per_head = att_dim_per_head
        self.att_heads = att_heads
        self.att_dim_total = att_heads * att_dim_per_head
        self.att_dropout = att_dropout
        self.forward_mask = forward_mask
        self.num_tasks = len(num_classes)
        self.device = device

        # Q, K, V, and other layers self-attention
        self.q = nn.Linear(self.doc_embed_size, self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.q.weight)
        self.q.bias.data.fill_(0.0)
        self.k = nn.Linear(self.doc_embed_size, self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.k.weight)
        self.k.bias.data.fill_(0.0)
        self.v = nn.Linear(self.doc_embed_size, self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.v.weight)
        self.v.bias.data.fill_(0.0)
        self.input_drop = nn.Dropout(p=att_dropout)
        self.att_drop = nn.Dropout(p=att_dropout)
        self.output_drop = nn.Dropout(p=att_dropout)

        # prediction layers
        self.classify_layers = nn.ModuleList()
        for n in num_classes:
            l = nn.Linear(self.att_dim_total, n)
            torch.nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(0.0)
            self.classify_layers.append(l)

    def _split_heads(self, x):
        '''
        splits final dim of tensor into multiple heads for multihead attention

        parameters:
          - x: torch.tensor (float) [batch_size x seq_len x dim]

        outputs:
          - torch.tensor (float) [batch_size x att_heads x seq_len x att_dim_per_head]
            reshaped tensor for multihead attention
        '''
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.att_heads, self.att_dim_per_head)
        return torch.transpose(x, 1, 2)

    def _attention(self, q, k, v, drop, mask, max_seq_len):
        '''
        flexible attention operation for self and target attention

        parameters:
          - q: torch.tensor (float) [batch x heads x seq_len x dim1]
          - k: torch.tensor (float) [batch x heads x seq_len x dim1]
          - v: torch.tensor (float) [batch x heads x seq_len x dim2]
            NOTE: q and k must have the same dimension, but v can be different
          - drop: torch.nn.Dropout layer
          - mask: torch.tensor (bool) [batch x seq_len]
          - forward_mask: boolean
        '''

        # generate attention matrix
        # batch x heads x seq_len x seq_len
        scores = torch.matmul(q, torch.transpose(k, -1, -2)) / math.sqrt(q.size(-1))

        # forward mask
        if self.forward_mask:
            mask_f = torch.arange(end=max_seq_len, device=self.device)[None, :] <= \
                     torch.arange(end=max_seq_len, device=self.device)[:, None]
            mask_f = torch.unsqueeze(mask_f, 0)
            mask_f = torch.unsqueeze(mask_f, 0)
            padding_mask = torch.logical_not(mask_f)
            scores -= 1.e7 * padding_mask.float()

        # this masks out empty entries in the attention matrix
        # and prevents the softmax function from assigning them any attention
        if mask is not None:
            mask_q = torch.unsqueeze(mask, 1)
            mask_q = torch.unsqueeze(mask_q, -2)
            padding_mask = torch.logical_not(mask_q)
            scores -= 1.e7 * padding_mask.float()

        # normalize attention matrix
        # batch x heads x seq_len x seq_len
        weights = F.softmax(scores, -1)

        # this removes empty rows in the normalized attention matrix
        # and prevents them from affecting the new output sequence
        if mask is not None:
            mask_k = torch.unsqueeze(mask, 1)
            mask_k = torch.unsqueeze(mask_k, -1)
            weights = torch.mul(weights, mask_k.type(weights.dtype))

        # optional attention dropout
        if drop is not None:
            weights = drop(weights)

        # use attention on values to generate new output sequence
        # batch x heads x seq_len x dim2
        result = torch.matmul(weights, v)

        # this applies padding to the entries in the output sequence
        # and ensures all padded entries are set to 0
        if mask is not None:
            mask_v = torch.unsqueeze(mask, 1)
            mask_v = torch.unsqueeze(mask_v, -1)
            result = torch.mul(result, mask_v.type(result.dtype))

        return result


    def forward(self, doc_embeds, num_docs):
        '''
        case level context forward pass

        parameters:
          - doc_embeds: torch.tensor (float) [batch_size x max_seq_length x doc_embed_size]
          - num_docs: torch.tensor (int) [batch_size]
            number of reports per case

        outputs:

        '''
        # create mask
        batch_size = num_docs.shape[0]
        max_seq_len = doc_embeds.shape[1]
        mask = torch.arange(end=max_seq_len, device=self.device)[None, :] < num_docs[:, None]

        # self-attention
        doc_embeds = self.input_drop(doc_embeds)
        q = F.elu(self._split_heads(self.q(doc_embeds)))                        # batch x heads x max_seq_len x dim
        k = F.elu(self._split_heads(self.k(doc_embeds)))                        # batch x heads x max_seq_len x dim
        v = F.elu(self._split_heads(self.v(doc_embeds)))                        # batch x heads x max_seq_len x dim
        att_out = self._attention(q, k, v, self.att_drop, mask, max_seq_len)         # batch x heads x max_seq_len x dim
        att_out = att_out.transpose(1, 2).reshape(
                  batch_size, max_seq_len, self.att_dim_total)                    # batch x max_seq_len x heads*dim
        att_out = self.output_drop(att_out)                                     # batch x max_seq_len x heads*dim

        # classify
        logits = []
        for _, l in enumerate(self.classify_layers):
            logit = l(att_out)
            logits.append(logit)

        return logits
