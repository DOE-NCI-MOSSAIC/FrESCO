import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class MTCNN(nn.Module):
    '''
    Multitask simple text CNN for classifying cancer pathology reports.

    Args:
        embedding_matrix (numpy.array): Numpy array of word embeddings.
            Each row should represent a word embedding.
            NOTE: The word index 0 is masked, so the first row is ignored.
        num_classes (list[int]): Number of possible output classes for each task.
        window_sizes (list[int], default: [3, 4, 5]): Window size (consecutive tokens examined) in parallel convolution layers.
            Must match the length of num_filters.
        num_filters (list[int], default: [300, 300, 300]): Number of filters used in parallel convolution layers.
            Must match the length of window_sizes.
        dropout (float, default: 0.5): Dropout rate applied to the final document embedding after maxpooling.
        bag_of_embeddings (bool, default: False): Adds a parallel bag of embeddings layer and concatenates it to the final document embedding.
        embeddings_scale (float, default: 2.5): Scaling of word embeddings matrix columns.

    Returns:
        None
    '''

    def __init__(self,
                 embedding_matrix,
                 num_classes,
                 window_sizes=None,
                 num_filters=None,
                 dropout=0.5,
                 bag_of_embeddings=True,
                 embeddings_scale=20
                ):

        if window_sizes is None:
            window_sizes = [3, 4, 5]
        if num_filters is None:
            num_filters = [300, 300, 300]

        super().__init__()

        # normalize and initialize embeddings
        embedding_matrix -= embedding_matrix.mean(axis=0)
        embedding_matrix /= (embedding_matrix.std(axis=0, ddof=1) * embeddings_scale)
        embedding_matrix[0] = 0
        self.embedding = nn.Embedding.from_pretrained(
                         torch.tensor(embedding_matrix,dtype=torch.float),
                         freeze=False,
                         padding_idx=0)

        # parallel convolution layers
        self.conv_layers = nn.ModuleList()
        for s,f in zip(window_sizes,num_filters):
            l = nn.Conv1d(embedding_matrix.shape[1],f,s)
            torch.nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(0.01)
            self.conv_layers.append(l)
        self.drop_layer = nn.Dropout(dropout)

        # optional bag of embeddings layers
        self.boe = bag_of_embeddings
        if self.boe:
            self.boe_dense = nn.Linear(embedding_matrix.shape[1],embedding_matrix.shape[1])
            torch.nn.init.xavier_uniform_(self.boe_dense.weight)
            self.boe_dense.bias.data.fill_(0.0)

        # dense classification layers
        self.classify_layers = nn.ModuleList()
        for n in num_classes:
            in_size = np.sum(num_filters)
            if self.boe:
                in_size += embedding_matrix.shape[1]
            l = nn.Linear(in_size,n)
            torch.nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(0.0)
            self.classify_layers.append(l)

    def forward(self, docs: torch.tensor, return_embeds: bool=False) -> list:
        '''
        MT-CNN forward pass.

        Args:
            docs (torch.tensor): Batch of documents to classify.
                Each document should be a 0-padded row of mapped word indices.

        Returns:
            list[torch.tensor]: List of predicted logits for each task.
        '''


        # generate masks for word padding
        # remove extra padding that exists across all documents in batch
        mask_words = (docs != 0)
        words_per_line = mask_words.sum(-1)
        max_words = words_per_line.max()
        max_words = max(max_words, 5)
        mask_words = torch.unsqueeze(mask_words[:,:max_words],-1)
        docs_input_reduced = docs[:,:max_words]

        # word embeddings
        word_embeds = self.embedding(docs_input_reduced)
        word_embeds = torch.mul(word_embeds,mask_words.type(word_embeds.dtype))
        word_embeds = word_embeds.permute(0,2,1)

        # parallel 1D word convolutions
        conv_outs = []
        for l in self.conv_layers:
            conv_out = F.relu(l(word_embeds))
            conv_outs.append(torch.max(conv_out,2)[0])
        concat = torch.cat(conv_outs,1)

        # bag of embeddings operations if enabled
        if self.boe:
            bag_embeds = torch.sum(word_embeds,-1)
            bag_embeds = torch.mul(bag_embeds,
                         1/torch.unsqueeze(words_per_line,-1).type(bag_embeds.dtype))
            bag_embeds = torch.tanh(self.boe_dense(bag_embeds))
            concat = torch.cat([concat,bag_embeds],1)

        # generate logits for each task
        doc_embeds = self.drop_layer(concat)
        logits = []
        for l in self.classify_layers:
            logits.append(l(doc_embeds))

        if return_embeds:
            return logits,doc_embeds
        return logits
