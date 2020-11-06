from utils import *
from torch.utils.data.dataset import *
from torch.utils.data.sampler import *
from torch.nn.utils.rnn import *
import bisect
from model import *
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

root_dir = "/home/radmard/repos/AL4ST/"

charset = Charset()
vocab = Vocabulary()
vocab.load(f"{root_dir}data/NYT_CoType/vocab.txt")
tag_set = Index()
tag_set.load(f"{root_dir}data/NYT_CoType/tag2id.txt")
relation_labels = Index()
relation_labels.load(f"{root_dir}data/NYT_CoType/relation_labels.txt")

train_data = load(f"{root_dir}data/NYT_CoType/train.pk")
test_data = load(f"{root_dir}data/NYT_CoType/test.pk")
# CHANGED FOR DEBUG
val_size = int(0.01 * len(train_data))
train_data, val_data = random_split(train_data, [len(train_data) - val_size, val_size])


class ModifiedKL(nn.Module):

    def __init__(self, weight):
        """
        KL mask not used at the moment as we bypassed the log in KL, so o.h.e vectors can be used for it
        Reverse KL is not supported here
        TODO: Looking into weighting labelled and non labelled loss?
        """

        super().__init__()

        # weights of size [1, 1, classes]
        self.weight = weight.reshape(1, 1, -1)

    def forward(self, pred_log_probs, target_probs, kl_mask = None):

        # loss of size [batch, length, classes]
        loss = - pred_log_probs * target_probs
        loss *= self.weight

        # loss of size [batch, length]
        return loss.sum()


def group(data, breakpoints):
    groups = [[] for _ in range(len(breakpoints) + 1)]
    for idx, item in enumerate(data):
        # i.e. group into similar sentence sizes
        i = bisect.bisect_left(breakpoints, len(item[0]))
        groups[i].append(idx)
    data_groups = [Subset(data, g) for g in groups]
    return data_groups


class GroupBatchRandomSampler(object):
    def __init__(self, data_groups, batch_size, drop_last):
        self.batch_indices = []
        for data_group in data_groups:
            self.batch_indices.extend(
                list(
                    BatchSampler(
                        SubsetRandomSampler(data_group.indices),
                        batch_size,
                        drop_last=drop_last,
                    )
                )
            )

    def __iter__(self):
        return (self.batch_indices[i] for i in torch.randperm(len(self.batch_indices)))

    def __len__(self):
        return len(self.batch_indices)


def get_batch(batch_indices, data, device):
    batch = [data[idx] for idx in batch_indices]
    sentences, tokens, tags = zip(*batch)

    padded_sentences, lengths = pad_packed_sequence(
        pack_sequence([torch.LongTensor(_) for _ in sentences], enforce_sorted=False),
        batch_first=True,
        padding_value=vocab["<pad>"],
    )
    padded_tokens, _ = pad_packed_sequence(
        pack_sequence([torch.LongTensor(_) for _ in tokens], enforce_sorted=False),
        batch_first=True,
        padding_value=charset["<pad>"],
    )
    padded_tags, _ = pad_packed_sequence(
        pack_sequence([torch.LongTensor(_) for _ in tags], enforce_sorted=False),
        batch_first=True,
        padding_value=tag_set["O"],
    )

    padded_tags = nn.functional.one_hot(padded_tags, num_classes=193).float()  # MAKE NUM CLASSES A PARAMETER?
    kl_mask = torch.zeros(padded_tags.shape[:2]).to(device)

    return (
        padded_sentences.to(device),
        padded_tokens.to(device),
        padded_tags.to(device),
        lengths.to(device),
        kl_mask
    )


def get_triplets(tags):
    temp = {}
    triplets = []
    for idx, tag in enumerate(tags):
        if tag == tag_set["O"]:
            continue
        pos, relation_label, role = tag_set[tag].split("-")
        if pos == "B" or pos == "S":
            if relation_label not in temp:
                temp[relation_label] = [[], []]
            temp[relation_label][int(role) - 1].append(idx)
    for relation_label in temp:
        role1, role2 = temp[relation_label]
        if role1 and role2:
            len1, len2 = len(role1), len(role2)
            if len1 > len2:
                for e2 in role2:
                    idx = np.argmin([abs(e2 - e1) for e1 in role1])
                    e1 = role1[idx]
                    triplets.append((e1, relation_label, e2))
                    del role1[idx]
            else:
                for e1 in role1:
                    idx = np.argmin([abs(e2 - e1) for e2 in role2])
                    e2 = role2[idx]
                    triplets.append((e1, relation_label, e2))
                    del role2[idx]
    return triplets


# [vocab[a] for a in test_data[0][0]]   gives a sentence
# [tag_set[a] for a in test_data[0][2]] gives the corresponding tagseq

# torch.Size([47463, 300])
train_data_groups = group(train_data, [10, 20, 30, 40, 50, 60])
val_data_groups = group(val_data, [10, 20, 30, 40, 50, 60])
test_data_groups = group(test_data, [10, 20, 30, 40, 50, 60])

word_embeddings = torch.tensor(np.load(f"{root_dir}data/NYT_CoType/word2vec.vectors.npy"))
word_embedding_size = word_embeddings.size(1)
pad_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
unk_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
word_embeddings = torch.cat([pad_embedding, unk_embedding, word_embeddings])
