import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import *
from torch.utils.data.sampler import *


class EarlyStopper:

    def __init__(self, patience: int, maximise: bool):
        """
        An early stopping & callback class.
        patience is an integer, the number of epochs that a non-optimal statistic is allowed (adding number of steps soon)
        maximise is set to True for scores, False for losses
        """

        self.patience = patience
        self.maximise = maximise
        self.model_state_dict = None
        self.model_state_dict_epoch = 0

    def assess_model(self, model, stats_list, epoch):
        """
        Returns:
            False if not stopping
            model_state_dict if stopping
        """
        if self.maximise:
            stats_list = [-a for a in stats_list] # Now we always minimise
        if len(stats_list) == 0:
            return False
        if self.check_stop(stats_list):
            return self.model_state_dict, self.model_state_dict_epoch
        if np.argmin(stats_list) == len(stats_list) - 1:
            self.model_state_dict = model.state_dict()
            self.model_state_dict_epoch = epoch

    def check_stop(self, stats_list):

        if self.patience < 0 or len(stats_list) < self.patience:
            return False
        if len(stats_list) - np.argmin(stats_list) > self.patience:
            return True
        else:
            return False


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

    def forward(self, pred_log_probs, target_probs):
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


