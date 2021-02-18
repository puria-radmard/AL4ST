import json
import os
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence


def total_sum(thing):
    if isinstance(thing, set):
        return sum(thing)
    elif isinstance(thing, list):
        first_sum = np.sum(thing)
        if isinstance(first_sum, int) or isinstance(first_sum, float):
            return first_sum
        else:
            return np.sum(first_sum)
    elif isinstance(thing, np.ndarray):
        return np.sum(thing)


class ActiveLearningDataset:

    def __init__(self,
                 data,
                 labels,
                 index_class,
                 semi_supervision_multiplier,
                 label_form=lambda data_point: data_point.shape,
                 ):

        # When initialised with labels, they must be for all the data.
        # Data without labels (i.e. in the real, non-simulation case), must be added later with self.data
        data = [np.array(d) for d in data]
        labels = [np.array(l) for l in labels]
        temp_labels = [np.ones_like(l)*np.nan for l in labels]
        last_preds = [[np.ones(label_form(d))*np.nan for d in data]]    # Preds and labels must be the same shape!!

        assert all(l.shape == label_form(data[i]) for i, l in enumerate(labels))

        # Non-rectangular np array
        self.data = data
        self.labels = labels
        self.index = index_class(self)
        self.temp_labels = temp_labels
        self.label_form = label_form
        self.last_preds = last_preds
        self.semi_supervision_multiplier = semi_supervision_multiplier

    def add_data(self, new_data):
        new_data = [np.array(nd) for nd in new_data]
        self.data.extend(new_data)
        self.labels.extend([np.ones(self.label_form(nd))*np.nan for nd in new_data])
        self.temp_labels.extend([np.ones(self.label_form(nd))*np.nan for nd in new_data])

    def add_labels(self, window, labels):
        self.labels[window.i][window.slice] = labels

    def add_temp_labels(self, window, temp_labels):
        self.temp_labels[window.i][window.slice] = temp_labels

    def data_from_window(self, window):
        return self.data[window.i][window.slice]

    def labels_from_window(self, window):
        return self.labels[window.i][window.slice]

    # def get_temporary_labels(self, i):
    #     temp_data = self.data[i]
    #     temp_labels = [self.data[i][j] if j in self.index.labelled_idx[i]
    #                    else self.temp_labels[i][j] for j in range(len(temp_data))]
    #     return temp_data, temp_labels

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return ActiveLearningDataset([self.data[idx]], [self.labels[idx]])
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return ActiveLearningDataset(self.data[start:stop:step], self.labels[start:stop:step])

    def __len__(self):
        return len(self.data)


class OneDimensionalSequenceTaggingDataset(ActiveLearningDataset):

    def __init__(self, data, labels, index_class, semi_supervision_multiplier, padding_token, empty_tag,
                 label_form=lambda data_point: data_point.shape):
        super().__init__(data, labels, index_class, semi_supervision_multiplier, label_form=label_form)
        self.empty_tag = empty_tag
        self.padding_token = padding_token

    def update_preds(self, batch_indices, preds, lengths):
        for j, i in enumerate(batch_indices):
            assert self.last_preds[i].shape == preds[j][:lengths[j]]
            self.last_preds[i] = preds[j][:lengths[j]]

    def get_batch(self, batch_indices, labels_important: bool): # batch_indices is a list, e.g. one of labelled_set
        """
        labels_important flag just to save a bit of time
        """

        sequences, tags = [self.data[i] for i in batch_indices], [self.labels[i] for i in batch_indices]

        padded_sentences, lengths = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in sequences], enforce_sorted=False),
            batch_first=True,
            padding_value=self.padding_token,
        )
        padded_tags, _ = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in tags], enforce_sorted=False),
            batch_first=True,
            padding_value=self.empty_tag,
        )

        # return padded_sentences, padded_tokens, padded_tags, lengths

        # self.model.eval()
        # model_log_probs = self.model(padded_sentences, padded_tokens).detach().to(agent.device)
        # self.model.train()
        semi_supervision_mask = torch.ones(padded_tags.shape)

        if labels_important:
            # Fill in the words that have not been queried
            for j, sentence_tags in enumerate(padded_tags):
                sentence_index = batch_indices[j]
                for token_idx in range(int(lengths[j])):
                    if token_idx in self.index.labelled_idx[sentence_index]:
                        pass
                    elif token_idx in self.index.temp_labelled_idx[sentence_index]:
                        padded_tags[j, token_idx] = self.temp_labels[sentence_index][token_idx]
                    elif token_idx in self.index.unlabelled_idx[sentence_index]:
                        padded_tags[j, token_idx] = \
                            torch.exp(self.last_preds[sentence_index][token_idx])       # Maybe change this exp?
                        semi_supervision_mask[j, token_idx] = self.semi_supervision_multiplier
                    else:  # Padding
                        continue

        return (
            padded_sentences,
            padded_tags,
            lengths,
            semi_supervision_mask
        )

class SentenceIndex:

    def __init__(self, dataset):
        self.__number_partially_labelled_sentences = 0
        self.labelled_idx = {j: set() for j in range(len(dataset.data))}
        self.unlabelled_idx = {j: set(range(len(dataset.data[j][0]))) for j in range(len(dataset.data))}
        self.temp_labelled_idx = {j: set() for j in range(len(dataset.data))}
        self.dataset = dataset

    def label_instance(self, i):
        self.labelled_idx[i] = self.unlabelled_idx[i]
        self.__number_partially_labelled_sentences += 1
        self.unlabelled_idx[i] = set()

    def label_window(self, window):
        if not self.labelled_idx[window.i] and window.size > 0:
            self.__number_partially_labelled_sentences += 1
        self.labelled_idx[window.i].update(range(*window.bounds))
        self.unlabelled_idx[window.i] -= set(range(*window.bounds))
        self.temp_labelled_idx[window.i] -= set(range(*window.bounds))

    def temporarily_label_window(self, window):
        self.unlabelled_idx[window.i] -= set(range(*window.bounds))
        self.temp_labelled_idx[window.i].update(range(*window.bounds))

    def new_window_unlabelled(self, new_window):
        if set(range(*new_window.range)).intersection(self.labelled_idx[new_window.i]):
            return False
        else:
            return True

    def is_partially_labelled(self, i):
        return total_sum(self.labelled_idx[i]) > 0

    def is_partially_temporarily_labelled(self, i):
        return total_sum(self.temp_labelled_idx[i]) > 0

    def has_any_labels(self, i):
        return self.is_partially_labelled(i) or self.is_partially_temporarily_labelled(i)

    def is_labelled(self, i):
        return total_sum(self.unlabelled_idx[i]) == 0

    def is_partially_unlabelled(self, i):
        return total_sum(self.unlabelled_idx[i]) > 0

    def get_number_partially_labelled_instances(self):
        return self.__number_partially_labelled_instances

    def make_nan_if_labelled(self, i, scores):
        res = []
        for j in range(len(scores)):
            if j in self.labelled_idx[i]:
                res.append(float('nan'))
            else:
                res.append(scores[j])
        return res

    def save(self, save_path):
        with open(os.path.join(save_path, "agent_index.pk"), "w") as f:
            json.dump(
                {
                    "labelled_idx": {k: list(v) for k, v in self.labelled_idx.items()},
                    "unlabelled_idx": {k: list(v) for k, v in self.unlabelled_idx.items()},
                    "temporarily_labelled_idx": {k: list(v) for k, v in self.temp_labelled_idx.items()},
                },
                f
            )


class SentenceSubsequence:

    def __init__(self, sentence_index, bounds, score):
        # self.list = [sentence_index, bounds, score]
        self.slice = slice(*bounds)
        self.bounds = bounds
        self.i = sentence_index
        self.score = score
        self.size = bounds[1] - bounds[0]

    # def __getitem__(self, idx):
    #    return self.list[idx]


class BeamSearchSolution:
    def __init__(self, windows, max_size, B, labelled_ngrams, init_size=None, init_score=None, init_overlap_index={}):
        self.windows = windows
        if not init_score:
            self.score = sum([w.score for w in windows])
        else:
            self.score = init_score
        if not init_size:
            self.size = sum([w.size for w in windows])
        else:
            self.size = init_size
        self.overlap_index = init_overlap_index
        self.max_size = max_size
        self.lock = False
        self.B = B
        self.labelled_ngrams = labelled_ngrams

    def add_window(self, new_window, train_set):
        if self.size >= self.max_size:
            self.lock = True
            return self
        init_size = self.size + new_window.size
        init_score = self.score + new_window.score
        init_overlap_index = self.overlap_index.copy()
        if new_window.i in init_overlap_index:
            init_overlap_index[new_window.i] = init_overlap_index[new_window.i].union(set(range(*new_window.bounds))) # Need to generalise this
        else:
            init_overlap_index[new_window.i] = set(range(*new_window.bounds))  # Need to generalise this
        new_ngram = train_set.data_from_window(new_window)
        ngram_annotations = train_set.labels_from_window(new_window)
        self.labelled_ngrams[new_ngram] = ngram_annotations
        return BeamSearchSolution(self.windows + [new_window], self.max_size, self.B, self.labelled_ngrams,
                                  init_size=init_size, init_score=init_score, init_overlap_index=init_overlap_index)

    def is_permutationally_distinct(self, other):
        # We do a proxy-check for permutation invariance by checking for score and size of solutions
        if abs(self.score - other.score) < 1e-6 and self.size == other.size:
            return False
        else:
            return True

    def all_permutationally_distinct(self, others):
        for other_solution in others:
            if not self.is_permutationally_distinct(other_solution):
                return False
        else:
            return True

    def new_window_unlabelled(self, new_window):
        if new_window.i not in self.overlap_index:
            self.overlap_index[new_window.i] = set() # Just in case!
            return True
        else:
            new_word_idx = set(range(*new_window.bounds))
            if self.overlap_index[new_window.i].intersection(new_word_idx):
                return False
            else:
                return True

    def branch_out(self, other_solutions, window_scores, train_set, allow_propagation):
        # ASSUME window_scores ALREADY SORTED
        local_branch = []
        for window in window_scores:
            if self.new_window_unlabelled(window):
                new_ngram = train_set.data_from_window(window)
                # i.e. if we are allowing automatic labelling and we've already seen this ngram, then skip
                if new_ngram in self.labelled_ngrams.keys() and allow_propagation:
                    continue
                else:
                    possible_node = self.add_window(window, train_set)
                if possible_node.all_permutationally_distinct(other_solutions):
                    local_branch.append(possible_node)
                if len(local_branch) == self.B:
                    return local_branch
            if self.lock:
                return [self]

        # No more windows addable
        if len(local_branch) == 0:
            self.lock = True
            return [self]
        else:
            return local_branch
