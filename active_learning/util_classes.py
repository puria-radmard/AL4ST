import json
import os
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
from typing import Callable


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


class ActiveLearningSubset:

    def __init__(self, dataset, indices):
        self.indices = indices
        self.dataset = dataset
        self.access_mode = 'data'

    def __getattr__(self, item):
        self.access_mode = item

    def __getitem__(self, idx):
        return self.dataset.__getattr__(self.access_mode)[idx]


class ALAttribute:

    def __init__(self, name: str, unit_form: Callable, initialisation: list, cache: bool = False):
        # might change cache to arbitrary length

        self.name = name
        self.unit_form = unit_form
        self.attr = initialisation
        self.cache = cache
        if cache:
            self.prev_attr = initialisation

    def __getitem__(self, idx):
        return self.attr[idx]

    def __setitem__(self, idx, value):
        self.attr[idx] = value

    def __len__(self):
        return len(self.attr)

    def generate_nans(self, new_data):
        if isinstance(new_data, list):
            return [np.ones(self.unit_form(nd)) * np.nan for nd in new_data]
        else:
            return np.ones(self.unit_form(new_data)) * np.nan

    def get_attr_by_window(self, window):
        return self.attr[window.i][window.slice]

    # Put asserts in here!!!
    def set_attr_with_window(self, window, new_attr):
        assert self.get_attr_by_window(window).shape == new_attr.shape
        if self.cache:
            self.prev_attr[window.i][window.slice] = self.attr[window.i][window.slice]
        self.attr[window.i][window.slice] = new_attr

    def expand_size(self, new_data):
        self.attr.extend(self.generate_nans(new_data))

    def add_new_data(self, new_data):
        self.attr.extend(new_data)


class ActiveLearningDataset:

    def __init__(self,
                 data,
                 labels,
                 index_class,
                 semi_supervision_multiplier,
                 al_attributes=[],
                 label_form=lambda data_point: data_point.shape,
                 ):

        # When initialised with labels, they must be for all the data.
        # Data without labels (i.e. in the real, non-simulation case), must be added later with self.data

        assert all(l.shape == label_form(data[i]) for i, l in enumerate(labels))

        self.attrs = {
            "data": ALAttribute(name="data", unit_form=lambda d: d.shape, initialisation=[np.array(d) for d in data]),
            "labels": ALAttribute(name="labels", unit_form=label_form, initialisation=[np.array(l) for l in labels]),
            "temp_labels": ALAttribute(name="temp_labels", unit_form=label_form, initialisation=[np.ones_like(l)*np.nan for l in labels]),
            "last_preds": ALAttribute(name="last_preds", unit_form=label_form, initialisation=[np.ones_like(l)*np.nan for l in labels], cache=True),
        }
        self.attrs.update({ala.name: ala for ala in al_attributes})
        self.label_form = label_form
        self.semi_supervision_multiplier = semi_supervision_multiplier

        if index_class.__class__ == type:
            self.index = index_class(self)
        else:
            self.index = index_class

    def __getattr__(self, attr):
        return self.attrs[attr]

    def add_attribute(self, new_attribute):
        attr_name = new_attribute.name
        if attr_name in self.attrs:
            raise AttributeError(f"Dataset already has attribute {new_attribute.name}")
        else:
            self.attrs[attr_name] = new_attribute

    def add_data(self, new_data):
        new_data = [np.array(nd) for nd in new_data]
        for attr_name, attr in self.attrs.items():
            if attr_name == 'data':
                attr.add_new_data(new_data)
            else:
                attr.expand_size(new_data)

    def add_labels(self, window, labels):
        self.labels.set_attr_with_window(window, labels)

    def add_temp_labels(self, window, temp_labels):
        self.temp_labels.set_attr_with_window(window, temp_labels)

    def data_from_window(self, window):
        return self.data.get_attr_by_window(window)

    def labels_from_window(self, window):
        return self.labels.get_attr_by_window(window)

    def update_attributes(self, batch_indices, new_attr, sizes):
        """This requires specific implementations (or does it?)"""
        pass

    # def get_temporary_labels(self, i):
    #     temp_data = self.data[i]
    #     temp_labels = [self.data[i][j] if j in self.index.labelled_idx[i]
    #                    else self.temp_labels[i][j] for j in range(len(temp_data))]
    #     return temp_data, temp_labels

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return ActiveLearningSubset(self, [idx])
        elif isinstance(idx, slice):
            idxs = list(range(*idx.indices(len(self))))
            return ActiveLearningSubset(self, idxs)

    def __len__(self):
        return len(self.data)


class OneDimensionalSequenceTaggingDataset(ActiveLearningDataset):

    def __init__(self, data, labels, index_class, semi_supervision_multiplier, padding_token, empty_tag,
                 al_attributes=[], label_form=lambda data_point: data_point.shape):
        super().__init__(data, labels, index_class, semi_supervision_multiplier, al_attributes, label_form=label_form)
        self.empty_tag = empty_tag
        self.padding_token = padding_token

    def update_attributes(self, batch_indices, new_attr_dict, lengths):
        for attr_name, attr_value in new_attr_dict.items():
            for j, i in enumerate(batch_indices):
                assert self.__getattr__(attr_name)[i].shape == attr_value[j][:lengths[j]].shape
                self.__getattr__(attr_name)[i] = attr_value[j][:lengths[j]]

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

        semi_supervision_mask = torch.ones(padded_tags.shape)

        if labels_important:
            # Fill in the words that have not been queried
            for j, sentence_tags in enumerate(padded_tags):
                sentence_index = batch_indices[j]
                for token_idx in range(int(lengths[j])):
                    if token_idx in self.index.labelled_idx[sentence_index]:
                        pass
                    elif token_idx in self.index.temp_labelled_idx[sentence_index]:
                        padded_tags[j, token_idx] = torch.tensor(self.temp_labels[sentence_index][token_idx])
                    elif token_idx in self.index.unlabelled_idx[sentence_index]:
                        padded_tags[j, token_idx] = \
                            torch.exp(torch.tensor(self.last_preds[sentence_index][token_idx]))       # Maybe change this exp?
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
        self.unlabelled_idx = {j: set(range(len(d))) for j, d in enumerate(dataset.data)}
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
        if set(range(*new_window.bounds)).intersection(self.labelled_idx[new_window.i]):
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

    def __getitem__(self, item):

        if isinstance(item, int):
            idx = [item]
        elif isinstance(item, slice):
            idx = list(range(*item.indices(len(self))))
        elif isinstance(item, list):
            idx = item
        else:
            raise TypeError(f"Cannot index SentenceIndex with type {type(item)}")

        return {
            i: {
                "labelled_idx": self.labelled_idx[i],
                "unlabelled_idx": self.unlabelled_idx[i],
                "temp_labelled_idx": self.temp_labelled_idx[i]
            } for i in self.labelled_idx
        }

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

    def savable(self):
        return [int(self.i), list(map(int, self.bounds)), float(self.score)]

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
        self.labelled_ngrams[tuple(new_ngram)] = ngram_annotations
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
                if tuple(new_ngram) in self.labelled_ngrams.keys() and allow_propagation:
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
