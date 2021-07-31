# noinspection PyInterpreter
import json
import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence

TQDM_MODE = True

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
    elif isinstance(thing, bool):
        return 1 if thing else 0


class ActiveLearningSubset:

    def __init__(self, dataset, indices):
        self.indices = indices
        self.dataset = dataset
        self.access_mode = 'data'

    def __getattr__(self, item):
        self.access_mode = item

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.dataset.__getattr__(self.access_mode)[idx]


class ALAttribute:

    def __init__(self, name: str, initialisation: list, cache: bool = False):
        # might change cache to arbitrary length

        self.name = name
        self.attr = initialisation
        self.cache = cache
        if cache:
            self.prev_attr = initialisation.copy()

    def __getitem__(self, idx):
        return self.attr[idx]

    def __setitem__(self, idx, value):
        raise Exception("Use update_attr_with_instance instead of setting item")
        self.attr[idx] = value

    def __len__(self):
        return len(self.attr)

    def generate_nans(self, new_data):
        if isinstance(new_data, list):
            return [np.nan for nd in new_data]
        else:
            return np.nan

    def get_attr_by_window(self, window):
        return self.attr[window.i][window.slice]

    # Put asserts in here!!!
    def update_attr_with_window(self, window, new_attr):
        assert self.get_attr_by_window(window).shape == new_attr.shape
        if self.cache:
            self.prev_attr[window.i][window.slice] = self.attr[window.i][window.slice].copy()
        self.attr[window.i][window.slice] = new_attr

    def update_attr_with_instance(self, i, new_attr):
        # assert self.attr[i].shape == new_attr.shape
        if self.cache:
            try:
                self.prev_attr[i] = self.attr[i].copy()
            except AttributeError:
                self.prev_attr[i] = self.attr[i]
        self.attr[i] = new_attr

    def expand_size(self, new_data):
        self.attr.extend(self.generate_nans(new_data))

    def add_new_data(self, new_data):
        self.attr.extend(new_data)
        
        
class MonteCarloAttribute(ALAttribute):
    
    def __init__(self, name: str, initialisation: list, ML: int, entropy_function, cache: bool = False):
        super(MonteCarloAttrbiute, self).__init__(name, intialisation, cache)
        assert all([len(it) == M for it in initialisation]), \
            f"Initialisation requires list of length M for each instance for {self.name}"
        self.M = M
        self.entropy_function = entropy_function
        
    def __setitem__(self, idx, value):
        raise Exception("Use update_attr_with_instance instead of setting item")
        assert len(value) == self.M, f"Require list of M MC draws for {self.name}"
        self.attr[idx] = value
        
    def get_attr_by_window(self, window):
        return [draw[window.slice] for draw in self.attr[window.i]]
    
        # Put asserts in here!!!
    def update_attr_with_window(self, window, new_attr):
        assert len(new_attr) == self.M, f"Require list of M MC draws for {self.name}"
        # assert self.get_attr_by_window(window).shape == new_attr.shape
        if self.cache:
            for m in range(self.M):
                self.prev_attr[window.i][window.slice][m] = self.attr[window.i][window.slice][m].copy()
        self.attr[window.i][window.slice] = new_attr

    def update_attr_with_instance(self, i, new_attr):
        # assert self.attr[i].shape == new_attr.shape
        assert len(new_attr) == self.M, f"Require list of M MC draws for {self.name}"
        if self.cache:
            try:
                self.prev_attr[i] = self.attr[i].copy()
            except AttributeError:
                self.prev_attr[i] = self.attr[i]
        self.attr[i] = new_attr
        
    def data_uncertainty(self, window):
        attr = self.get_attr_by_window(window)
        entropies = [self.entropy_function(a) for a in attr]
        return np.mean(entropies)
    
    def total_uncertainty(self, window):
        attr = self.get_attr_by_window(window)
        average_attr = np.mean(attr)
        return self.entropy_function(average_attr)
    
    def knowledge_uncertainty(self, window):
        return self.total_uncertainty(window) - self.data_uncertainty(window)
        
        

class ActiveLearningDataset:

    def __init__(self,
                 data,
                 labels,
                 index_class,
                 semi_supervision_multiplier,
                 al_attributes=[],
                 ):

        # When initialised with labels, they must be for all the data.
        # Data without labels (i.e. in the real, non-simulation case), must be added later with self.data

        self.attrs = {
            "data": ALAttribute(name="data", initialisation=[np.array(d) for d in data]),
            "labels": ALAttribute(name="labels", initialisation=[np.array(l) for l in labels] if labels else [np.nan for l in data]),
            "temp_labels": ALAttribute(name="temp_labels", initialisation=[np.nan for l in data]),
            "last_preds": ALAttribute(name="last_preds", initialisation=[np.nan for l in data], cache=True),
        }
        self.attrs.update({ala.name: ala for ala in al_attributes})
        assert all([len(v) == len(data) for v in al_attributes])
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
        self.labels.update_attr_with_window(window, labels)

    def add_temp_labels(self, window, temp_labels):
        self.temp_labels.update_attr_with_window(window, temp_labels)

    def data_from_window(self, window):
        return self.data.get_attr_by_window(window)

    def labels_from_window(self, window):
        return self.labels.get_attr_by_window(window)

    def update_attributes(self, batch_indices, new_attr, sizes):
        raise NotImplementedError

    def update_preds(self, batch_indices, preds, lengths):
        raise NotImplementedError

    def process_scores(self, scores, lengths):
        raise NotImplementedError

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


class DimensionlessDataset(ActiveLearningDataset):

    def __init__(self, data, labels, index_class, semi_supervision_multiplier,
                 data_reading_method = lambda x: x, label_reading_method = lambda x: x,
                 al_attributes = []):
        super(DimensionlessDataset, self).__init__(data, labels, index_class, semi_supervision_multiplier, al_attributes)
        self.data_reading_method = data_reading_method
        self.label_reading_method = label_reading_method

    def update_attributes(self, batch_indices, new_attr_dict, lengths):
        for attr_name, attr_value in new_attr_dict.items():
            for j, i in tqdm(enumerate(batch_indices), disable = not TQDM_MODE):
                self.__getattr__(attr_name).update_attr_with_instance(i, attr_value[j])

    def process_scores(self, scores, lengths=None):
        return scores

    def get_batch(self, batch_indices, labels_important: bool):

        X = torch.vstack([self.data_reading_method(self.data[i]) for i in batch_indices])
        y = torch.vstack([self.label_reading_method(self.labels[i]) for i in batch_indices])
        semi_supervision_mask = torch.ones(y.shape)

        if labels_important:
            # Fill in with semi-supervision labels
            for j, label in enumerate(y):
                instance_index = batch_indices[j]
                if self.index.labelled_idx[instance_index]:
                    pass
                elif self.index.temp_labelled_idx[instance_index]:
                    y[j] = self.temp_labels[instance_index]
                elif self.index.unlabelled_idx[instance_index]:
                    y[j] = torch.exp(torch.tensor(self.last_preds[instance_index]))
                    semi_supervision_mask[j] = self.semi_supervision_multiplier
                else:
                    raise Exception("Instance index does not appear in any of the annotation status lists")

            return X, y, torch.tensor([]), semi_supervision_mask

        return (
            X,
            torch.tensor([]),
            torch.tensor([]), # was lengths
            semi_supervision_mask
        )
    
    
class ImageClassificationDataset(ActiveLearningDataset):

    def __init__(self, data, labels, index_class, semi_supervision_multiplier,
                 al_attributes = []):
        al_attributes.append(ALAttribute(name="data", initialisation=np.array(data)))
        
        super(ImageClassificationDataset, self).__init__(data, labels, index_class, semi_supervision_multiplier, al_attributes)

    def update_attributes(self, batch_indices, new_attr_dict, sizes):
        for attr_name, attr_value in new_attr_dict.items():
            for j, i in enumerate(batch_indices):
                # Implement windows at some point, but not in this class!
                self.__getattr__(attr_name).update_attr_with_instance(i, attr_value[j])

    def process_scores(self, scores, sizes=None):
        return scores

    def get_batch(self, batch_indices, labels_important: bool):

        X = torch.tensor(self.data[batch_indices])
        y = torch.tensor([int(self.labels[i]) for i in batch_indices])
        semi_supervision_mask = torch.ones(y.shape)

        if labels_important:
            # Fill in with semi-supervision labels
            for j, label in enumerate(y):
                instance_index = batch_indices[j]
                if self.index.labelled_idx[instance_index]:
                    pass
                elif self.index.temp_labelled_idx[instance_index]:
                    y[j] = self.temp_labels[instance_index]
                elif self.index.unlabelled_idx[instance_index]:
                    y[j] = torch.exp(torch.tensor(self.last_preds[instance_index]))
                    semi_supervision_mask[j] = self.semi_supervision_multiplier
                else:
                    raise Exception("Instance index does not appear in any of the annotation status lists")

            return X, y, torch.tensor([]), semi_supervision_mask

        return (
            X,
            torch.tensor([]),
            torch.tensor([]), # was lengths
            semi_supervision_mask
        )


class OneDimensionalSequenceTaggingDataset(ActiveLearningDataset):

    def __init__(self, data, labels, index_class, semi_supervision_multiplier, padding_token, empty_tag,
                 al_attributes=[]):
        super().__init__(data, labels, index_class, semi_supervision_multiplier, al_attributes)
        self.empty_tag = empty_tag
        self.padding_token = padding_token

    def update_attributes(self, batch_indices, new_attr_dict, lengths):
        for attr_name, attr_value in new_attr_dict.items():
            for j, i in enumerate(batch_indices):
                self.__getattr__(attr_name)[i] = attr_value[j][:lengths[j]]

    def process_scores(self, scores, lengths):
        return [scores[i, :length].reshape(-1) for i, length in enumerate(lengths)]

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

        return (
            padded_sentences,
            torch.tensor([]),
            lengths,
            semi_supervision_mask
        )


class Index:

    def __init__(self, dataset):
        self.dataset = dataset
        self.number_partially_labelled_instances = 0
        self.labelled_idx = None
        self.unlabelled_idx = None
        self.temp_labelled_idx = None

    def label_instance(self, i):
        raise NotImplementedError

    def label_window(self, window):
        raise NotImplementedError

    def temporarily_label_window(self, window):
        raise NotImplementedError

    def new_window_unlabelled(self, new_window):
        raise NotImplementedError

    def is_partially_labelled(self, i):
        return bool(total_sum(self.labelled_idx[i]))

    def is_partially_temporarily_labelled(self, i):
        return bool(total_sum(self.temp_labelled_idx[i]))

    def has_any_labels(self, i):
        return bool(self.is_partially_labelled(i)) or bool(self.is_partially_temporarily_labelled(i))

    def is_labelled(self, i):
        return not bool(total_sum(self.unlabelled_idx[i]))

    def is_partially_unlabelled(self, i):
        return bool(total_sum(self.unlabelled_idx[i]))

    def get_number_partially_labelled_instances(self):
        return self.number_partially_labelled_instances

    def __getitem__(self, item):

        if isinstance(item, int):
            idx = [item]
        elif isinstance(item, slice):
            idx = list(range(*item.indices(len(self.labelled_idx))))
        elif isinstance(item, list):
            idx = item
        else:
            raise TypeError(f"Cannot index SentenceIndex with type {type(item)}")

        return {
            i: {
                "labelled_idx": self.labelled_idx[i],
                "unlabelled_idx": self.unlabelled_idx[i],
                "temp_labelled_idx": self.temp_labelled_idx[i]
            } for i in item
        }


class DimensionlessIndex(Index):

    def __init__(self, dataset):
        super(DimensionlessIndex, self).__init__(dataset)
        self.labelled_idx = {j: False for j in range(len(dataset.data))}
        self.unlabelled_idx = {j: True for j, d in enumerate(dataset.data)}
        self.temp_labelled_idx = {j: False for j in range(len(dataset.data))}

    def label_instance(self, i):
        self.number_partially_labelled_instances += 1
        self.labelled_idx[i] = True
        self.unlabelled_idx[i] = False

    def label_window(self, window):
        if not isinstance(window, DimensionlessAnnotationUnit):
            raise TypeError("DimensionlessIndex requires DimensionlessAnnotationUnit")
        self.label_instance(window.i)

    def temporarily_label_window(self, window):
        self.unlabelled_idx[window.i] = False
        self.temp_labelled_idx[window.i] = True

    def new_window_unlabelled(self, new_window):
        return not self.labelled_idx[new_window.i]

    def make_nan_if_labelled(self, i, scores):
        # Zero-dimensional datapoints => no partial labelling to worry about
        return scores

    def save(self, save_path):
        with open(os.path.join(save_path, "agent_index.pk"), "w") as f:
            json.dump(
                {
                    "labelled_idx": self.labelled_idx,
                    "unlabelled_idx": self.unlabelled_idx,
                    "temporarily_labelled_idx": self.temp_labelled_idx,
                },
                f
            )


class SentenceIndex(Index):

    def __init__(self, dataset):
        super(SentenceIndex, self).__init__(dataset)
        self.labelled_idx = {j: set() for j in range(len(dataset.data))}
        self.unlabelled_idx = {j: set(range(len(d))) for j, d in enumerate(dataset.data)}
        self.temp_labelled_idx = {j: set() for j in range(len(dataset.data))}

    def label_instance(self, i):
        if not self.is_partially_labelled(i):
            self.number_partially_labelled_instances += 1
        self.labelled_idx[i] = set(range(len(self.dataset.data[i])))
        self.unlabelled_idx[i] = set()

    def label_window(self, window):
        if not self.labelled_idx[window.i] and window.size > 0:
            self.number_partially_labelled_instances += 1
        self.labelled_idx[window.i].update(range(*window.bounds))
        self.unlabelled_idx[window.i] -= window.get_index_set()
        self.temp_labelled_idx[window.i] -= window.get_index_set()

    def temporarily_label_window(self, window):
        self.unlabelled_idx[window.i] -= window.get_index_set()
        self.temp_labelled_idx[window.i].update(window.get_index_set())

    def new_window_unlabelled(self, new_window):
        if new_window.get_index_set().intersection(self.labelled_idx[new_window.i]):
            return False
        else:
            return True

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


class AnnotationUnit:

    def __init__(self, data_index, bounds, score):
        self.i = data_index
        self.bounds = bounds
        self.score = score

    def savable(self):
        return [self.i, self.bounds, self.score]

    def get_index_set(self):
        return None


class SentenceSubsequence(AnnotationUnit):

    def __init__(self, data_index, bounds, score):
        super(SentenceSubsequence, self).__init__(data_index, bounds, score)
        self.size = bounds[1] - bounds[0]
        self.slice = slice(*bounds)

    def savable(self):
        return [int(self.i), list(map(int, self.bounds)), float(self.score)]

    def get_index_set(self):
        return set(range(*self.bounds))


class DimensionlessAnnotationUnit(AnnotationUnit):

    def __init__(self, data_index, bounds, score):
        super(DimensionlessAnnotationUnit, self).__init__(data_index, bounds, score)
        self.size = 1
        self.slice = slice(None)

    def savable(self):
        return [int(self.i), None, float(self.score)]

    def get_index_set(self):
        return {0}
    
    
class EarlyStopper:

    def __init__(self, model, patience: int):  # , maximise: bool):
        """
        An early stopping & callback class.
        patience is an integer, the number of epochs that a non-optimal statistic is allowed (adding number of steps soon)
        maximise is set to True for scores, False for losses
        """
        self.patience = patience
        # self.maximise = maximise
        self.model_state_dict = None
        self.model = model
        self.saved_epoch = 0
        self.scores = []
        self.min_score = float('inf')

    def is_overfitting(self, score):
        '''
        Unused
        '''
        scores = self.scores
        if len(scores) < self.patience:
            self.scores.append(score)
            return False

        if score < self.min_score:
            self.model_state_dict = self.model.state_dict()
            self.min_score = score

        scores.append(score)
        all_increasing = True
        s0 = scores[0]
        for s1 in scores[1:]:
            if s0 >= s1:
                all_increasing = False
                break
            s0 = s1
        self.scores = scores[1:]

        if all_increasing:
            print('reloading model\n')
            self.model.load_state_dict(self.model_state_dict)

        return all_increasing

    def check_stop(self, stats_list):
        
        if len(stats_list) > self.patience:
            if stats_list[-1] < stats_list[-2]:
                self.model_state_dict = self.model.state_dict()
                self.saved_epoch = len(stats_list)

        if self.patience < 0 or len(stats_list) < self.patience:
            return False
        if stats_list[-self.patience:] == sorted(stats_list[-self.patience:]):
            print(f'reloading from epoch {self.saved_epoch}')
            self.model.load_state_dict(self.model_state_dict)
            return True
        else:
            return False
