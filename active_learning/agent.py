import logging
from random import sample
from typing import List, Dict
import os
import json

from torch.utils.data import BatchSampler, SubsetRandomSampler, Subset
from tqdm import tqdm
from .util_classes import SentenceSubsequence, OneDimensionalSequenceTaggingDataset


class ActiveLearningAgent:

    def __init__(
            self,
            train_set,
            batch_size,
            round_size,
            acquisition_class,
            selector_class,
            helper,
            model,
            device,
            propagation_mode,
            budget_prop=0.5
    ):
        """
        score: function used to score single words
        Inputs:
            output: Tensor, shape (batch size, sequence length, number of possible tags), model outputs of all instances
        Outputs:
            a score, with higher meaning better to pick

        budget: total number of elements we can label (words)
        round_size: total number instances we label each round (sentences)
        """

        self.round_size = round_size
        self.batch_size = batch_size
        self.train_set = train_set
        self.acquisition = acquisition_class
        self.selector = selector_class
        self.selector.assign_agent(self)
        self.helper = helper
        self.model = model
        self.device = device
        self.propagation_mode = propagation_mode
        self.budget_prop = budget_prop
        self.window_class = SentenceSubsequence     # parameterise this

        num_units = sum([len(instance) for instance in self.train_set.data])
        self.budget = num_units * budget_prop
        self.initial_budget = self.budget

        self.unlabelled_set = None
        self.labelled_set = None
        self.num = 1
        self.round_all_word_scores = {}

    def init(self, n):
        logging.info('starting random init')
        self.random_init(n)
        self.update_datasets()
        self.num = 0
        logging.info('finished random init')

    def step(self):
        logging.info('step')
        instance_scores: Dict[int, List[float]] = self.get_unitwise_scores()
        self.update_index(instance_scores)
        self.update_datasets()
        logging.info('finished step')

    def budget_spent(self):
        return self.initial_budget - self.budget

    def save(self, save_path):
        self.train_set.index.save(save_path)
        self.selector.save(save_path)
        with open(os.path.join(save_path, "all_word_scores_no_nan.json"), "w") as f:
            json.dump(self.round_all_word_scores, f)

    def random_init(self, num_instances):
        """
        Randomly initialise self.labelled_idx dictionary
        """
        randomly_selected_indices = sample(list(self.train_set.index.unlabelled_idx.keys()), num_instances)

        budget_spent = 0
        for i in randomly_selected_indices:
            self.train_set.index.label_instance(i)
            budget_spent += len(self.train_set.data[i])

        self.budget -= budget_spent

        logging.info(
            f"""
            total instances: {len(self.train_set)}  |   total words: {self.budget + budget_spent}
            initialised with {budget_spent} words  |   remaining word budget: {self.budget}
            """)

    def update_index(self, instance_scores):
        """
        After a full pass on the unlabelled pool, apply a policy to get the top scoring phrases and add them to
        self.labelled_idx.

        Input:
            sentence_scores: {j: [list, of, scores, per, word, nan, nan]} where nan means the word has alread been
            labelled i.e. full list of scores/Nones
        Output:
            No output, but extends self.labelled_idx:
            {
                j: [5, 6, 7],
                i: [1, 2, 3, 8, 9, 10, 11],
                ...
            }
            meaning words 5, 6, 7 of word j are chosen to be labelled.
        """
        logging.info("update index")

        all_windows = []
        for i, word_scores in tqdm(instance_scores.items()):
            # Skip if already all labelled
            if self.train_set.index.is_labelled(i):
                continue
            windows = self.selector.score_extraction(word_scores)
            all_windows.extend(
                [self.window_class(i, window["bounds"], window["score"]) for window in windows]
            )

        all_windows.sort(key=lambda e: e.score, reverse=True)
        best_window_scores, labelled_ngrams_lookup, budget_spent = \
            self.selector.select_best(all_windows, self.propagation_mode != 0)
        self.budget -= budget_spent
        if self.budget < 0:
            logging.warning('no more budget left!')

        labelled_ngrams_lookup = {k: v for k,v in labelled_ngrams_lookup.items() if sum(v)}

        total_units = 0
        for window in best_window_scores:
            total_units += window.size
            self.train_set.index.label_window(window)

        if self.propagation_mode:
            # This must come after labelling initial set
            propagated_windows = self.propagate_labels(all_windows, labelled_ngrams_lookup)

            for window in propagated_windows:
                total_units += window.size
                self.train_set.index.temporarily_label_window(window)

        logging.info(f'added {total_units} words to index mapping, of which {budget_spent} manual')

        # No more windows of this size left
        if total_units < self.round_size:
            self.selector.reduce_window_size()

    def propagate_labels(self, all_windows, labelled_ngrams_lookup):

        out_windows = []

        for window in all_windows:
            if self.train_set.index.new_window_unlabelled(window):
                units = self.train_set.data_from_window(window)
                if units in labelled_ngrams_lookup.keys():
                    out_windows.append(window)
                    self.train_set.add_temp_labels(window, labelled_ngrams_lookup[units])  # Move lookup to dataset class too!!!

        return out_windows

    def get_unitwise_scores(self):
        """
        Score unlabelled instances in terms of their suitability to be labelled next.
        Add the highest scoring instance indices in the dataset to self.labelled_idx
        """

        if self.budget <= 0:
            logging.warning('no more budget left!')

        instance_scores_no_nan = {}
        # logging.info('get sentence scores')
        for batch_indices in tqdm(self.unlabelled_set):
            # Use normal get_batch here since we don't want to fill anything in, but it doesn't really matter
            # for functionality
            batch = [self.train_set.data[i] for i in batch_indices]
            instances, _, lengths = [a.to(self.device) for a in self.train_set.get_batch(batch, labels_important=False)]
            preds = self.model(instances, anneal=True).detach().cpu()
            batch_scores = self.acquisition.score(preds=preds, lengths=lengths)
            self.train_set.update_preds(batch_indices, preds, lengths)
            for j, i in enumerate(batch_indices):
                instance_scores_no_nan[i] = batch_scores[j].tolist()

        instance_scores = {}
        for i, scores in instance_scores_no_nan.items():
            instance_scores[i] = self.train_set.index.make_nan_if_labelled(i, scores)

        self.round_all_word_scores = instance_scores_no_nan
        return instance_scores

    def update_datasets(self):
        unlabelled_instances = set()
        labelled_instances = set()

        logging.info("update datasets")
        for i in tqdm(range(len(self.train_set))):
            if self.train_set.index.is_partially_unlabelled(i):
                unlabelled_instances.add(i)
            if self.propagation_mode == 2:
                if self.train_set.index.is_partially_labelled(i):
                    labelled_instances.add(i)
            else:
                if self.train_set.index.has_any_labels(i):
                    labelled_instances.add(i)

        unlabelled_subset = Subset(self.train_set, list(unlabelled_instances))
        labelled_subset = Subset(self.train_set, list(labelled_instances))

        self.unlabelled_set = \
            list(BatchSampler(SubsetRandomSampler(unlabelled_subset.indices), self.batch_size, drop_last=False))

        self.labelled_set = \
            list(BatchSampler(SubsetRandomSampler(labelled_subset.indices), self.batch_size, drop_last=False))

    def __iter__(self):
        return self

    def __next__(self):
        num = self.num
        self.num += 1
        if num < 0:
            raise StopIteration
        if num > 0:
            self.step()
        if self.budget <= 0:
            self.num = -1
        return self.budget
