import logging
import random
from random import sample
from typing import List, Dict
import os
import json

from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm

TQDM_MODE = True


class ActiveLearningAgent:
    def __init__(
        self, train_set, batch_size, selector_class, model, device, budget_prop=0.5
    ):

        self.batch_size = batch_size
        self.train_set = train_set
        self.selector = selector_class
        self.selector.assign_agent(self)
        self.model = model
        self.device = device
        self.budget_prop = budget_prop

        num_units = sum(
            [instance.size for instance in self.train_set.data]
        )  # parameterise this
        self.budget = num_units * budget_prop
        self.initial_budget = self.budget

        self.unlabelled_set = None
        self.labelled_set = None
        self.num = 1
        self.round_all_word_scores = {}

    def init(self, n, seed=42):
        logging.info("starting random init")
        self.random_init(n, seed)
        self.update_datasets()
        self.num = 0
        logging.info("finished random init")

    def step(self):
        logging.info("step")
        # TODO: type *everything*
        self.update_dataset_attributes()
        self.update_index()
        self.update_datasets()
        logging.info("finished step")

    def budget_spent(self):
        return self.initial_budget - self.budget

    def num_instances(self):
        return sum([len(l) for l in self.labelled_set])

    def save(self, save_path):
        self.train_set.index.save(save_path)
        self.selector.save(save_path)
        with open(os.path.join(save_path, "all_word_scores_no_nan.json"), "w") as f:
            json.dump(self.round_all_word_scores, f)

    def random_init(self, num_instances, seed):
        """
        Randomly initialise self.labelled_idx dictionary
        """
        init_sampler = random.Random(seed)
        randomly_selected_indices = init_sampler.sample(
            list(self.train_set.index.unlabelled_idx.keys()), num_instances
        )

        budget_spent = 0
        for i in randomly_selected_indices:
            self.train_set.index.label_instance(i)
            budget_spent += self.train_set.data[i].size

        self.budget -= budget_spent

        logging.info(
            f"""
            total instances: {len(self.train_set)}  |   total words: {self.budget + budget_spent}
            initialised with {budget_spent} words  |   remaining word budget: {self.budget}
            """
        )

    def update_index(self):
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
            meaning words 5, 6, 7 of word j are chosen to be labelled.def update_index
        """
        logging.info("update index")

        all_windows = []
        for i in tqdm(range(len(self.train_set)), disable=not TQDM_MODE):
            if self.train_set.index.is_labelled(i):
                continue
            windows = self.selector.window_generation(i, self.train_set)
            all_windows.extend(windows)

        all_windows.sort(key=lambda e: e.score, reverse=True)
        best_windows, budget_spent = self.selector.select_best(all_windows)
        self.budget -= budget_spent
        if self.budget < 0:
            logging.warning("no more budget left!")

        total_units = 0
        for window in best_windows:
            total_units += window.size
            self.train_set.index.label_window(window)

        logging.info(f"added {total_units} words to index mappingl")

        # No more windows of this size left
        if total_units < self.selector.round_size:
            self.selector.reduce_window_size()

    def update_dataset_attributes(self):
        """
        Score unlabelled instances in terms of their suitability to be labelled next.
        Add the highest scoring instance indices in the dataset to self.labelled_idx
        """

        if self.budget <= 0:
            logging.warning("no more budget left!")

        # logging.info('get sentence scores')
        for batch_indices in tqdm(
            self.unlabelled_set + self.labelled_set, disable=not TQDM_MODE
        ):
            # Use normal get_batch here since we don't want to fill anything in, but it doesn't really matter
            # for functionality
            instances, _, lengths, _ = [
                a.to(self.device)
                for a in self.train_set.get_batch(batch_indices, labels_important=False)
            ]
            model_attrs = self.model(instances.float(), anneal=True)
            model_attrs = {k: v.detach().cpu().numpy() for k, v in model_attrs.items()}
            self.train_set.update_attributes(batch_indices, model_attrs, lengths)

    def update_datasets(self):
        unlabelled_instances = set()
        labelled_instances = set()

        logging.info("update datasets")
        for i in tqdm(range(len(self.train_set)), disable=not TQDM_MODE):
            if self.train_set.index.is_partially_unlabelled(i):
                unlabelled_instances.add(i)
            if self.train_set.index.has_any_labels(i):
                labelled_instances.add(i)

        self.unlabelled_set = list(
            BatchSampler(
                SubsetRandomSampler(list(unlabelled_instances)),
                self.batch_size,
                drop_last=False,
            )
        )

        self.labelled_set = list(
            BatchSampler(
                SubsetRandomSampler(list(labelled_instances)),
                self.batch_size,
                drop_last=False,
            )
        )

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


class SubsetSelectionAgent:
    def __init__(
        self, train_set, batch_size, selector_class, model, device, budget_prop=0.5
    ):

        self.batch_size = batch_size
        self.train_set = train_set
        self.selector = selector_class
        self.selector.assign_agent(self)
        self.model = model
        self.device = device
        self.budget_prop = budget_prop

        num_units = sum(
            [instance.size for instance in self.train_set.data]
        )  # parameterise this
        self.budget = num_units * budget_prop
        self.initial_budget = self.budget

        self.unlabelled_set = None
        self.labelled_set = None
        self.num = 1
        self.round_all_word_scores = {}

    def init(self, n, seed=42):
        logging.info("starting random init")
        self.random_init(n, seed)
        self.update_datasets()
        self.num = 0
        logging.info("finished random init")

    def step(self, update_dataset=True):
        logging.info("step")
        # TODO: type *everything*
        if update_dataset:
            self.update_dataset_attributes()
        self.update_index()
        self.update_datasets()
        logging.info("finished step")

    def budget_spent(self):
        return self.initial_budget - self.budget

    def num_instances(self):
        return sum([len(l) for l in self.labelled_set])

    def save(self, save_path):
        self.train_set.index.save(save_path)
        self.selector.save(save_path)
        with open(os.path.join(save_path, "all_word_scores_no_nan.json"), "w") as f:
            json.dump(self.round_all_word_scores, f)

    def random_init(self, num_instances, seed):
        """
        Randomly initialise self.labelled_idx dictionary
        """
        init_sampler = random.Random(seed)
        randomly_selected_indices = init_sampler.sample(
            list(self.train_set.index.unlabelled_idx.keys()), num_instances
        )

        budget_spent = 0
        for i in randomly_selected_indices:
            self.train_set.index.label_instance(i)
            budget_spent += self.train_set.data[i].size

        self.budget -= budget_spent

        logging.info(
            f"""
            total instances: {len(self.train_set)}  |   total words: {self.budget + budget_spent}
            initialised with {budget_spent} words  |   remaining word budget: {self.budget}
            """
        )

    def update_index(self):
        logging.info("update index")

        all_windows = []
        for i in range(len(self.train_set)):
            if self.train_set.index.is_labelled(i):
                continue
            windows = self.selector.window_generation(i, self.train_set)
            all_windows.extend(windows)

        best_windows, budget_spent = self.selector.select_best(all_windows)
        self.budget -= budget_spent
        if self.budget < 0:
            logging.warning("no more budget left!")

        total_units = 0
        for window in best_windows:
            total_units += window.size
            self.train_set.index.label_window(window)

        logging.info(f"added {total_units} words to index mappingl")

        # No more windows of this size left
        if total_units < self.selector.round_size:
            self.selector.reduce_window_size()

    def update_dataset_attributes(self):
        """
        Score unlabelled instances in terms of their suitability to be labelled next.
        Add the highest scoring instance indices in the dataset to self.labelled_idx
        """

        if self.budget <= 0:
            logging.warning("no more budget left!")

        # logging.info('get sentence scores')
        for batch_indices in tqdm(
            self.unlabelled_set + self.labelled_set, disable=not TQDM_MODE
        ):
            # Use normal get_batch here since we don't want to fill anything in, but it doesn't really matter
            # for functionality
            instances, _, lengths, _ = [
                a.to(self.device)
                for a in self.train_set.get_batch(batch_indices, labels_important=False)
            ]
            model_attrs = self.model(instances, anneal=True)
            model_attrs = {k: v.detach().cpu().numpy() for k, v in model_attrs.items()}
            self.train_set.update_attributes(batch_indices, model_attrs, lengths)

    def update_datasets(self):
        unlabelled_instances = set()
        labelled_instances = set()

        logging.info("update datasets")
        for i in range(len(self.train_set)):
            if self.train_set.index.is_partially_unlabelled(i):
                unlabelled_instances.add(i)
            if self.train_set.index.has_any_labels(i):
                labelled_instances.add(i)

        self.unlabelled_set = list(
            BatchSampler(
                SubsetRandomSampler(list(unlabelled_instances)),
                self.batch_size,
                drop_last=False,
            )
        )

        self.labelled_set = list(
            BatchSampler(
                SubsetRandomSampler(list(labelled_instances)),
                self.batch_size,
                drop_last=False,
            )
        )

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
