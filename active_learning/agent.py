import logging
from random import sample

import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler, Subset
from tqdm import tqdm


class SentenceIndex:

    def __init__(self, train_set):
        self.__number_partially_labelled_sentences = 0
        self.labelled_idx = {j: set() for j in range(len(train_set))}
        self.unlabelled_idx = {j: set(range(len(train_set[j][0]))) for j in range(len(train_set))}

    def label_sentence(self, i):
        self.labelled_idx[i] = self.unlabelled_idx[i]
        self.__number_partially_labelled_sentences += 1
        self.unlabelled_idx[i] = set()

    def is_partially_labelled(self, i):
        return len(self.labelled_idx[i]) > 0

    def is_partially_unlabelled(self, i):
        return len(self.unlabelled_idx[i]) > 0

    def get_number_partially_labelled_sentences(self):
        return self.__number_partially_labelled_sentences


class ActiveLearningAgent:

    def __init__(
            self,
            train_set,
            batch_size,
            round_size,
            acquisition_class,
            selector_class,
            helper,
            device
    ):
        """
        train_set: loaded from pickle
        test_data: loaded from pickle

        score: function used to score single words
        Inputs:
            output: Tensor, shape (batch size, sequence length, number of possible tags), model outputs of all instances
        Outputs:
            a score, with higher meaning better to pick

        budget: total number of elements we can label (words)
        round_size: total number instances we label each round (sentences)
        """

        # !!! Right now, following the pipeline, we assume we can do word-wise aggregation of scores
        # This might have to change....

        self.round_size = round_size
        self.batch_size = batch_size
        self.train_set = train_set
        self.acquisition = acquisition_class
        self.selector = selector_class
        self.helper = helper
        self.device = device

        # Dictionaries mapping {sentence idx: [list, of, word, idx]} for labelled and unlabelled words
        self.index = SentenceIndex(train_set)

        num_tokens = sum([len(sentence) for sentence, _, _ in self.train_set])
        self.budget = num_tokens
        self.initial_budget = self.budget

        self.unlabelled_set = None
        self.labelled_set = None

    def init(self, n):
        print("Starting random init")
        self.random_init(n)
        self.update_datasets()
        print("Finished random init")

    def budget_spent(self):
        return self.initial_budget - self.budget

    def random_init(self, num_sentences):
        """
        Randomly initialise self.labelled_idx dictionary
        """
        randomly_selected_indices = sample(list(self.index.unlabelled_idx.keys()), num_sentences)

        budget_spent = 0
        for i in randomly_selected_indices:
            self.index.label_sentence(i)
            budget_spent += len(self.train_set[i][0])

        self.budget -= budget_spent

        print(
            f"""
            Total sentences: {len(self.train_set)}  |   Total words: {self.budget + budget_spent}
            Initialised with {budget_spent} words  |   Remaining word budget: {self.budget}
            """)

    def get_batch(self, i):
        # Use selector get_batch here as we want to fill things in if needed
        batch = [self.train_set[j] for j in self.labelled_set[i]]
        batch_items = self.selector.get_batch(batch, self)
        return tuple(a.to(self.device) for a in batch_items)

    @staticmethod
    def purify_entries(entries):
        """Sort and remove disjoint entries of form [([list, of, word, idx], score), ...]"""
        start_entries = sorted(entries, key=lambda x: x[-1], reverse=True)
        final_entries = []
        highest_idx = set()
        for entry in start_entries:
            if highest_idx.intersection(entry[0]):
                pass
            else:
                highest_idx = highest_idx.union(entry[0])
                final_entries.append(entry)
        return final_entries

    def extend_indices(self, sentence_scores):
        """
        After a full pass on the unlabelled pool, apply a policy to get the top scoring phrases and add them to
        self.labelled_idx.

        Input:
            sentence_scores: {j: [list, of, scores, per, word, None, None]} where None means the word has alread been
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

        temp_score_list = []

        print("\nExtending indices")
        for sentence_idx, scores_list in tqdm(sentence_scores.items()):
            # Skip if already all Nones
            if all([type(i) == type(None) for i in scores_list]):
                continue
            entries = self.selector.score_extraction(scores_list)
            entries = self.purify_entries(
                entries)  # entries = [([list, of, word, idx], score), ...] that can be compared to temp_score_list
            temp_score_list.extend([(sentence_idx, entry[0], entry[1]) for entry in entries])

        temp_score_list.sort(key=lambda e: e[-1], reverse=True)
        temp_score_list = temp_score_list[:self.round_size]

        j = 0
        for sentence_idx, word_inds, score in temp_score_list:
            self.budget -= len(word_inds)
            if self.budget < 0:
                print("No more budget left!")
                break
            j += len(word_inds)
            self.index.labelled_idx[sentence_idx] = self.index.labelled_idx[sentence_idx].union(word_inds)
            for w in word_inds:
                self.index.unlabelled_idx[sentence_idx].remove(w)

        print(f"Added {j} words to index mapping")
        return temp_score_list

    def get_all_scores(self):
        """
        Score unlabelled instances in terms of their suitability to be labelled next.
        Add the highest scoring instance indices in the dataset to self.labelled_idx
        """
        if self.budget <= 0:
            logging.warning("No more budget left!")

        sentence_scores = {}

        print("\nUpdating indices")
        for batch_index in tqdm(self.unlabelled_set):
            # Use normal get_batch here since we don't want to fill anything in, but it doesn't really matter
            # for functionality
            batch = [self.train_set[j] for j in batch_index]
            sentences, tokens, _, lengths = self.helper.get_batch(batch, self.device)
            word_scores = self.acquisition.score(sentences=sentences, sentence_lengths=lengths, tokens=tokens)
            for i, b in enumerate(batch_index):
                sentence_scores[b] = [
                    float(word_scores[i][j]) if j in self.index.unlabelled_idx[b] else None
                    for j in range(lengths[i])
                ]  # scores of unlabelled words --> float, scores of labelled words --> None

        return sentence_scores

    def update_indices(self):

        sentence_scores = self.get_all_scores()
        temp_score_list = self.extend_indices(sentence_scores)

        return sentence_scores, temp_score_list

    def update_datasets(self):
        unlabelled_sentences = set()
        labelled_sentences = set()

        print("\nCreating extended dataset samplers")

        for i in tqdm(range(len(self.train_set))):
            if self.index.is_partially_labelled(i):
                unlabelled_sentences.add(i)
            if self.index.is_partially_unlabelled(i):
                labelled_sentences.add(i)

        unlabelled_subset = Subset(self.train_set, list(unlabelled_sentences))
        labelled_subset = Subset(self.train_set, list(labelled_sentences))

        self.unlabelled_set = \
            list(BatchSampler(SubsetRandomSampler(unlabelled_subset.indices), self.batch_size, drop_last=False))

        self.labelled_set = \
            list(BatchSampler(SubsetRandomSampler(labelled_subset.indices), self.batch_size, drop_last=False))

    def __iter__(self):
        # DONT FORGET: DO self.selector.get_batch on self.train_data
        return (
            self.labelled_set[i]
            for i in torch.randperm(len(self.labelled_set))
        )

    def __len__(self):
        return len(self.labelled_set)
