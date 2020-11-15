import logging
from random import sample
from typing import List, Dict

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

    def label_window(self, i, r):
        if not self.labelled_idx[i] and r[1] - r[0] > 0:
            self.__number_partially_labelled_sentences += 1
        self.labelled_idx[i].update(range(r[0], r[1]))
        self.unlabelled_idx[i] -= set(range(r[0], r[1]))

    def is_partially_labelled(self, i):
        return len(self.labelled_idx[i]) > 0

    def is_labelled(self, i):
        return len(self.unlabelled_idx[i]) == 0

    def is_partially_unlabelled(self, i):
        return len(self.unlabelled_idx[i]) > 0

    def get_number_partially_labelled_sentences(self):
        return self.__number_partially_labelled_sentences

    def make_nan_if_labelled(self, i, scores):
        res = []
        for j in range(len(scores)):
            if j in self.unlabelled_idx[i]:
                res.append(scores[j])
            else:
                res.append(float('nan'))
        return res


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
        self.num = 1

    def init(self, n):
        logging.info('starting random init')
        self.random_init(n)
        self.update_datasets()
        self.num = 0
        logging.info('finished random init')

    def step(self):
        logging.info('step')
        sentence_scores: Dict[int, List[float]] = self.get_sentence_scores()
        self.update_index(sentence_scores)
        self.update_datasets()
        logging.info('finished step')

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
        return self.selector.get_batch(batch)

    # @staticmethod
    # def purify_entries(entries):
    #    """Sort and remove disjoint entries of form [([list, of, word, idx], score), ...]"""
    #    start_entries = sorted(entries, key=lambda x: x[-1], reverse=True)
    #    final_entries = []
    #    highest_idx = set()
    #    for entry in start_entries:
    #        if highest_idx.intersection(entry[0]):
    #            pass
    #        else:
    #            highest_idx = highest_idx.union(entry[0])
    #            final_entries.append(entry)
    #    return final_entries

    def update_index(self, sentence_scores):
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
        logging.info("update index")

        window_scores = []
        for i, word_scores in tqdm(sentence_scores.items()):
            # Skip if already all Nones
            if self.index.is_labelled(i):
                continue
            windows = self.selector.score_extraction(word_scores)
            window_scores.extend([(i, window[0], window[1]) for window in windows])

        window_scores.sort(key=lambda e: e[-1], reverse=True)

        window_scores = window_scores[:self.round_size]

        n_spent = 0
        for i, r, _ in window_scores:
            cost = r[1] - r[0]
            self.budget -= cost
            if self.budget < 0:
                logging.warning('no more budget left!')
                break
            n_spent += cost
            self.index.label_window(i, r)

        logging.info(f'added {n_spent} words to index mapping')

    def get_sentence_scores(self):
        """
        Score unlabelled instances in terms of their suitability to be labelled next.
        Add the highest scoring instance indices in the dataset to self.labelled_idx
        """

        if self.budget <= 0:
            logging.warning('no more budget left!')

        sentence_scores = {}
        logging.info('get sentence scores')
        for batch_index in tqdm(self.unlabelled_set):
            # Use normal get_batch here since we don't want to fill anything in, but it doesn't really matter
            # for functionality
            batch = [self.train_set[i] for i in batch_index]
            sentences, tokens, _, lengths = self.helper.get_batch(batch)
            batch_scores = self.acquisition.score(sentences=sentences, lengths=lengths, tokens=tokens)

            for j, i in enumerate(batch_index):
                sentence_scores[i] = self.index.make_nan_if_labelled(i, batch_scores[j])

        return sentence_scores

    def update_datasets(self):
        unlabelled_sentences = set()
        labelled_sentences = set()

        logging.info("update datasets")
        for i in tqdm(range(len(self.train_set))):
            if self.index.is_partially_unlabelled(i):
                unlabelled_sentences.add(i)
            if self.index.is_partially_labelled(i):
                labelled_sentences.add(i)

        unlabelled_subset = Subset(self.train_set, list(unlabelled_sentences))
        labelled_subset = Subset(self.train_set, list(labelled_sentences))

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

