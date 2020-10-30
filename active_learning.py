import argparse
from utils import *
from training_utils import *
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
from tqdm import tqdm
import copy
import logging
import random

def configure_al_agent(args, device, model, train_data, test_data):

    num_sentences_init = int(len(train_data) * args.initprop)
    round_size = int(args.roundsize)

    if args.window != -1:
        selector = WordWindowSelector
    else:
        selector = FullSentenceSelector

    if args.acquisition == 'rand':
        acquisition_class = RandomBaselineAcquisition
    elif args.acquisition == 'lc':
        acquisition_class = LowestConfidenceAcquisition
    elif args.acquisition == 'maxent':
        acquisition_class = MaximumEntropyAcquisition
    elif args.acquisition == 'bald':
        acquisition_class = BALDAcquisition
    else:
        raise ValueError(args.acquisition)

    class AgentClass(selector, acquisition_class):

        def __init__(
            self,
            train_data,
            test_data,
            num_sentences_init,
            round_size,
            batch_size,
            model,
            device,
        ):
            super(AgentClass, self).__init__(
                train_data=train_data,
                test_data=test_data,
                num_sentences_init=num_sentences_init,
                round_size=round_size,
                batch_size=args.batch_size,
                model=model,
                device=device
            )

    agent = AgentClass(
        train_data=train_data,
        test_data=test_data,
        num_sentences_init=num_sentences_init,
        round_size=round_size,
        batch_size=args.batch_size,
        model=model,
        device=device
    )

    return agent


class ActiveLearningDataset(object):
    def __init__(
        self,
        train_data,
        test_data,
        batch_size,
        round_size,
        num_sentences_init,
        model,
        drop_last=False,
        device="cuda",
        **kwargs
    ):
        """
        train_data: loaded from pickle
        test_data: loaded from pickle

        word_scoring_func: function used to score single words
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
        self.drop_last = drop_last

        self.train_data = train_data
        self.test_data = test_data

        # Exact copy for now. Will only change instances pointed to by self.labelled_batch_idx
        self.autolabelled_data = copy.deepcopy(self.train_data)

        # Short term storage of labels used for filling in sentence blanks
        self.st_labels = {j: [] for j in range(len(self.train_data))}

        # Dictionaries mapping {sentence idx: [list, of, word, idx]} for labelled and unlabelled words
        self.labelled_idx = {j: set() for j in range(len(self.train_data))}
        self.unlabelled_idx = {
            j: set(range(len(train_data[j][0]))) for j in range(len(self.train_data))
        }

        self.device = device
        print("Starting random init")
        self.random_init(num_sentences=num_sentences_init)
        self.update_datasets(model)
        print("Finished random init")



    def random_init(self, num_sentences):
        """
        Randomly initialise self.labelled_idx dictionary
        """
        num_words_labelled = 0
        thres = num_sentences / len(self.train_data)
        randomly_selected_indices = [
            i for i in range(len(self.train_data)) if random.random() < thres
        ]
        for j in randomly_selected_indices:
            l = len(self.train_data[j][0])
            self.labelled_idx[j] = set(range(l))
            self.unlabelled_idx[j] = set()
            num_words_labelled += l
        total_words = sum([len(train_data[i][0]) for i in range(len(self.train_data))])
        self.budget = total_words - num_words_labelled
        print(
            f"""
        Total sentences: {len(self.train_data)}  |   Total words: {total_words}
        Initialised with {num_words_labelled} words  |   Remaining word budget: {self.budget}
        """
        )

    @staticmethod
    def purify_entries(entries):
        """Sort and remove disjoint entries of form [([list, of, word, idx], score), ...]"""
        start_entries = sorted(entries, key = lambda x: x[-1])
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
        After a full pass on the unlabelled pool, apply a policy to get the top scoring phrases and add them to self.labelled_idx.

        Input:
            sentence_scores: {j: [list, of, scores, per, word, None, None]} where None means the word has alread been labelled
                                                                                i.e. full list of scores/Nones
        Output:
            No output, but extends self.labelled_idx:
            {
                j: [5, 6, 7],
                i: [1, 2, 3, 8, 9, 10, 11],
                ...
            }
            meaning words 5, 6, 7 of word j are chosen to be labelled.
        """
        temp_score_list = [
            (-1, [], -np.inf) for _ in range(self.round_size)
        ]  # (sentence_idx, [list, of, word, idx], score) to be added to self.labelled_idx at the end
        scores_from_temp_list = lambda x=temp_score_list: [
            a[-1] for a in x
        ]  # Get list of scores

        j = 0
        print("\nExtending indices")
        for sentence_idx, scores_list in tqdm(sentence_scores.items()):
            # Skip if entirely Nones
            if all([type(j) == type(None) for j in scores_list]):
                continue
            entries = self.score_extraction(scores_list)
            entries = self.purify_entries(entries)
            # entries = [([list, of, word, idx], score), ...] that can be compared to temp_score_list
            for entry in entries:
                if entry[-1] > temp_score_list[0][-1]:
                    temp_score_list[0] = (sentence_idx, entry[0], entry[1])
                    temp_score_list.sort(key=lambda y: y[-1])
                else:
                    pass

        # Not entirely sure why we have to do this
        temp_score_list = list(
            filter(
                lambda x: x[0] >= 0,
                temp_score_list
            )
        )

        for sentence_idx, word_inds, score in temp_score_list:
            self.budget -= len(word_inds)
            if self.budget < 0:
                print("No more budget left!")
                break
            j += len(word_inds)
            self.labelled_idx[sentence_idx] = self.labelled_idx[sentence_idx].union(word_inds)
            for w in word_inds:
                self.unlabelled_idx[sentence_idx].remove(w)

        print(f"Added {j} words to index mapping")

    def update_indices(self, model):
        """
        Score unlabelled instances in terms of their suitability to be labelled next.
        Add the highest scoring instance indices in the dataset to self.labelled_idx
        """
        if self.budget <= 0:
            logging.warning("No more budget left!")

        sentence_scores = {}

        print("\nUpdating indices")
        for batch_index in tqdm(self.unlabelled_batch_indices):

            sentences, tokens, targets, lengths = get_batch(batch_index, self.train_data, self.device)
            word_scores, st_preds = self.word_scoring_func(sentences, tokens, model)
            b = batch_index[0]
            self.st_labels[b] = st_preds
            sentence_scores[b] = [
                float(word_scores[j]) if j in self.unlabelled_idx[b] else None
                for j in range(len(word_scores))
            ]  # scores of unlabelled words --> float, scores of labelled words --> None

        self.extend_indices(sentence_scores)

    def autolabel_sentence(self, model, i):

        # This would not be needed for full sentence labelling!
        sentences, tokens, targets, lengths = get_batch(
            [i], self.autolabelled_data, self.device
        )
        st_preds = self.st_labels[i]

        for j in range(len(st_preds)):
            if j in self.labelled_idx[i]:
                self.autolabelled_data[i][2][j] = self.train_data[i][-1][j]
            else:
                self.autolabelled_data[i][2][j] = st_preds[j]

    def autolabel_dataset(self, model):

        # TODO: We might want to change the threshold number labels needed to include sentence
        # Right now it is just one (i.e. not empty)

        # We keep the same indexing so that we can use the same indices as with train_data
        # We edit the ones that have labels, which appear in partially_labelled_sentence_idx

        partially_labelled_sentence_idx = set()
        print("\nAutolabelling data")
        
        for i in tqdm(range(len(self.autolabelled_data))):
            if not self.labelled_idx[i]:
                continue
            else:
                partially_labelled_sentence_idx = partially_labelled_sentence_idx.union({i})
                self.autolabel_sentence(model, i)

        labelled_subset = Subset(
            self.autolabelled_data, list(partially_labelled_sentence_idx)
        )
        self.labelled_batch_indices = list(
            BatchSampler(
                SubsetRandomSampler(labelled_subset.indices),
                self.batch_size,
                drop_last=self.drop_last,
            )
        )

    def make_unlabelled_dataset(self):

        unlabelled_sentence_idx = [
            j for j in self.unlabelled_idx.keys() if self.unlabelled_idx[j]
        ]
        unlabelled_subset = Subset(self.train_data, unlabelled_sentence_idx)

        self.unlabelled_batch_indices = list(
            BatchSampler(
                SequentialSampler(unlabelled_subset.indices),
                1,
                drop_last=self.drop_last,
            )
        )

    def update_datasets(self, model):
        """
        After ranking the full dataset, use the extended self.labelled_idx to create
        new dataset objects for labelled and unlabelled instances
        """

        self.autolabel_dataset(model)
        self.make_unlabelled_dataset()

    def __iter__(self):
        # DONT FORGET: DO get_batch USING OWN self.autolabelled_data NOT FULL THING
        return (
            self.labelled_batch_indices[i]
            for i in torch.randperm(len(self.labelled_batch_indices))
        )

    def __len__(self):
        return len(self.labelled_batch_indices)


class FullSentenceSelector(object):

    def __init__(self, **kwargs):
        pass

    def score_aggregation(self, scores_list):
        # Just take the per-word normalised score list.
        # This is the average score of each word for the whole sentence
        sentence_score = sum(scores_list)/len(scores_list)
        return sentence_score,

    def score_extraction(self, scores_list):
        """
        Input:
            scores_list: [list, of, scores, from, a, sentence, None, None]
            None REPRESENTS PREVIOUSLY LABELLED WORD - WHICH WILL NOT APPEAR FOR THIS STRATEGY
        Output:
            entries = [([list, of, word, idx], score), ...] for all possible extraction batches
            For this strategy, entries is one element, with all the indices of this sentence
        """
        score = self.score_aggregation(scores_list)
        indices = list(range(len(scores_list)))
        return [(indices, score)]


class WordWindowSelector(object):

    def __init__(self, window_size, **kwargs):
        self.window_size = window_size

    def score_aggregation(self, scores_list):
        # Just take the per-word normalised score list.
        # This is the average score of each word for the whole sentence
        sentence_score = sum(scores_list)/len(scores_list)
        return sentence_score

    def score_extraction(self, scores_list):
        """
        Input:
            scores_list: [list, of, scores, from, a, sentence, None, None]
            None REPRESENTS PREVIOUSLY LABELLED WORD - WHICH WILL NOT APPEAR FOR THIS STRATEGY
        Output:
            entries = [([list, of, word, idx], score), ...] for all possible extraction batches
            For this strategy, entries is one element, with all the indices of this sentence
        """
        indices_and_word_scores = [
            (
                list(range(j, j + self.window_size)),
                scores_list[j:j + self.window_size]
            ) for j in range(len(scores_list) - self.window_size + 1)
        ]
        out_list = []
        for lt in indices_and_word_scores:
            if any(l == None for l in lt[0]):
                continue
            score = self.score_aggregation(lt[1])
            out_list.append((lt[0], score))

        return out_list


class RandomBaselineAcquisition(object):

    def __init__(self, **kwargs):
        pass

    def word_scoring_func(self, sentences, tokens, model):
        # Preds can be random here since we are using full sentences
        scores, preds = tuple([random.random() for _ in range(sentences.shape[-1])] for _ in range(2))

        return scores, preds


class LowestConfidenceAcquisition(object):

    def __init__(self, **kwargs):
        pass

    def word_scoring_func(self, sentences, tokens, model):
        model_output = model(
            sentences, tokens
        ).detach().cpu()  # Log probabilities of shape [batch_size (1), length_of_sentence, num_tags (193)]
        scores, preds = model_output[0].max(dim=1)
        # Take most likely sequence but then minus it - i.e. low probs score highly
        return (
            (-scores.cpu().numpy().reshape(-1)).tolist(),
            preds.cpu().numpy().reshape(-1).tolist()
        )


class MaximumEntropyAcquisition(object):

    def __init__(self, **kwargs):
        pass

    def word_scoring_func(self, sentences, tokens, model):
        model_output = model(
            sentences, tokens
        ).detach().cpu()  # Log probabilities of shape [batch_size (1), length_of_sentence, num_tags (193)]
        _, preds = model_output[0].max(dim=1)
        preds = preds.cpu().numpy().reshape(-1).tolist()
        entropies = torch.sum(
            -model_output*np.exp(model_output)[0], dim = -1).cpu().numpy().reshape(-1)
        ).tolist()

        # Take most likely sequence but then minus it - i.e. low probs score highly
        return (entropies, preds)


class BALDAcquisition(object):

    def __init__(self, **kwargs):
        # We might want to set a range of ps and change them using model.*.p = p[i] during the M runs
        self.M = 100

    def word_scoring_func(self, sentences, tokens, model):
        # Preds can be random here since we are using full sentences
        original_model_output = model(sentences, tokens).detach().cpu()
        _, preds = original_model_output[0].max(dim=1)
        preds = preds.cpu().numpy().reshape(-1).tolist()

        # Do the M forward passes:
        dropout_model_preds = [
            model(sentences, tokens).detach().cpu()[0].max(dim=1)[1].numpy() for _ in range(self.M) # TODO: Fix for batch size
        ] # list of self.M tensors of size (seq_len)
        dropout_model_preds = np.dstack(dropout_model_preds, dim=-1) # Of size (seq_len, self.M)
        majority_vote = np.array([np.argmax(np.bincount(dropout_model_preds[i])) for i in range(self.M)])
        scores = 1 - np.array(
            [sum(dropout_model_preds[j] == majority_vote[j]) for j in range(dropout_model_preds.shape[0])]
        ) / self.M
        return scores, preds

