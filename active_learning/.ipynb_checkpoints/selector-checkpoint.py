import os
import json
import logging

import torch
import numpy as np
from .util_classes import BeamSearchSolution, total_sum


class Selector:

    def __init__(self, normalisation_index: float, round_size, beam_search_parameter, acquisition, window_class):
        self.normalisation_index = normalisation_index
        self.round_size = round_size
        self.round_selection = []
        self.all_round_windows = []
        self.beam_search_parameter = beam_search_parameter
        self.acquisition = acquisition
        self.labelled_ngrams = {}
        self.window_class = window_class

    def window_generation(self, i, dataset):    # Could work better as a decorator?
        unit_scores = self.acquisition.score(i)
        unit_scores = dataset.index.make_nan_if_labelled(i, unit_scores)
        window_args = self.score_extraction(unit_scores)
        return [self.window_class(i, window["bounds"], window["score"]) for window in window_args]

    def assign_agent(self, agent):
        self.agent = agent

    def score_extraction(self, unit_scores):
        return [{}]

    def score_aggregation(self, word_scores):
        """
        Standard score aggregation where word-wise scores are added or averaged
        """
        score = np.sum(word_scores)
        score *= len(word_scores)**(-self.normalisation_index)
        return score

    def select_best(self, window_scores, allow_propagation):
        # window_scores = [(i, [r1, r2], score), ...]
        logging.info("beginning beam search: ")
        print("0 words branched to")
        self.all_round_windows = window_scores

        # Initialise with best B scores
        b_solutions = [BeamSearchSolution([], self.round_size, self.beam_search_parameter,
                                          labelled_ngrams=self.labelled_ngrams)
                       for _ in range(self.beam_search_parameter)]
        b_solutions = [sol.add_window(window_scores[j], self.agent.train_set) for j, sol in enumerate(b_solutions)]

        while all([not b.lock for b in b_solutions]):
            temporary_solutions = [] # -> self.beam_search_parameter**2
            for solution in b_solutions:
                local_branch = solution.branch_out(temporary_solutions, window_scores, train_set=self.agent.train_set,
                                                   allow_propagation=allow_propagation)
                temporary_solutions.extend(local_branch)
            temporary_solutions.sort(key=lambda x: x.score, reverse=True)
            b_solutions = temporary_solutions[:self.beam_search_parameter]
            print(f"at least {min([b.size for b in b_solutions])}/{self.round_size} words branched to", end="\r")

        best_solution = max(b_solutions, key=lambda x: x.score)
        best_windows = best_solution.windows
        labelled_ngrams = best_solution.labelled_ngrams
        budget_spent = best_solution.size

        self.labelled_ngrams.update(labelled_ngrams)
        self.round_selection = best_windows.copy()
        return best_windows, labelled_ngrams, budget_spent

    def reduce_window_size(self):
        pass

    def save(self, save_path):
        # savable_lookup = [{"tokens": k, "labels": v} for k, v in self.labelled_ngrams.items()]
        with open(os.path.join(save_path, "round_selection.pk"), "w") as f:
            json.dump(
                {
                    "all_round_windows": [w.savable() for w in self.all_round_windows],
                    "round_selection_windows": [w.savable() for w in self.round_selection],
                    # "cumulative_labelled_ngrams": savable_lookup
                }, f
            )

    @staticmethod
    def purify_entries(entries):
        """
        Sort and remove disjoint entries of form [([list, of, word, idx], score), ...]
        """
        start_entries = sorted(entries, key=lambda x: x[-1], reverse=True)
        final_entries = []
        highest_idx = set()
        for entry in start_entries:
            span = set(range(*entry[0]))
            if highest_idx.intersection(span):
                pass
            else:
                highest_idx = highest_idx.union(span)
                final_entries.append(entry)
        return final_entries

    def windows_selection(self, indices_and_word_scores):
        out_list = []
        for idx, scores in indices_and_word_scores:
            score = self.score_aggregation(scores)
            if not np.isnan(score):  # i.e. does not overlap with already labelled words
                out_list.append({"bounds": idx, "score": score})

        return out_list


class DimensionlessSelector(Selector):

    def __init__(self, round_size, acquisition, window_class):
        super(DimensionlessSelector, self).__init__(
            normalisation_index=1, round_size=round_size, beam_search_parameter=1, acquisition=acquisition, window_class=window_class
        )

    def score_aggregation(self, score):
        return float(score)

    def score_extraction(self, score):
        score = self.score_aggregation(score)
        return [{"bounds": ..., "score": score}]


class SentenceSelector(Selector):

    def __init__(self, normalisation_index, round_size, acquisition, window_class):
        super(SentenceSelector, self).__init__(normalisation_index=normalisation_index, round_size=round_size,
                         beam_search_parameter=1, acquisition=acquisition, window_class=window_class)

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
        return [{"bounds": (0, len(scores_list)), "score": score}]


class FixedWindowSelector(Selector):

    def __init__(self, window_size, beta, round_size, beam_search_parameter, acquisition, window_class):
        super(FixedWindowSelector, self).__init__(normalisation_index=1.0, round_size=round_size,
                         beam_search_parameter=beam_search_parameter, acquisition=acquisition,
                         window_class=window_class)
        self.window_size = window_size
        self.beta = beta

    def reduce_window_size(self):
        self.window_size -= 1
        if self.window_size <= 0:
            self.window_size = 1

    def score_extraction(self, scores_list):
        """
        Input:
            scores_list: [list, of, scores, from, a, sentence, nan, nan]
            None REPRESENTS PREVIOUSLY LABELLED WORD
        Output:
            entries = [([list, of, word, idx], score), ...] for all possible extraction batches
        """
        indices_and_word_scores = [
            (
                [j, j + self.window_size],
                scores_list[j:j + self.window_size]
            ) for j in range(len(scores_list) - self.window_size + 1)
        ]

        outlist = self.windows_selection(indices_and_word_scores)
        return outlist


class VariableWindowSelector(Selector):

    def __init__(self, window_range, beta, round_size, beam_search_parameter, normalisation_index, acquisition, window_class):
        super(VariableWindowSelector, self).__init__(normalisation_index=normalisation_index, round_size=round_size,
                         beam_search_parameter=beam_search_parameter, acquisition=acquisition, window_class=window_class)
        self.window_range = window_range
        self.beta = beta

    def reduce_window_size(self):
        self.window_range[0] -= 1
        if self.window_range[0] <= 0:
            self.window_range[0] = 1
        self.window_range[1] -= 1
        if self.window_range[1] <= 0:
            self.window_range[1] = 1

    def score_extraction(self, scores_list):
        """
        Input:
            scores_list: [list, of, scores, from, a, sentence, nan, nan]
            None REPRESENTS PREVIOUSLY LABELLED WORD
        Output:
            entries = [([list, of, word, idx], score), ...] for all possible extraction batches
        """
        indices_and_word_scores = []
        for w in range(*self.window_range):
            indices_and_word_scores.extend(
                [([j, j + w], scores_list[j:j + w]) for j in range(len(scores_list) - w + 1)]
            )

        outlist = self.windows_selection(indices_and_word_scores)
        return outlist
