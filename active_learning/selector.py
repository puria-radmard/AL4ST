import os
import json
import logging
import torch
from .beam_search import GreedyBeamSearchSolution, EpsilonGreedyBeamSearchSolution, StochasticBeamSearchSolution
from .batch_querying import *

TQDM_MODE = True


class Selector:
    def __init__(
        self,
        normalisation_index: float,
        round_size,
        beam_search_parameter,
        acquisition,
        window_class,
        diversity_policy,
        selection_mode,
        epsilon=0.2
    ):
        self.normalisation_index = normalisation_index
        self.round_size = round_size
        self.round_selection = []
        self.all_round_windows = []
        self.beam_search_parameter = beam_search_parameter
        self.acquisition = acquisition
        self.window_class = window_class
        self.diversity_policy = diversity_policy

        self.epsilon = epsilon
        if selection_mode == 'argmax':
            self.beam_search_class = GreedyBeamSearchSolution
        elif selection_mode == 'epsilon_greedy':
            self.beam_search_class = EpsilonGreedyBeamSearchSolution
        elif selection_mode == 'softmax':
            self.beam_search_class = StochasticBeamSearchSolution
        else:
            raise ValueError(f'selection_mode "{selection_mode}" not valid')

    def window_generation(self, i, dataset):  # Could work better as a decorator?
        unit_scores = self.acquisition.score(i)
        # TODO: TEST WITH SENTENCES AND WRITE ANY PROCESSING NEEDED HERE
        unit_scores = dataset.index.make_nan_if_labelled(i, unit_scores)
        window_args = self.score_extraction(unit_scores)
        return [
            self.window_class(i, window["bounds"], window["score"])
            for window in window_args
        ]

    def assign_agent(self, agent):
        self.agent = agent

    def score_extraction(self, unit_scores):
        return [{}]

    def score_aggregation(self, word_scores):
        """
        Standard score aggregation where word-wise scores are added or averaged
        """
        score = torch.sum(word_scores)
        score *= len(word_scores) ** (-self.normalisation_index)
        return score

    def initialise_solution(self):
        return self.beam_search_class(
            [],
            self.round_size,
            self.beam_search_parameter,
            self.diversity_policy,
            None,
            None,
            {},
            self.epsilon
        )

    def initialise_solutions(self, unit_scores):
        # Initialise with best B scores
        b_solutions = [self.initialise_solution() for _ in range(self.beam_search_parameter)]
        b_solutions = [
            sol.add_window(unit_scores[j], self.agent.train_set)
            for j, sol in enumerate(b_solutions)
        ]
        return b_solutions

    def extend_solutions(self, b_solutions, unit_scores, usable_mask):
        temporary_solutions = []  # -> self.beam_search_parameter**2
        for solution in b_solutions:
            local_branch, usable_mask = solution.branch_out(
                temporary_solutions, unit_scores, usable_mask, train_set=self.agent.train_set
            )
            temporary_solutions.extend(local_branch)
        temporary_solutions.sort(key=lambda x: x.score, reverse=True)
        b_solutions = temporary_solutions[: self.beam_search_parameter]
        return b_solutions, usable_mask

    def select_best(self, window_scores):
        # window_scores = [(i, [r1, r2], score), ...]
        if TQDM_MODE:
            logging.info("beginning beam search: ")
            print("initialising diversity policy...")
        self.all_round_windows = window_scores
        self.diversity_policy.init_round(window_scores, self.agent.train_set)
        if TQDM_MODE:
            print("0 words branched to")

        # i.e. the B solutions are completely disjoint! This might have to be changed later if beam search is
        # actually being used
        usable_mask = torch.tensor([1 for _ in window_scores])
        b_solutions = self.initialise_solutions(window_scores)
        usable_mask[:self.beam_search_parameter] = 0
        while all([not b.lock for b in b_solutions]):
            b_solutions = self.extend_solutions(b_solutions, window_scores, usable_mask)
            if TQDM_MODE:
                print(
                    f"at least {min([b.size for b in b_solutions])}/{self.round_size} atoms branched to",
                    end="\r",
                )

        best_solution = max(b_solutions, key=lambda x: x.score)
        best_windows = best_solution.windows
        budget_spent = best_solution.size

        self.round_selection = best_windows.copy()
        return best_windows, budget_spent

    def reduce_window_size(self):
        pass

    def save(self, save_path):
        # savable_lookup = [{"tokens": k, "labels": v} for k, v in self.labelled_ngrams.items()]
        with open(os.path.join(save_path, "round_selection.pk"), "w") as f:
            json.dump(
                {
                    "all_round_windows": [w.savable() for w in self.all_round_windows],
                    "round_selection_windows": [
                        w.savable() for w in self.round_selection
                    ],
                    # "cumulative_labelled_ngrams": savable_lookup
                },
                f,
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
            if not torch.isnan(score):  # i.e. does not overlap with already labelled words
                out_list.append({"bounds": idx, "score": score})

        return out_list


class DimensionlessSelector(Selector):
    def __init__(self, round_size, acquisition, window_class, diversity_policy):
        super(DimensionlessSelector, self).__init__(
            normalisation_index=1,
            round_size=round_size,
            beam_search_parameter=1,
            acquisition=acquisition,
            window_class=window_class,
            diversity_policy=diversity_policy,
        )

    def score_aggregation(self, score):
        return float(score)

    def score_extraction(self, score):
        score = self.score_aggregation(score)
        return [{"bounds": ..., "score": score}]


class SentenceSelector(Selector):
    def __init__(
        self,
        normalisation_index,
        round_size,
        acquisition,
        window_class,
        diversity_policy,
    ):
        super(SentenceSelector, self).__init__(
            normalisation_index=normalisation_index,
            round_size=round_size,
            beam_search_parameter=1,
            acquisition=acquisition,
            window_class=window_class,
            diversity_policy=diversity_policy,
        )

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
        # This can be merged with dimensionless selector no?
        return [{"bounds": (0, len(scores_list)), "score": score}]


class FixedWindowSelector(Selector):
    def __init__(
        self,
        window_size,
        beta,
        round_size,
        beam_search_parameter,
        acquisition,
        window_class,
        diversity_policy,
    ):
        super(FixedWindowSelector, self).__init__(
            normalisation_index=1.0,
            round_size=round_size,
            beam_search_parameter=beam_search_parameter,
            acquisition=acquisition,
            window_class=window_class,
            diversity_polcy=diversity_policy,
        )
        self.window_size = window_size
        self.beta = beta

    def reduce_window_size(self):
        self.window_size -= 1
        if self.window_size <= 0:
            self.window_size = 1

    def score_extraction(self, scores_list):
        indices_and_word_scores = [([j, j + self.window_size], scores_list[j : j + self.window_size])
                                   for j in range(len(scores_list) - self.window_size + 1)]

        outlist = self.windows_selection(indices_and_word_scores)
        return outlist


class VariableWindowSelector(Selector):
    def __init__(
        self,
        window_range,
        beta,
        round_size,
        beam_search_parameter,
        normalisation_index,
        acquisition,
        window_class,
        diversity_policy
    ):
        super(VariableWindowSelector, self).__init__(
            normalisation_index=normalisation_index,
            round_size=round_size,
            beam_search_parameter=beam_search_parameter,
            acquisition=acquisition,
            window_class=window_class,
            diversity_policy=diversity_policy
        )
        self.window_range = window_range
        self.beta = beta

    def reduce_window_size(self):
        self.window_range[0] = min([1, self.window_range[0] - 1])
        self.window_range[1] = min([2, self.window_range[1] - 1])

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
            indices_and_word_scores.extend([([j, j+w], scores_list[j:j+w]) for j in range(len(scores_list)-w+1)])
        outlist = self.windows_selection(indices_and_word_scores)
        return outlist


class SubsetSelector:
    def __init__(self, round_size, acquisition, window_class, *args):
        self.round_size = round_size
        self.round_selection = []
        self.all_round_windows = []
        self.acquisition = acquisition  # ADD CHECK FOR THIS
        self.window_class = window_class

    def window_generation(self, i, dataset):  # Could work better as a decorator?
        # TODO: TEST WITH SENTENCES AND WRITE ANY PROCESSING NEEDED HERE
        return [self.window_class(i, ..., None)]

    def assign_agent(self, agent):
        self.agent = agent

    def select_best(self, window_scores):
        next_windows = self.acquisition.select_next_subset(window_scores, self.round_size)
        return next_windows, len(next_windows)
