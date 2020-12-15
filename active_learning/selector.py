import os
import json
import logging

import torch
import numpy as np


class BeamSearchSolution:
    def __init__(self, windows, max_size, B, init_size=None, init_score=None):
        self.windows = windows
        if not init_score:
            self.score = sum([w[-1] for w in windows])
        else:
            self.score = init_score
        if not init_size:
            self.size = sum([w[1][1] - w[1][0] for w in windows])
        else:
            self.size = init_size
        self.max_size = max_size
        self.lock = False
        self.B = B

    def add_window(self, new_window):
        if self.size >= self.max_size:
            self.lock = True
            return self
        init_size = self.size + new_window[1][1] - new_window[1][0]
        init_score = self.score + new_window[-1]
        return BeamSearchSolution(self.windows + [new_window], self.max_size, self.B, init_size=init_size,
                                  init_score=init_score)

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

    @staticmethod
    def windows_overlap(window1, window2):
        if window1[0] != window2[0]:
            return False
        else:
            window1_words = set(range(window1[1][0], window1[1][1]))
            window2_words = set(range(window2[1][0], window2[1][1]))
            if window1_words.intersection(window2_words):
                return True
        return False

    def new_window_viable(self, new_window):
        for window in self.windows:
            if self.windows_overlap(window, new_window):
                return False
        else:
            return True

    def branch_out(self, other_solutions, window_scores):
        # ASSUME window_scores ALREADY SORTED
        local_branch = []
        for window in window_scores:
            if self.new_window_viable(window):
                possible_node = self.add_window(window)
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


class Selector:

    def __init__(self, helper, normalisation_index: float, round_size, beam_search_parameter):
        self.helper = helper
        self.normalisation_index = normalisation_index
        self.round_size = round_size
        self.round_selection = []
        self.all_round_windows = []
        self.beam_search_parameter = beam_search_parameter

    def score_aggregation(self, word_scores):
        """
        Standard score aggregation where word-wise scores are added or averaged
        """
        score = np.sum(word_scores)
        score *= len(word_scores)**(-self.normalisation_index)
        return score

    def select_best(self, window_scores):
        # window_scores = [(i, [r1, r2], score), ...]
        logging.info("beginning beam search: ")
        print("0 words branched to")
        self.all_round_windows = window_scores

        # Initialise with best B scores
        B_solutions = [
            BeamSearchSolution([w], self.round_size, self.beam_search_parameter) for w in
            window_scores[:self.beam_search_parameter]
        ]

        while all([not b.lock for b in B_solutions]):
            temporary_solutions = []
            for solution in B_solutions:
                local_branch = solution.branch_out(temporary_solutions, window_scores)
                temporary_solutions.extend(local_branch)
            temporary_solutions.sort(key=lambda x: x.score, reverse=True)
            B_solutions = temporary_solutions[:self.beam_search_parameter]
            print(f"at least {min([b.size for b in B_solutions])}/{self.round_size} words branched to", end="\r")
        best_solution = max(B_solutions, key=lambda x: x.score)
        best_windows = best_solution.windows

        self.round_selection = best_windows
        return best_windows

    def reduce_window_size(self):
        pass

    def save(self, save_path):
        with open(os.path.join(save_path, "round_selection.pk"), "w") as f:
            json.dump(
                {
                    "all_round_windows": self.all_round_windows,
                    "round_selection_windows": self.round_selection
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
        for lt in indices_and_word_scores:
            score = self.score_aggregation(lt[1])
            if not np.isnan(score):  # i.e. does not overlap with already labelled words
                out_list.append((lt[0], score))

        # Not used when we have beam search!
        if self.beam_search_parameter == 1:

            out_list = self.purify_entries(out_list)
        return out_list

    def get_batch(self, batch, batch_indices, agent):
        """
        Same as the original get batch, except targets are now given with a dimension of size num_tags in there.
        If the word is used in training and appears in self.labelled_idx, this is just one hot encoding
        else, it is the probability distribution that the most latest model has predicted
        """

        padded_sentences, padded_tokens, padded_tags, lengths = \
            [a.to(agent.device) for a in self.helper.get_batch(batch)]
        self.model.eval()
        model_log_probs = self.model(padded_sentences, padded_tokens).detach().to(agent.device)
        self.model.train()
        self_supervision_mask = torch.ones(padded_tags.shape)

        # Fill in the words that have not been queried
        for sentence_idx, sentence_tags in enumerate(padded_tags):
            sentence_index = batch_indices[sentence_idx]
            for word_idx in range(int(lengths[sentence_idx])):
                if word_idx in agent.index.labelled_idx[sentence_index]:  # Labelled
                    pass
                elif word_idx in agent.index.unlabelled_idx[sentence_index]:  # Not labelled
                    padded_tags[sentence_idx, word_idx] = \
                        torch.exp(model_log_probs[sentence_idx, word_idx])
                    self_supervision_mask[sentence_idx, word_idx] = self.beta
                else:  # Padding
                    continue

        return (
            padded_sentences,
            padded_tokens,
            padded_tags,
            lengths,
            self_supervision_mask
        )


class SentenceSelector(Selector):

    def __init__(self, helper, normalisation_index, round_size):
        super().__init__(helper=helper, normalisation_index=normalisation_index, round_size=round_size,
                         beam_search_parameter=1)

    def score_extraction(self, word_scores):
        """
        Input:
            scores_list: [list, of, scores, from, a, sentence, None, None]
            None REPRESENTS PREVIOUSLY LABELLED WORD - WHICH WILL NOT APPEAR FOR THIS STRATEGY
        Output:
            entries = [([list, of, word, idx], score), ...] for all possible extraction batches
            For this strategy, entries is one element, with all the indices of this sentence
        """
        score = self.score_aggregation(word_scores)
        return [((0, len(word_scores)), score)]

    def get_batch(self, batch, **args):
        """
        No model predictions required!
        self_supervision mask is all zeros, since everything in the sentence is labelled
        """
        padded_sentences, padded_tokens, padded_tags, lengths = self.helper.get_batch(batch)
        self_supervision_mask = torch.ones(padded_tags.shape)
        return padded_sentences, padded_tokens, padded_tags, lengths, self_supervision_mask


class FixedWindowSelector(Selector):

    def __init__(self, helper, window_size, beta, model, round_size, beam_search_parameter):
        super().__init__(helper=helper, normalisation_index=1.0, round_size=round_size,
                         beam_search_parameter=beam_search_parameter)
        self.window_size = window_size
        self.model = model
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

    def __init__(self, helper, window_range, beta, model, round_size, beam_search_parameter):
        super().__init__(helper=helper, normalisation_index=1.0, round_size=round_size,
                         beam_search_parameter=beam_search_parameter)
        self.window_range = window_range
        self.model = model
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
