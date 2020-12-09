import torch
import numpy as np


class Selector:

    def __init__(self, helper, normalisation_index: float):
        self.helper = helper
        self.normalisation_index = normalisation_index

    def score_aggregation(self, word_scores):
        """
        Standard score aggregation where word-wise scores are added or averaged
        """
        score = np.sum(word_scores)
        score *= len(word_scores)**(-self.normalisation_index)
        return score

    def reduce_window_size(self):
        pass

    @staticmethod
    def purify_entries(entries):
        """Sort and remove disjoint entries of form [([list, of, word, idx], score), ...]"""
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

    def __init__(self, helper, normalisation_index):
        super().__init__(helper=helper, normalisation_index=normalisation_index)

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

    def __init__(self, helper, window_size, beta, model):
        super().__init__(helper=helper, normalisation_index=1.0)
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

    def __init__(self, helper, window_range, beta, model):
        super().__init__(helper=helper, normalisation_index=1.0)
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
