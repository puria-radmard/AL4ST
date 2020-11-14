import torch


class FullSentenceSelector:

    def __init__(self, helper):
        self.helper = helper

    def score_aggregation(self, scores_list):
        # Just take the per-word normalised score list.
        # This is the average score of each word for the whole sentence
        sentence_score = sum(scores_list) / len(scores_list)
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
        score = self.score_aggregation(scores_list)
        indices = list(range(len(scores_list)))
        return [(indices, score)]

    def get_batch(self, batch, agent):
        """
        No model predictions required!
        KL mask is all zeros, since everything in the sentence is labelled, therefore one hot encoded
        """
        return self.helper.get_batch(batch)


class WordWindowSelector(FullSentenceSelector):

    def __init__(self, window_size, **kwargs):
        self.window_size = window_size

    def score_aggregation(self, scores_list):
        # Just take the per-word normalised score list.
        # This is the average score of each word for the whole sentence
        sentence_score = sum(scores_list) / len(scores_list)
        return sentence_score

    def score_extraction(self, scores_list):
        """
        Input:
            scores_list: [list, of, scores, from, a, sentence, None, None]
            None REPRESENTS PREVIOUSLY LABELLED WORD
        Output:
            entries = [([list, of, word, idx], score), ...] for all possible extraction batches
        """
        indices_and_word_scores = [
            (
                list(range(j, j + self.window_size)),
                scores_list[j:j + self.window_size]
            ) for j in range(len(scores_list) - self.window_size + 1)
        ]
        out_list = []
        for lt in indices_and_word_scores:
            if any(l == None for l in lt[1]):
                continue
            score = self.score_aggregation(lt[1])
            out_list.append((lt[0], score))

        return out_list

    def get_batch(self, batch_indices, al_agent, model):
        vocab = None
        charset = None
        tag_set = None
        """
        Same as the original get batch, except targets are now given with a dimension of size num_tags in there.
        If the word is used in training and appears in self.labelled_idx, this is just one hot encoding
        else, it is the probability distribution that the most latest model has predicted
        """

        batch = [al_agent.train_data[idx] for idx in batch_indices]
        sentences, tokens, tags = zip(*batch)

        padded_sentences, lengths = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in sentences], enforce_sorted=False),
            batch_first=True,
            padding_value=vocab["<pad>"],
        )
        padded_tokens, _ = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in tokens], enforce_sorted=False),
            batch_first=True,
            padding_value=charset["<pad>"],
        )
        padded_tags, _ = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in tags], enforce_sorted=False),
            batch_first=True,
            padding_value=tag_set["O"],
        )

        padded_sentences = padded_sentences.to(al_agent.device)
        padded_tokens = padded_tokens.to(al_agent.device)
        model_log_probs = model(padded_sentences, padded_tokens)

        padded_tags = torch.nn.functional.one_hot(padded_tags, num_classes=193).float()  # MAKE NUM CLASSES A PARAMETER?
        kl_mask = torch.zeros(padded_tags.shape[:2]).to(al_agent.device)

        # Fill in the words that have not been queried
        for sentence_idx, sentence_tags in enumerate(padded_tags):
            sentence_index = batch_indices[sentence_idx]
            for word_idx in range(int(lengths[sentence_idx])):
                if word_idx in al_agent.labelled_idx[sentence_index]:  # Labelled
                    pass
                elif word_idx in al_agent.unlabelled_idx[sentence_index]:  # Not labelled
                    padded_tags[sentence_idx, word_idx] = \
                        torch.exp(model_log_probs[sentence_idx, word_idx])
                    kl_mask[sentence_idx, word_idx] = 1
                else:  # Padding
                    continue

        return (
            padded_sentences,
            padded_tokens,
            padded_tags.to(al_agent.device),
            lengths.to(al_agent.device),
            kl_mask
        )
