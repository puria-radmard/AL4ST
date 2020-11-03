import logging
import random

from tqdm import tqdm

from training_utils import *


def configure_al_agent(args, device, model, train_data, test_data):
    num_sentences_init = int(len(train_data) * args.initprop)
    round_size = int(args.roundsize)

    if args.window != -1:
        selector = WordWindowSelector(window_size=args.window)
    else:
        selector = FullSentenceSelector()

    if args.acquisition == 'rand':
        acquisition_class = RandomBaselineAcquisition()
    elif args.acquisition == 'lc':
        acquisition_class = LowestConfidenceAcquisition()
    elif args.acquisition == 'maxent':
        acquisition_class = MaximumEntropyAcquisition()
    elif args.acquisition == 'bald':
        acquisition_class = BALDAcquisition()
    else:
        raise ValueError(args.acquisition)

    agent = AgentClass(
        train_data=train_data,
        test_data=test_data,
        num_sentences_init=num_sentences_init,
        acquisition_class=acquisition_class,
        selector_class=selector,
        round_size=round_size,
        batch_size=args.batch_size,
        model=model,
        device=device
    )

    return agent


class ActiveLearningDataset:

    def __init__(
            self,
            train_data,
            test_data,
            batch_size,
            round_size,
            num_sentences_init,
            acquisition_class,
            selector_class,
            model,
            device,
            drop_last=False,
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

        # Short term storage of labels used for filling in sentence blanks
        self.st_labels = {j: [] for j in range(len(self.train_data))}

        # Dictionaries mapping {sentence idx: [list, of, word, idx]} for labelled and unlabelled words
        self.labelled_idx = {j: set() for j in range(len(self.train_data))}
        self.unlabelled_idx = {
            j: set(range(len(train_data[j][0]))) for j in range(len(self.train_data))
        }

        self.acquisition = acquisition_class
        self.selector = selector_class

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
        start_entries = sorted(entries, key=lambda x: x[-1])
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

        print("\nExtending indices")
        for sentence_idx, scores_list in tqdm(sentence_scores.items()):
            # Skip if entirely Nones
            if all([type(i) == type(None) for i in scores_list]):
                continue
            entries = self.selector.score_extraction(scores_list)
            entries = self.purify_entries(entries)
            # entries = [([list, of, word, idx], score), ...] that can be compared to temp_score_list
            for entry in entries:
                if entry[-1] > temp_score_list[0][-1]:
                    temp_score_list[0] = (sentence_idx, entry[0], entry[1])
                    temp_score_list.sort(key=lambda y: y[-1])
                else:
                    pass

        j = 0
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
            # Use normal get_batch here since we don't want to fill anything in, but it doesn't really matter for functionality
            sentences, tokens, targets, lengths, kl_mask = get_batch(batch_index, self.train_data, self.device)
            word_scores = self.acquisition.word_scoring_func(sentences, tokens, model)
            b = batch_index[0]
            sentence_scores[b] = [
                float(word_scores[j]) if j in self.unlabelled_idx[b] else None
                for j in range(len(word_scores))
            ]  # scores of unlabelled words --> float, scores of labelled words --> None

        self.extend_indices(sentence_scores)

    def make_labelled_dataset(self, model):

        # We keep the same indexing so that we can use the same indices as with train_data
        # We edit the ones that have labels, which appear in partially_labelled_sentence_idx

        partially_labelled_sentence_idx = set()
        print("\nCreating partially labelled dataset")

        # TODO: We might want to change the threshold number labels needed to include sentence
        # Right now it is just one (i.e. not empty)
        for i in tqdm(range(len(self.train_data))):
            if not self.labelled_idx[i]:
                continue
            else:
                partially_labelled_sentence_idx = partially_labelled_sentence_idx.union({i})

        labelled_subset = Subset(
            self.train_data, list(partially_labelled_sentence_idx)
        )
        self.labelled_batch_indices = list(
            BatchSampler(
                SubsetRandomSampler(labelled_subset.indices),
                self.batch_size,
                drop_last=self.drop_last,
            )
        )

    def make_unlabelled_dataset(self):

        self.unlabelled_batch_indices = unlabelled_sentence_idx = [
            [j] for j in self.unlabelled_idx.keys() if self.unlabelled_idx[j]
        ]

    def update_datasets(self, model):
        """
        After ranking the full dataset, use the extended self.labelled_idx to create
        new dataset objects for labelled and unlabelled instances
        """

        self.make_labelled_dataset(model)
        self.make_unlabelled_dataset()

    def __iter__(self):
        # DONT FORGET: DO self.selector.get_batch on self.train_data
        return (
            self.labelled_batch_indices[i]
            for i in torch.randperm(len(self.labelled_batch_indices))
        )

    def __len__(self):
        return len(self.labelled_batch_indices)


class AgentClass(ActiveLearningDataset):

    def __init__(
            self,
            train_data,
            test_data,
            num_sentences_init,
            acquisition_class,
            selector_class,
            round_size,
            batch_size,
            model,
            device,
    ):
        ActiveLearningDataset.__init__(
            self,
            train_data=train_data,
            test_data=test_data,
            num_sentences_init=num_sentences_init,
            acquisition_class=acquisition_class,
            selector_class=selector_class,
            round_size=round_size,
            batch_size=batch_size,
            model=model,
            device=device
        )


class FullSentenceSelector(object):

    def __init__(self, **kwargs):
        super().__init__()

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

    def get_batch(self, batch_indices, al_agent, model):
        """
        No model predictions required!
        KL mask is all zeros, since everything in the sentence is labelled, therefore one hot encoded
        """

        padded_sentences, padded_tokens, padded_tags, lengths, kl_mask = get_batch(batch_indices, al_agent.train_data,
                                                                                   al_agent.device)
        kl_mask = torch.zeros(padded_tags.shape[:2]).to(al_agent.device)

        padded_sentences = padded_sentences.to(al_agent.device)
        padded_tokens = padded_tokens.to(al_agent.device)
        model_log_probs = model(padded_sentences, padded_tokens)

        return (
            padded_sentences,
            padded_tokens,
            padded_tags,
            lengths,
            kl_mask
        )


class WordWindowSelector(object):

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
        """
        Same as the original get batch, except targets are now given with a dimension of size num_tags in there.
        If the word is used in training and appears in self.labelled_idx, this is just one hot encoding
        else, it is the probability distribution that the most latest model has predicted
        """

        batch = [al_agent.train_data[idx] for idx in batch_indices]
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sentences, tokens, tags = zip(*sorted_batch)

        padded_sentences, lengths = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in sentences]),
            batch_first=True,
            padding_value=vocab["<pad>"],
        )
        padded_tokens, _ = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in tokens]),
            batch_first=True,
            padding_value=charset["<pad>"],
        )
        padded_tags, _ = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in tags]),
            batch_first=True,
            padding_value=tag_set["O"],
        )

        padded_sentences = padded_sentences.to(al_agent.device)
        padded_tokens = padded_tokens.to(al_agent.device)
        model_log_probs = model(padded_sentences, padded_tokens)

        padded_tags = nn.functional.one_hot(padded_tags, num_classes=193).float()  # MAKE NUM CLASSES A PARAMETER?
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


class RandomBaselineAcquisition(object):

    def __init__(self, **kwargs):
        super().__init__()

    def word_scoring_func(self, sentences, tokens, model):
        # Preds can be random here since we are using full sentences
        scores, _ = tuple([random.random() for _ in range(sentences.shape[-1])] for _ in range(2))

        return scores


class LowestConfidenceAcquisition(object):

    def __init__(self, **kwargs):
        super().__init__()

    def word_scoring_func(self, sentences, tokens, model):
        model_output = model(
            sentences, tokens
        ).detach().cpu()  # Log probabilities of shape [batch_size (1), length_of_sentence, num_tags (193)]
        scores = (-model_output[0].max(dim=1)[0].cpu().numpy().reshape(-1)).tolist()
        # Take most likely sequence but then minus it - i.e. low probs score highly
        return scores


class MaximumEntropyAcquisition(object):

    def __init__(self, **kwargs):
        super().__init__()

    def word_scoring_func(self, sentences, tokens, model):
        model_output = model(
            sentences, tokens
        ).detach().cpu()  # Log probabilities of shape [batch_size (1), length_of_sentence, num_tags (193)]
        entropies = torch.sum(-model_output * np.exp(model_output)[0], dim=-1).cpu().numpy().reshape(-1).tolist()
        return entropies


class BALDAcquisition(object):

    def __init__(self, **kwargs):
        # We might want to set a range of ps and change them using model.*.p = p[i] during the M runs
        self.M = 100
        super().__init__()

    def word_scoring_func(self, sentences, tokens, model):
        # Do the M forward passes:
        model.char_encoder.drop.train()
        model.word_encoder.drop.train()
        model.drop.train()

        dropout_model_preds = [model(sentences, tokens).detach().cpu()[0].max(dim=1)[1].numpy() for _ in
                               range(self.M)]  # list of self.M tensors of size (seq_len)
        dropout_model_preds = np.dstack(dropout_model_preds)[0]  # Of size (seq_len, self.M)
        majority_vote = np.array([np.argmax(np.bincount(dropout_model_preds[:, i])) for i in range(self.M)])
        scores = 1 - np.array(
            [sum(dropout_model_preds[j] == majority_vote[j]) for j in range(dropout_model_preds.shape[0])]
        ) / self.M

        model.eval()

        return scores
