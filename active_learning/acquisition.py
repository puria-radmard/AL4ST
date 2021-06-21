import numpy as np
import torch


class AcquisitionAggregation:

    def __init__(self, functions, dataset):
        self.functions = functions
        self.dataset = dataset

    def aggregation_step(self):
        # This is for learning aggregations, e.g. LSA bandit set up
        pass

    def acquisition_aggregation(self, scores):
        pass

    def score(self, i):
        scores = []
        for function in self.functions:
            scores.append(function.score(i))
        return self.acquisition_aggregation(scores).reshape(-1)

    def step(self):
        self.aggregation_step()
        for function in self.functions:
            if isinstance(function, UnitwiseAcquisition):
                pass
            elif isinstance(function, DataAwareAcquisition):
                function.step()
            else:
                raise NotImplementedError(f"{type(function)} not a function type")
        pass
    

class SimpleAggregation(AcquisitionAggregation):
    
    def __init__(self, functions, dataset, weighting):
        weighting_norm = sum(w**2 for w in weighting)**0.5
        self.weighting = np.array([weighting])/weighting_norm
        super(SimpleAggregation, self).__init__(functions, dataset)

    def acquisition_aggregation(self, scores):
        return self.weighting @ scores


class LearningAggregation(AcquisitionAggregation):

    def __init__(self, functions, dataset):
        super(LearningAggregation, self).__init__(functions, dataset)
        raise NotImplementedError

    def acquisition_aggregation(self, scores):
        raise NotImplementedError

    def aggregation_step(self):
        raise NotImplementedError


class DataAwareAcquisition:

    def __init__(self, dataset):
        self.dataset = dataset

    def score(self, i):
        pass

    def step(self):
        pass


class PredsKLAcquisition(DataAwareAcquisition):

    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, i):
        preds = self.dataset.preds[i]
        previous_preds = self.dataset.preds.prev_attr[i]
        log_term = preds - previous_preds
        kl_div = np.sum(preds * log_term, axis=-1)
        return kl_div

    def step(self):
        pass


class EmbeddingMigrationAcquisition(DataAwareAcquisition):

    def __init__(self, dataset, embedding_name):
        super().__init__(dataset=dataset)
        self.embedding_name = embedding_name

    def score(self, i):
        embs = self.dataset.__getattr__(self.embedding_name)[i]
        previous_embs = self.dataset.__getattr__(self.embedding_name).prev_attr[i]
        euc = np.linalg.norm(embs - previous_embs, axis=-1)
        return euc

    def step(self):
        pass


class UnitwiseAcquisition:

    def __init__(self, dataset):
        self.dataset = dataset

    def score(self, i):
        pass


class RandomBaselineAcquisition(UnitwiseAcquisition):

    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, i):
        preds = self.dataset.last_preds[i]
        return np.random.randn(len(preds))


class LowestConfidenceAcquisition(UnitwiseAcquisition):

    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, i):
        # logits (batch_size x sent_length x num_tags [193])
        preds = self.dataset.last_preds[i]
        scores = -preds.max(axis=-1)  # negative highest logits (batch_size x sent_length)
        return scores.reshape(-1)


class MaximumEntropyAcquisition(UnitwiseAcquisition):

    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, i):
        # logits (batch_size x sent_length x num_tags [193])
        preds = self.dataset.last_preds[i]
        scores = np.sum(-preds * np.exp(preds), axis=-1)  # entropies of shape (batch_size x sent_length)
        return scores.reshape(-1) #[scores[i, :length].reshape(-1) for i, length in enumerate(lengths)]


class BALDAcquisition(UnitwiseAcquisition):
    """
        SUSPENDED FOR NOW
    """

    def __init__(self, m):
        # We might want to set a range of ps and change them using model.*.p = p[i] during the M runs
        self.m = 100
        super().__init__()

    def score(self, sentences, tokens, model):
        # Do the M forward passes:
        model.char_encoder.drop.train()
        model.word_encoder.drop.train()
        model.drop.train()

        dropout_model_preds = [model(sentences, tokens).detach().cpu()[0].max(dim=1)[1].numpy() for _ in
                               range(self.m)]  # list of self.M tensors of size (seq_len)
        dropout_model_preds = np.dstack(dropout_model_preds)
        [0]  # Of size (seq_len, self.M)                                # Test scores here
        majority_vote = np.array([np.argmax(np.bincount(dropout_model_preds[:, i])) for i in range(self.M)])
        scores = 1 - np.array(
            [sum(dropout_model_preds[j] == majority_vote[j]) for j in range(dropout_model_preds.shape[0])]
        ) / self.m

        model.eval()

        return scores
