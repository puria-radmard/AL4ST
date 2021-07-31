import numpy as np
import torch
from functools import lru_cache
import json
from tqdm import tqdm
from .util_classes import DimensionlessAnnotationUnit

TQDM_MODE = True

class AcquisitionAggregation:

    def __init__(self, functions, dataset):
        self.functions = functions
        self.dataset = dataset

    def aggregation_step(self):
        # This is for learning aggregations, e.g. LSA bandit set up
        raise NotImplementedError

    def acquisition_aggregation(self, scores):
        raise NotImplementedError

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
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class PredsKLAcquisition(DataAwareAcquisition):

    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, i):
        preds = self.dataset.last_preds[i]
        previous_preds = self.dataset.last_preds.prev_attr[i]
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
        raise NotImplementedError


class RandomBaselineAcquisition(UnitwiseAcquisition):

    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, i):
        preds = self.dataset.last_preds[i]
        scores_shape = preds.max(axis=-1)
        return np.random.randn(*scores_shape.shape)


class LowestConfidenceAcquisition(UnitwiseAcquisition):

    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, i):
        preds = self.dataset.last_preds[i]
        scores = -preds.max(axis=-1)
        return scores


class MaximumEntropyAcquisition(UnitwiseAcquisition):

    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, i):
        preds = self.dataset.last_preds[i]
        scores = np.sum(-preds * np.exp(preds), axis=-1)
        return scores


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

    
class BatchAcquisition:
    """No scores are used for this class of acquisition functions - """
    
    def __init__(self, dataset):
        self.dataset = dataset

    def select_next_subset(self, candidate_windows, batchsize):
        raise NotImplementedError
     
    
class TFIDFFeatureFunctionBatchAcquisition(BatchAcquisition):
    '''
        SUBMODULAR SUBSET SELECTION FOR LARGE-SCALE SPEECH TRAINING DATA - Wei et al 2014
            --> set tfidf_feature to trigrams - remember to make them jsonable
        This is a very basic, one step coreset selection. Can offer base for other, sequential variants.
        This class DOES NOT currently support subinstance annotation - only works for unit data/sequences.
    '''
    
    def __init__(self, dataset, tfidf_feature, d_cache_path = 'd_cache.json', index_log_path = 'submodular_log.log'):
        super(TFIDFFeatureFunctionBatchAcquisition, self).__init__(dataset)
        self.existing_mapper = {}
        self.d_cache = {}
        self.current_indices = []
        self.current_mu_mapper = {}
        self.d_cache_path = d_cache_path
        self.index_log_path = index_log_path
        self.tfidf_attribute = self.dataset.__getattr__(tfidf_feature)
    
        with open(d_cache_path, 'r') as jfile:
            self.d_cache = json.load(jfile)
        
        with open(index_log_path, 'r') as f:
            lines = f.read().split('\n')[:-1]
            if len(lines) > 0:
                all_indices = list(dataset.data.attr)
                self.load_previous_set([int(ind) for ind in lines], all_indices)

        
    def d(self, feature, all_windows):
        """NEEDS TO BE TRIGGERED IF DATASET IS EXPANDED, MAINLY FOR ROUNDWISE ACQUISITION"""
        if feature in self.d_cache.keys():
            return self.d_cache[feature]
        else:
            d = 0
            for window in all_windows:
                ith_sequence_features = self.tfidf_attribute.get_attr_by_window(window)
                if feature in ith_sequence_features:
                    d += 1
            self.d_cache[feature] = d
            self.save_d_cache()
            return d

    @staticmethod
    def g(x):
        return x ** 0.5
    
    @staticmethod
    def tf(feature, all_features):
        return all_features.count(feature)
    
    def idf(self, feature, all_windows):
        return np.log(self.V/self.d(feature, all_windows))
    
    def m_u(self, feature, all_features, all_windows):
        return self.tf(feature, all_features) * self.idf(feature, all_windows)
    
    @staticmethod
    def add_dictionaries(d1, d2):
        # THIS CAN BE SPED UP
        d3 = {}
        unadded_k = set(d2.keys())
        for k, v in d1.items():
            if k in d2:
                d3[k] = v + d2[k]
                unadded_k = unadded_k - {k}
            else:
                d3[k] = v
        for k in unadded_k:
            d3[k] = d2[k]
        return d3
            
    def get_mu_dict(self, window, all_windows):
        all_features = self.tfidf_attribute.get_attr_by_window(window)
        unique_features = []
        mu_dict = {}
        for fe in all_features:
            if fe not in unique_features:
                unique_features.append(fe)
        for fe in unique_features:
            mu_dict[fe] = self.m_u(fe, all_features, all_windows)
        return mu_dict
    
    def f_feature(self, current_windows, new_windows, all_windows):
        mu_scores = self.current_mu_mapper.copy()
        for window in new_windows:
            muj = self.get_mu_dict(window, all_windows)
            mu_scores = self.add_dictionaries(muj, mu_scores)
        return sum([self.g(m) for m in mu_scores.values()]), mu_scores
        
    def greedy_increment(self, candidate_windows):
        current_max = -np.inf
        chosen_idx = None
        next_mapper = None
        current_windows = [candidate_windows[j] for j in self.current_indices]
        for i, w in enumerate(candidate_windows):
            if i in self.current_indices:
                continue
            ith_score, candidate_mapper = self.f_feature(current_windows, [w], candidate_windows)
            if ith_score > current_max:
                current_max = ith_score
                chosen_idx = i
                next_mapper = candidate_mapper
        self.current_indices.append(chosen_idx)
        self.current_mu_mapper = next_mapper.copy()
        with open(self.index_log_path, 'a') as f:
            print(chosen_idx, '\t', current_max, '\n', file = f)
        return current_max
    
    def load_previous_set(self, previous_indices, all_windows):
        self.V = len(all_windows)
        self.current_indices = previous_indices
        self.current_mu_mapper = {}
        print('loading previous progress!')
        for i in tqdm(previous_indices, disable = not TQDM_MODE):
            self.current_mu_mapper = self.add_dictionaries(
                self.current_mu_mapper,
                self.get_mu_dict(all_windows[i], all_windows)
            )
        print('loading finished!')
    
    def select_next_subset(self, candidate_windows, batchsize):
        self.V = len(candidate_windows)
        score_history = []
        for ra in tqdm(range(batchsize), disable = not TQDM_MODE):
            if len(self.current_indices) > ra:
                continue
            new_score = self.greedy_increment(candidate_windows)
            score_history.append(new_score)
            if len(self.current_indices) >= batchsize:
                print('DONE')
                break
        return [candidate_windows[i] for i in self.current_indices]
    
    def save_d_cache(self):
        with open(self.d_cache_path, 'w') as jfile:
            json.dump(self.d_cache, jfile)            
            
            

class UncertaintyAugmentedTFIDFFeatureFunctionBatchAcquisition(TFIDFFeatureFunctionBatchAcquisition):

    def __init__(self, dataset, tfidf_feature, score_attribute, d_cache_path = 'd_cache.json', index_log_path = 'submodular_log.log'):
        super(UncertaintyAugmentedTFIDFFeatureFunctionBatchAcquisition, self).__init__(dataset, tfidf_feature, d_cache_path, index_log_path)
        self.score_attribute = self.dataset.__getattr__(score_attribute)
    
    def get_mu_dict(self, window, all_windows):
        all_features = self.tfidf_attribute.get_attr_by_window(window)
        score = self.score_attribute.get_attr_by_window(window)
        score = np.mean(score)
        unique_features = []
        mu_dict = {}
        for fe in all_features:
            if fe not in unique_features:
                unique_features.append(fe)
        for fe in unique_features:
            mu_dict[fe] = self.m_u(fe, all_features, all_windows) * float(score)
        return mu_dict

    
class KMeansCentroidBatchAcquisition(BatchAcquisition):
    """
    This is BADGE: https://arxiv.org/abs/1906.03671 if you set the relevant attribute to be made as:
    
        output = model(torch.tensor(batch).to('cuda'), False)
        hyp_preds = nn.functional.one_hot(output['last_preds'].argmax(axis = -1), 10)
        hyp_loss = (hyp_preds * output['last_preds']).sum()
        hyp_loss.backward()
        model.state_dict(keep_vars=True)['fc.weight'].grad.shape
        
    This would also require batch_size = 1 on the agent! This needs to be dealt with elsewhere
    """

    def __init__(self, dataset, attribute_name, pca_comps = 16):
        super(KMeansCentroidBatchAcquisition, self).__init__(dataset)
        self.attribute_name = attribute_name
        self.pca_comps = pca_comps
        
    def select_next_subset(self, candidate_windows, batchsize):
        mechanism = al.batch_querying.SequentialKMeansBatchQuerying(batchsize, self.attribute_name, pca_comps=self.pca_comps)
        chosen_indices = mechanism.init_round(candidate_windows, self.dataset)
        return [DimensionlessAnnotationUnit(i, ..., None) for i in chosen_indices]
    