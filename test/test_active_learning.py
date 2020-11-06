from unit_test_utils import *
from active_learning import *
import pytest


@pytest.mark.parametrize("agent", [(agent)])
def test_purify_elements(agent: ActiveLearningDataset):

    entries = [
        ([1, 2, 3, 4, 5], 5),
        ([2, 3, 4, 5, 6], 4),
        ([6, 7, 8, 9, 10], 3),
        ([10, 11, 12, 13, 14], 2),
        ([15, 16, 17, 18, 19], 1),
    ]

    target_entries = [
        ([15, 16, 17, 18, 19], 1),
        ([6, 7, 8, 9, 10], 3),
        ([1, 2, 3, 4, 5], 5),
    ]

    assert sorted(agent.purify_entries(entries)) == sorted(target_entries)


@pytest.mark.parametrize("agent", [(agent)])
def test_update_indices(agent: ActiveLearningDataset):

    sentence_scores = agent.get_all_scores(model)

    # Make sure entries going into purification have Nones in the right place
    for sentence_idx, sentence_score in sentence_scores.items():
        for i, word_score in enumerate(sentence_score):
            if i in agent.labelled_idx[sentence_idx]:
                assert word_score == None
            elif i in agent.unlabelled_idx[sentence_idx]:
                assert type(word_score) == float
            else:
                raise IndexError("sentence_scores from agent.update_indices has sentence too long")
            # Check if idxs are too high?

    temp_score_list = agent.extend_indices(sentence_scores)

    # For each sentence...
    for sentence_idx in agent.labelled_idx.keys():

        sentence_temp_score_list = [a for a in temp_score_list if a[0] == sentence_idx]
        all_word_indices = agent.labelled_idx[sentence_idx].union(agent.unlabelled_idx[sentence_idx])
        for _, word_indices, score in sentence_temp_score_list:

            # Make sure index list entries in temp_score_list are within bounds
            assert set(word_indices).issubset(all_word_indices)

            # Make sure there is no overlap (there can be unused words here)
            for w in word_indices:
                all_word_indices.remove(w)

    # Make sure sentence indices are still consistent
    check_consistent_sentence_lengths(agent=agent)


@pytest.mark.parametrize("agent", [(agent)])
def test_update_datasets(agent: ActiveLearningDataset):

    agent.update_datasets(model)

    for sentence_idx, labelled_idx in agent.labelled_idx.items():

        # Will need to change this for thresholding
        if labelled_idx:
            assert search_list_of_lists(sentence_idx, agent.labelled_batch_indices)
        else:
            assert search_list_of_lists(sentence_idx, agent.unlabelled_batch_indices)