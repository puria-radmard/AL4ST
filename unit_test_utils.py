import torch
import random
from torch import nn
from active_learning.agent import *

NUM_SENTENCES_TRAIN = 500
NUM_SENTENCES_TEST = 100
VOCAB_SIZE = 50000
MIN_SENTENCE_LENGTH = 30
MAX_SENTENCE_LENGTH = 70
NUM_TAGS = 193
TOKEN_LENGTH = 20
TOKEN_MAX_VAL = 94

length_list = list(range(MIN_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH + 1))
tags_list = list(range(NUM_TAGS))
vocab_list = list(range(VOCAB_SIZE))
token_val_list = list(range(TOKEN_MAX_VAL))

TRAIN_DATA = []
TEST_DATA = []

for _ in range(NUM_SENTENCES_TRAIN):
    sent_length = random.choice(length_list)
    one = [random.choice(vocab_list) for _ in range(sent_length)]
    two = [
        [random.choice(token_val_list) for _ in range(TOKEN_LENGTH)]
        for _ in range(sent_length)
    ]
    three = [random.choice(tags_list) for _ in range(sent_length)]
    TRAIN_DATA.append([one, two, three])

for _ in range(NUM_SENTENCES_TRAIN):
    sent_length = random.choice(length_list)
    one = [random.choice(vocab_list) for _ in range(sent_length)]
    two = [
        [random.choice(token_val_list) for _ in range(TOKEN_LENGTH)]
        for _ in range(sent_length)
    ]
    three = [random.choice(tags_list) for _ in range(sent_length)]
    TEST_DATA.append([one, two, three])

num_sentences_init = int(len(TRAIN_DATA) * 0.1)
acquisition_class = RandomBaselineAcquisition()
selector = WordWindowSelector(window_size=5)
round_size = 100
batch_size = 32
device = torch.device("cuda")


def model(sentences, tokens):
    """
    Random values from [0, 1], proportioned, and logged.
    Input is number of sentences (padded), each of the maximum sentence size
    Output is of size [num sentences, maximum sequence length, number of tags]
    """
    num_sentences = len(sentences)
    sentence_size = len(sentences[0])

    output = torch.log(
        nn.functional.softmax(
            torch.rand(num_sentences, sentence_size, NUM_TAGS), dim=-1
        )
    )

    return output


def check_consistent_sentence_lengths(agent: ActiveLearningDataset):

    for sentence_idx, labelled_indices in agent.labelled_idx.items():
        assert len(labelled_indices) + len(agent.unlabelled_idx[sentence_idx]) == len(
            TRAIN_DATA[sentence_idx][0]
        )


def search_list_of_lists(ele, lol):

    for l in lol:
        if ele in l:
            return True
    else:
        return False


agent = ActiveLearningDataset(
    train_data=TRAIN_DATA,
    test_data=TEST_DATA,
    num_sentences_init=num_sentences_init,
    acquisition_class=acquisition_class,
    selector_class=selector,
    round_size=round_size,
    batch_size=batch_size,
    model=model,
    device=device,
)

print("TESTING VARIABLES DEFINED")
