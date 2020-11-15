import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence


class Helper:

    def __init__(self, vocab, tag_set, charset):
        self.vocab = vocab
        self.tag_set = tag_set
        self.charset = charset

    def get_batch(self, batch):
        sentences, tokens, tags = zip(*batch)

        padded_sentences, lengths = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in sentences], enforce_sorted=False),
            batch_first=True,
            padding_value=self.vocab["<pad>"],
        )
        padded_tokens, _ = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in tokens], enforce_sorted=False),
            batch_first=True,
            padding_value=self.charset["<pad>"],
        )
        padded_tags, _ = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in tags], enforce_sorted=False),
            batch_first=True,
            padding_value=self.tag_set["O"],
        )

        padded_tags = nn.functional.one_hot(padded_tags, num_classes=193).float()  # MAKE NUM CLASSES A PARAMETER?

        return padded_sentences, padded_tokens, padded_tags, lengths
