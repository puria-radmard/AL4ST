import logging
from typing import List, Dict

import numpy as np
import pandas as pd


def make_vocab_txt(files: List[str], col_names: List[str], token_col: str, target_file: str = "vocab.txt"):
    for filename in files:

        df = pd.read_csv(filename, sep='\t', skip_blank_lines=False, names=col_names, error_bad_lines=False)

        # Right now we do counts on both train and test, might want to change this/migth not make difference
        # test file definitely needs fixing
        token_series = df[token_col].dropna().to_numpy()
        tokens, counts = np.unique(token_series, return_counts=True)

        token_counter = dict(zip(tokens, counts))
        token_counter = {k: v for k, v in sorted(token_counter.items(), key=lambda item: item[1], reverse=True)}

        with open(target_file) as vocab_txt:

            # This needs some fixing/purging
            for token, count in token_counter.items():
                vocab_txt.write(f"{token}\t{count}\n")


def construct_data_dictionary(sentence_df: pd.DataFrame, token_col: str, label_col: str):

    sentence_df = sentence_df[1:].reset_index(inplace=False)
    label_list = [
            {
                "start": i,
                "label": label,
                "text": sentence_df[token_col][i]
            } for i, label in enumerate(sentence_df[label_col])
        if label != 'O'
    ]

    data_dict = {
        "sentText": " ".join(sentence_df[token_col]),
        "articleId": None,
        "sentId": "1",
        "relationMentions": [],
        "entityMentions": label_list
    }

    return data_dict


def make_dataset_jsons(file_mappings: Dict[str, str], col_names: List[str], token_col: str, label_col: str):

    for fin, fout in file_mappings.items():

        with open(fout) as j_file:

            df = pd.read_csv(fin, sep='\t', skip_blank_lines=False, names=col_names, error_bad_lines=False)
            sentence_list = np.split(df, df[df.isnull().all(1)].index)

            for sentence_df in sentence_list:

                if not len(sentence_df):
                    continue

                data_dictionary = construct_data_dictionary(sentence_df, token_col, label_col)

                j_file.write(str(data_dictionary))
                j_file.write('\n')


if __name__ == '__main__':

    col_names = ["tokens", "POS", "LING", "NER"]
    token_col = "tokens"
    dataset_json_mappings = {
        'data/OntoNotes-5.0/onto.test.ner': 'data/OntoNotes-5.0/POS/test.json',
        'data/OntoNotes-5.0/onto.train.ner': 'data/OntoNotes-5.0/POS/train.json'
    }
    corpus_files=list(dataset_json_mappings.keys())
    label_col = "POS"
    vocab_txt = 'data/OntoNotes-5.0/POS/vocab.txt'

    logging.info("Started making vocab.txt")
    make_vocab_txt(
        files=corpus_files,
        col_names=col_names,
        token_col=token_col,
        target_file=vocab_txt
    )
    logging.info("Finished making vocab.txt")

    logging.info("Started making dataset jsons")
    make_dataset_jsons(
        file_mappings=dataset_json_mappings,
        col_names=col_names,
        token_col=token_col,
        label_col=label_col,
    )
    logging.info("Finished making dataset jsons")
