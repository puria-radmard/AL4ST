import os
import json
import sys
from gensim.models.word2vec import LineSentence, Word2Vec


def func(fin, fout):
    for line in fin:
        line = line.strip()
        if not line:
            continue
        sentence = json.loads(line)
        sentence = sentence["sentText"].strip().strip('"').lower()
        fout.write(sentence + "\n")


def make_corpus():
    # print("-------------haha")
    with open(os.path.join(root_dir, "corpus.txt"), "wt", encoding="utf-8") as fout:
        with open(os.path.join(root_dir, "train.json"), "rt", encoding="utf-8") as fin:
            func(fin, fout)
        with open(os.path.join(root_dir, "test.json"), "rt", encoding="utf-8") as fin:
            func(fin, fout)


if __name__ == "__main__":

    root_dir = sys.argv[1]  # Fix this later
    # e.g. "data/NYT_CoType"

    if not os.path.exists(os.path.join(root_dir, "corpus.txt")):
        make_corpus()
    print("Made corpus")

    sentences = LineSentence(os.path.join(root_dir, "corpus.txt"))
    print("Made sentences")

    model = Word2Vec(sentences, sg=1, size=300, workers=4, iter=8, negative=8)
    print("Made model")

    word_vectors = model.wv
    print("Made WVs")

    word_vectors.save(os.path.join(root_dir, "word2vec"))
    print("Saved WVs")

    word_vectors.save_word2vec_format(
        os.path.join(root_dir, "word2vec.txt"), fvocab=os.path.join(root_dir, "vocab.txt")
    )
    print("Complete!")
