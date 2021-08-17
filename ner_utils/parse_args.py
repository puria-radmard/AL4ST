import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint Extraction of Entities and Relations"
    )
    parser.add_argument(
        "-W",
        "--window",
        nargs="+",
        help="size of window acquired each time in words. set -1 for full sentence, two for range",
        required=True,
    )
    parser.add_argument(
        "-A",
        "--acquisition",
        type=str,
        help="acquisition function used by agent. choose from 'rand' and 'lc'",
        required=True,
    )
    parser.add_argument(
        "-I",
        "--initprop",
        type=float,
        help="proportion of sentences of training set labelled before first round. [0,1]",
        default=0.1,
    )
    parser.add_argument(
        "-R",
        "--roundsize",
        type=int,
        help="number of words acquired made per round (rounded up to closest possible each round)",
        default=80000,
    )
    parser.add_argument(
        "-propagation",
        "--propagation_mode",
        type=int,
        help="0 = no propagation, 1 = propagation, training on all sentences, 2 = propagation, training only on sentences with some real labels",
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Weight (should be in [0,1]) that self-supervised losses are multiplied by",
        required=True,
    )
    parser.add_argument(
        "-alpha",
        "--alpha",
        type=float,
        help="sub-sequence are normalised by L^-alpha where L is the subsequence length",
        required=True,
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        help="Temperature of scoring annealing. Does not affect W=-1 and beta=0 cases",
        required=True,
    )
    parser.add_argument(
        "-B",
        "--beam_search",
        type=int,
        default=1,
        help="Beam search parameter. B=1 means a greedy search",
    )
    parser.add_argument(
        "-D",
        "--data_path",
        type=str,
        default="/home/pradmard/repos/data/OntoNotes-5.0/NER/",
    )
    # parser.add_argument(
    #     "--labelthres", type=float, help="proportion of sentence that must be manually labelled before it is used
    #     for training", required = True
    # )

    parser.add_argument(
        "--earlystopping",
        type=int,
        help="number of epochs of F1 decrease before early stopping",
        default=2,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="batch size (default: 32)",
    )
    parser.add_argument(
        "--cuda", default=True, action="store_true", help="use CUDA (default: True)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="dropout applied to layers (default: 0.5)",
    )
    parser.add_argument(
        "--emb_dropout",
        type=float,
        default=0.25,
        help="dropout applied to the embedded layer (default: 0.25)",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.35,
        help="gradient clip, -1 means no clip (default: 0.35)",
    )
    # debug
    parser.add_argument(
        "--epochs", type=int, default=30, help="upper epoch limit (default: 30)"
    )
    parser.add_argument(
        "--char_kernel_size",
        type=int,
        default=3,
        help="character-level kernel size (default: 3)",
    )
    parser.add_argument(
        "--word_kernel_size",
        type=int,
        default=3,
        help="word-level kernel size (default: 3)",
    )
    parser.add_argument(
        "--emsize",
        type=int,
        default=50,
        help="size of character embeddings (default: 50)",
    )
    parser.add_argument(
        "--char_layers",
        type=int,
        default=3,
        help="# of character-level convolution layers (default: 3)",
    )
    parser.add_argument(
        "--word_layers",
        type=int,
        default=3,
        help="# of word-level convolution layers (default: 3)",
    )
    parser.add_argument(
        "--char_nhid",
        type=int,
        default=50,
        help="number of hidden units per character-level convolution layer (default: 50)",
    )
    parser.add_argument(
        "--word_nhid",
        type=int,
        default=300,
        help="number of hidden units per word-level convolution layer (default: 300)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="report interval (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=1, help="initial learning rate (default: 1)"
    )
    parser.add_argument(
        "--lr_decrease",
        type=float,
        default=1,
        help="learning rate annealing factor on non-improving epochs (default: 1)",
    )
    parser.add_argument(
        "--optim", type=str, default="SGD", help="optimizer type (default: SGD)"
    )
    parser.add_argument(
        "--seed", type=int, default=1111, help="random seed (default: 1111)"
    )
    parser.add_argument(
        "--save", type=str, default="model.pt", help="path to save the final model"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=10.0,
        help='manual rescaling weight given to each tag except "O"',
    )
    return parser.parse_args()
