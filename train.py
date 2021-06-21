# This file is only to show the interaction with the AL agent
# All intermediary/task specific routines are imported from elsewhere

from torch._utils import _accumulate

from active_learning.util_classes import OneDimensionalSequenceTaggingDataset, SentenceIndex
from model.ner_model import Model
from model.utils import Helper
from utils import Charset, Vocabulary, Index, load

from ner_utils.parse_args import parse_args
from ner_utils.training_utils import *
from ner_utils.logging_utils import *
from ner_utils.training_routines import *

from active_learning.acquisition import RandomBaselineAcquisition, LowestConfidenceAcquisition, \
    MaximumEntropyAcquisition, BALDAcquisition, SimpleAggregation, EmbeddingMigrationAcquisition
from active_learning.agent import ActiveLearningAgent
from active_learning.util_classes import ALAttribute, SentenceSubsequence
from active_learning.selector import FixedWindowSelector, SentenceSelector, VariableWindowSelector


embedding_form = lambda x: None

acquisition_dict = {
    "rand": RandomBaselineAcquisition,
    "baseline": RandomBaselineAcquisition,
    "lc": LowestConfidenceAcquisition,
    "maxent": MaximumEntropyAcquisition,
    "emb_mig": EmbeddingMigrationAcquisition
}

extra_al_attributes = {
    "emb_mig": {"embeddings": embedding_form}
}


def configure_al_agent(args, device, model, train_set, helper):

    round_size = int(args.roundsize)
    acquisition_class = acquisition_dict[args.acquisition](train_set)
    acquisition_agg = SimpleAggregation([acquisition_class], train_set, [1])

    if len(args.window) == 1 and int(args.window[0]) == -1:
        if args.beam_search != 1: raise ValueError("Full sentence selection requires a beam search parameter of 1")
        selector = SentenceSelector(helper, normalisation_index=args.alpha, round_size=round_size,
                                    acquisition=acquisition_agg, window_class=SentenceSubsequence)
    elif len(args.window) == 1 and int(args.window[0]) != -1:
        selector = FixedWindowSelector(
            helper, window_size=int(args.window[0]), beta=args.beta, round_size=round_size,
            beam_search_parameter=args.beam_search, acquisition=acquisition_agg, window_class=SentenceSubsequence
        )
    elif len(args.window) == 2:
        selector = VariableWindowSelector(
            helper=helper, window_range=[int(a) for a in args.window], beta=args.beta, round_size=round_size,
            beam_search_parameter=args.beam_search, normalisation_index=args.alpha, acquisition=acquisition_agg,
            window_class=SentenceSubsequence
        )
    else:
        raise ValueError(f"Windows must be of one or two size, not {args.window}")

    if args.acquisition == 'baseline' and args.initprop != 1.0:
        raise ValueError("To run baseline, you must set initprop == 1.0")

    agent = ActiveLearningAgent(
        train_set=train_set,
        selector_class=selector,
        round_size=round_size,
        batch_size=args.batch_size,
        helper=helper,
        device=device,
        model=model,
        propagation_mode=args.propagation_mode
    )

    return agent


def random_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths)).tolist()
    slicers = [indices[offset - length:offset] for offset, length in zip(_accumulate(lengths), lengths)]

    return indices, [[dataset[s] for s in sl] for sl in slicers]


def load_dataset(path):
    charset = Charset()

    vocab = Vocabulary()
    vocab.load(f"{path}/vocab.txt")

    tag_set = Index()
    tag_set.load(f"{path}/tag2id.txt")

    measure_type = get_measure_type(path)

    tag_set = Index()
    if measure_type == 'relations':
        tag_set.load(f"{path}/tag2id.txt")
    elif measure_type == 'entities':
        tag_set.load(f"{path}/entity_labels.txt")

    helper = Helper(vocab, tag_set, charset, measure_type=measure_type)

    # relation_labels = Index()
    # relation_labels.load(f"{path}/relation_labels.txt")

    train_data = load(f"{path}/train.pk")[:1000]
    test_data = load(f"{path}/test.pk")

    word_embeddings = np.load(f"{path}/word2vec.vectors.npy")

    return helper, word_embeddings, train_data, test_data, tag_set


def active_learning_train(args):
    # set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            logging.warning("you have a CUDA device, so you should probably run with --cuda")

    device = "cpu"#torch.device("cuda" if args.cuda else "cpu")

    helper, word_embeddings, train_set, test_set, tag_set = load_dataset(args.data_path)
    val_size = int(0.01 * len(train_set))
    indices, (train_set, val_set) = random_split(train_set, [len(train_set) - val_size, val_size])

    val_data_groups = group(val_set, [10, 20, 30, 40, 50, 60])
    test_data_groups = group(test_set, [10, 20, 30, 40, 50, 60])

    al_attributes = [
        ALAttribute(name=k, unit_form=v,
                    initialisation=[np.nan*np.ones(v(np.array(d[0]))) for d in train_set], cache=True)
        for k, v in extra_al_attributes.get(args.acquisition, {}).items()
    ]

    data_setify = lambda set, ind: [np.array(d[ind]) for d in set]

    train_set = OneDimensionalSequenceTaggingDataset(
        data=data_setify(train_set, 0),
        labels=[torch.nn.functional.one_hot(torch.tensor(d[-1]), len(tag_set)) for d in train_set],
        index_class=SentenceIndex,
        semi_supervision_multiplier=args.beta,
        padding_token=helper.vocab["<pad>"],
        empty_tag=helper.tag_set["O"],
        label_form=lambda x: (x.shape[0], len(tag_set.idx2key)),
        al_attributes=al_attributes
    )

    # [vocab[a] for a in test_data[0][0]]   gives a sentence
    # [tag_set[a] for a in test_data[0][2]] gives the corresponding tagseq

    test_set = OneDimensionalSequenceTaggingDataset(
        data=data_setify(test_set, 0),
        labels=[torch.nn.functional.one_hot(torch.tensor(d[-1]), len(tag_set)) for d in test_set],
        index_class=SentenceIndex,
        semi_supervision_multiplier=args.beta,
        padding_token=helper.vocab["<pad>"],
        empty_tag=helper.tag_set["O"],
        label_form=lambda x: (x.shape[0], len(tag_set.idx2key))
    )

    val_set = OneDimensionalSequenceTaggingDataset(
        data=data_setify(val_set, 0),
        labels=[torch.nn.functional.one_hot(torch.tensor(d[-1]), len(tag_set)) for d in val_set],
        index_class=SentenceIndex,
        semi_supervision_multiplier=args.beta,
        padding_token=helper.vocab["<pad>"],
        empty_tag=helper.tag_set["O"],
        label_form=lambda x: (x.shape[0], len(tag_set.idx2key))
    )

    for i in val_set.index.labelled_idx:
        val_set.index.label_instance(i)
    for i in test_set.index.labelled_idx:
        test_set.index.label_instance(i)

    model, criterion = initialise_model(word_embeddings, args, helper, device, model_class=Model, loss_type=ModifiedKL)

    agent = configure_al_agent(args, device, model, train_set, helper)
    agent.init(int(len(train_set) * args.initprop))

    # logger
    root_dir = make_root_dir(args, indices)

    round_num = 0
    for _ in agent:
        original_lr = args.lr
        round_results = train_full(model, device, agent, helper, val_set, val_data_groups, original_lr, criterion, args)
        test_loss, test_precision, test_recall, test_f1 = evaluate(
            model, GroupBatchRandomSampler(test_data_groups, args.batch_size, drop_last=False), test_set, helper,
            criterion, device)
        log_round(root_dir, round_results, agent, test_loss, test_precision, test_recall, test_f1, round_num)
        round_num += 1


if __name__ == "__main__":
    configure_logger()
    op_args = parse_args()
    active_learning_train(args=op_args)
