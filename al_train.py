import argparse
import datetime
import os
import time

from torch import optim

from active_learning.helper import configure_al_agent
from model import Model, Helper
from training_utils import *
from utils import Charset, Vocabulary, Index, load


def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint Extraction of Entities and Relations"
    )
    parser.add_argument(
        "-W", "--window", type=int, help="size of window acquired each time in words. set -1 for full sentence",
        required=False, default=-1
    )
    parser.add_argument(
        "-A", "--acquisition", type=str, help="acquisition function used by agent. choose from 'rand' and 'lc'",
        required=True
    )
    parser.add_argument(
        "-I", "--initprop", type=float,
        help="proportion of sentences of training set labelled before first round. [0,1]",
        required=True
    )
    parser.add_argument(
        "-R", "--roundsize", type=int, help="number of acquisitions made per round (unitless)", required=True
    )
    parser.add_argument(
        "--earlystopping", type=int, help="number of epochs of F1 decrease before early stopping", default=3
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature of prediction annealing ", default=1.0
    )
    # parser.add_argument(
    #     "--labelthres", type=float, help="proportion of sentence that must be manually labelled before it is used for training", required = True
    # )

    parser.add_argument(
        "--batch_size", type=int, default=32, metavar="N", help="batch size (default: 32)"
    )
    parser.add_argument("--cuda", default=False, action="store_false", help="use CUDA (default: True)")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="dropout applied to layers (default: 0.5)"
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
        "--emsize", type=int, default=50, help="size of character embeddings (default: 50)"
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
        "--lr", type=float, default=0.02, help="initial learning rate (default: 4)"
    )
    parser.add_argument(
        "--lr_decrease", type=float, default=2,
        help="learning rate annealing factor on non-improving epochs (default: 2)"
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


def make_root_dir(args):
    rn = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    root_dir = os.path.join('.', f"record-{args.acquisition}-{rn}")
    os.mkdir(root_dir)

    with open(os.path.join(root_dir, "config.txt"), "w") as config_file:
        config_file.write(f"Run started at: {rn} \n")
        config_file.write(f"\n")
        # config_file.write(f"Number of train sentences {len(train_data)} \n")
        # config_file.write(f"Number of test sentences {len(test_data)} \n")
        config_file.write(f"Proportion of initial labelled sentences: {args.initprop} \n")
        config_file.write(f"Number of acquisitions per round: {args.roundsize} \n")
        config_file.write(f"\n")
        config_file.write(f"Acquisition strategy: {args.acquisition} \n")
        config_file.write(f"Size of windows (-1 means whole sentence): {args.window} \n")
        config_file.write(f"\n")
        config_file.write(f"All args: \n")
        for k, v in vars(args).items():
            config_file.write(str(k))
            config_file.write(" ")
            config_file.write(str(v))
            config_file.write("\n")

    return root_dir


def early_stopping_original(f1_list, num):
    if len(f1_list) < num:
        return False
    elif sorted(f1_list[-num:], reverse=True) == f1_list[-num:]:
        return True
    else:
        return False


def early_stopping(f1_list, num=3):
    if num < 0:
        return False
    if len(f1_list) < num:
        return False
    elif len(f1_list) - np.argmax(f1_list) > num:
        return True
    else:
        return False


def measure(output, targets, lengths):
    assert output.size(0) == targets.size(0) and targets.size(0) == lengths.size(0)
    tp = 0
    tp_fp = 0
    tp_fn = 0
    batch_size = output.size(0)
    output = torch.argmax(output, dim=-1)
    targets = torch.argmax(targets, dim=-1)
    for i in range(batch_size):
        length = lengths[i]
        out = output[i][:length].tolist()
        target = targets[i][:length].tolist()
        out_triplets = get_triplets(out)
        tp_fp += len(out_triplets)
        target_triplets = get_triplets(target)
        tp_fn += len(target_triplets)
        for target_triplet in target_triplets:
            for out_triplet in out_triplets:
                if out_triplet == target_triplet:
                    tp += 1
    return tp, tp_fp, tp_fn


def train_epoch(model, al_agent, start_time, epoch):
    model.train()
    total_loss = 0
    count = 0

    sampler = al_agent.labelled_set
    for idx, batch_indices in enumerate(sampler):

        model.eval()
        sentences, tokens, targets, lengths = al_agent.get_batch(idx)
        model.train()

        optimizer.zero_grad()
        output = model(sentences, tokens)
        # output in shape [batch_size, length_of_sentence, num_tags (193)]

        # output = pack_padded_sequence(output, lengths.cpu(), batch_first=True).data
        # targets = pack_padded_sequence(targets, lengths.cpu(), batch_first=True).data

        loss = criterion(output, targets)
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        count += len(targets)

        if (idx + 1) % args.log_interval == 0:
            cur_loss = total_loss / count
            elapsed = time.time() - start_time
            percent = ((epoch - 1) * len(sampler) + (idx + 1)) / (
                    args.epochs * len(sampler)
            )
            remaining = elapsed / percent - elapsed
            print(
                "| Epoch {:2d}/{:2d} | Batch {:5d}/{:5d} | Elapsed Time {:s} | Remaining Time {:s} | "
                "lr {:4.2e} | Loss {:5.3f} |".format(
                    epoch,
                    args.epochs,
                    idx + 1,
                    len(sampler),
                    time_display(elapsed),
                    time_display(remaining),
                    args.lr,
                    cur_loss,
                )
            )
            total_loss = 0
            count = 0


# NOT CHANGED BY AL
def evaluate(model, data_sampler, dataset):
    model.eval()
    total_loss = 0
    count = 0
    TP = 0
    TP_FP = 0
    TP_FN = 0
    with torch.no_grad():
        print("Beginning evaluation")
        for batch_indices in data_sampler:
            sentences, tokens, targets, lengths, kl_mask = get_batch(
                batch_indices, dataset, device
            )

            output = model(sentences, tokens)
            tp, tp_fp, tp_fn = measure(output, targets, lengths)
            # tp, tp_fp, tp_fn = 5,5,5

            TP += tp
            TP_FP += tp_fp
            TP_FN += tp_fn

            loss = criterion(output, targets, kl_mask)
            total_loss += loss.item()

            count += len(targets)
    if TP_FP == 0:
        TP_FP = 1
    if TP_FN == 0:
        TP_FN = 1
    return total_loss / count, TP / TP_FP, TP / TP_FN, 2 * TP / (TP_FP + TP_FN)


def train_full(model, agent, val_set):
    lr = args.lr

    all_val_loss = []
    all_val_precision = []
    all_val_recall = []
    all_val_f1 = []

    all_train_loss = []
    all_train_precision = []
    all_train_recall = []
    all_train_f1 = []

    start_time = time.time()
    print("-" * 118)

    num_sentences = agent.index.get_number_partially_labelled_sentences()
    num_words = agent.budget_spent()

    print(f"Starting training with {num_words} words labelled in {num_sentences} sentences")
    for epoch in range(1, args.epochs + 1):

        if early_stopping(all_val_f1, args.earlystopping):
            break

        train_epoch(model, agent, start_time, epoch)

        val_loss, val_precision, val_recall, val_f1 = \
            evaluate(model, GroupBatchRandomSampler(val_data_groups, args.batch_size, drop_last=False), val_set)
        train_loss, train_precision, train_recall, train_f1 = evaluate(model, agent.labelled_batch_indices,
                                                                       agent.train_data)

        elapsed = time.time() - start_time
        print(
            "| End of Epoch {:2d} | Elapsed Time {:s} | Validation Loss {:5.3f} | Precision {:5.3f} "
            "| Recall {:5.3f} | F1 {:5.3f} |".format(
                epoch, time_display(elapsed), val_loss, val_precision, val_recall, val_f1
            )
        )

        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        if len(all_val_loss) and val_loss > max(all_val_loss):
            lr = lr / args.lr_decrease
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        all_val_loss.append(val_loss)
        all_val_precision.append(val_precision)
        all_val_recall.append(val_recall)
        all_val_f1.append(val_f1)

        all_train_loss.append(train_loss)
        all_train_precision.append(train_precision)
        all_train_recall.append(train_recall)
        all_train_f1.append(train_f1)

    return {
        "num_words": num_words,
        "num_sentences": num_sentences,
        "all_val_loss": all_val_loss,
        "all_val_precision": all_val_precision,
        "all_val_recall": all_val_recall,
        "all_val_f1": all_val_f1,
        "all_train_loss": all_train_loss,
        "all_train_precision": all_train_precision,
        "all_train_recall": all_train_recall,
        "all_train_f1": all_train_f1,
    }


def log_round(root_dir, round_results, agent, test_loss, test_precision, test_recall, test_f1):
    num_words = round_results["num_words"]
    num_sentences = round_results["num_sentences"]

    round_dir = os.path.join(root_dir, f"round-{round}")
    os.mkdir(round_dir)
    print(f"Logging round {round}")

    with open(
            os.path.join(round_dir, f"record.tsv"), "wt", encoding="utf-8"
    ) as f:
        f.write("Epoch\tT_LOSS\tT_PREC\tT_RECL\tT_F1\tV_LOSS\tV_PREC\tV_RECL\tV_F1\n")
        for idx in range(len(round_results["all_val_loss"])):
            f.write(
                "{:d}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n".format(
                    idx + 1,
                    round_results["all_train_loss"][idx],
                    round_results["all_train_precision"][idx],
                    round_results["all_train_recall"][idx],
                    round_results["all_train_f1"][idx],
                    round_results["all_val_loss"][idx],
                    round_results["all_val_precision"][idx],
                    round_results["all_val_recall"][idx],
                    round_results["all_val_f1"][idx],
                )
            )
        f.write(
            "\nTEST_LOSS\tTEST_PREC\tTEST_RECALL\tTEST_F1\n"
        )
        f.write(
            "{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n".format(
                test_loss, test_precision, test_recall, test_f1
            )
        )
        f.write(
            f"\n{num_words} words in {num_sentences} sentences. Total {len(train_set)} sentences.\n\n"
        )

    with open(
            os.path.join(round_dir, f"sentence_prop-{round}.tsv"), "wt", encoding="utf-8"
    ) as f:

        f.write("Proportions of sentences in words that have been manually labelled \n")
        f.write("The remainder of the words in the sentence have been automatically labelled by")
        f.write("the model before training, as per the threshold argument (if implemented) \n")
        f.write("\n")

        i = 0
        for sentence_idx, labelled_idx in agent.labelled_idx.items():
            num_labelled = len(labelled_idx)
            num_unlabelled = len(agent.unlabelled_set[sentence_idx])
            prop = num_labelled / (num_labelled + num_unlabelled)
            f.write(str(prop))
            i += 1
            if i % 10 == 0:
                f.write("\n")
                i = 0
            else:
                f.write("\t")

    # with open(
    #         os.path.join(round_dir, f"sentence_labels-{round}"), "wt", encoding="utf-8"
    # ) as f:
    #     f.write("This file shows sentences used in training for this round.\n")
    #     f.write("For each sentence, first row represents the words in the sentence.\n")
    #     f.write(
    #         "\tUPPERCASE are words labelled by the annotator, lowercase are the words automatically labelled by the model.\n")
    #     f.write("The second row is the ground truth labels.\n")
    #     f.write("The third row is the labels used by the model in training.\n")
    #     f.write("\n")
    #     f.write(
    #         "For full sentence labelling all words will be uppercase. For oracles the second and third row will be the same for uppercase words\n")
    #     f.write("\n")
    #
    #     for batch_indices in agent.labelled_batch_indices:
    #
    #         sentences, _, used_targets, _, _ = agent.get_batch(
    #             batch_indices, agent.autolabelled_data, device
    #         )
    #         _, _, real_targets, _, _ = get_batch(
    #             batch_indices, train_data, device
    #         )
    #
    #         for j, batch_idx in enumerate(batch_indices):
    #             labelled_idx = agent.labelled_idx[batch_idx]
    #             f.write("\n")
    #             f.write(f"Sentence {batch_idx}\n")
    #             f.write(
    #                 "\t".join([
    #                     vocab[int(v)].upper() if k in labelled_idx else vocab[int(v)].upper() for k, v in
    #                     enumerate(sentences[j])
    #                 ])
    #             )
    #             f.write("\n")
    #             f.write(
    #                 "\t".join([
    #                     tag_set[int(v)] for v in real_targets[j]
    #                 ])
    #             )
    #             f.write("\n")
    #             f.write(
    #                 "\t".join([
    #                     tag_set[int(v)] for v in used_targets[j]
    #                 ])
    #             )
    #             f.write("\n")

    print(f"Finished logging round {round}")


def load_dataset(path):
    charset = Charset()

    vocab = Vocabulary()
    vocab.load(f"{path}/vocab.txt")

    tag_set = Index()
    tag_set.load(f"{path}/tag2id.txt")

    helper = Helper(vocab, tag_set, charset)

    relation_labels = Index()
    relation_labels.load(f"{path}/relation_labels.txt")

    train_data = load(f"{path}/train.pk")
    test_data = load(f"{path}/test.pk")

    word_embeddings = np.load(f"{path}/word2vec.vectors.npy")

    return helper, word_embeddings, train_data, test_data


if __name__ == "__main__":
    args = parse_args()

    # set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")

    # TODO: make the path a parameter
    helper, word_embeddings, train_set, test_set = load_dataset('./data/NYT_CoType')

    # CHANGED FOR DEBUG
    val_size = int(0.01 * len(train_set))
    train_set, val_set = random_split(train_set, [len(train_set) - val_size, val_size])

    # [vocab[a] for a in test_data[0][0]]   gives a sentence
    # [tag_set[a] for a in test_data[0][2]] gives the corresponding tagseq

    val_data_groups = group(val_set, [10, 20, 30, 40, 50, 60])
    test_data_groups = group(test_set, [10, 20, 30, 40, 50, 60])

    word_embeddings = torch.tensor(word_embeddings)
    word_embedding_size = word_embeddings.size(1)
    pad_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
    unk_embedding = torch.empty(1, word_embedding_size).uniform_(-0.5, 0.5)
    word_embeddings = torch.cat([pad_embedding, unk_embedding, word_embeddings])

    char_channels = [args.emsize] + [args.char_nhid] * args.char_layers

    word_channels = [word_embedding_size + args.char_nhid] + [
        args.word_nhid
    ] * args.word_layers

    weight = [args.weight] * len(helper.tag_set)
    weight[helper.tag_set["O"]] = 1
    weight = torch.tensor(weight).to(device)

    criterion = ModifiedKL(weight)

    model = Model(
        charset_size=len(helper.charset),
        char_embedding_size=args.emsize,
        char_channels=char_channels,
        char_padding_idx=helper.charset["<pad>"],
        char_kernel_size=args.char_kernel_size,
        weight=word_embeddings,
        word_embedding_size=word_embedding_size,
        word_channels=word_channels,
        word_kernel_size=args.word_kernel_size,
        num_tag=len(helper.tag_set),
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        T=args.temperature
    ).to(device)

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

    agent = configure_al_agent(args, device, model, train_set, helper)
    agent.init(int(len(train_set) * args.initprop))

    # logger
    root_dir = make_root_dir(args)

    round = 0
    while agent.budget > 0:
        round_results = train_full(model, agent, val_set)

        # Run on test data
        test_loss, test_precision, test_recall, test_f1 = evaluate(
            model, GroupBatchRandomSampler(test_data_groups, args.batch_size, drop_last=False), test_set
        )

        print("=" * 118)
        print(
            "| End of Training | Test Loss {:5.3f} | Precision {:5.3f} "
            "| Recall {:5.3f} | F1 {:5.3f} |".format(test_loss, test_precision, test_recall, test_f1)
        )
        print("=" * 118)

        log_round(root_dir, round_results, agent, test_loss, test_precision, test_recall, test_f1)

        model.eval()
        agent.update_indices(model)
        agent.update_datasets(model)
        round += 1
