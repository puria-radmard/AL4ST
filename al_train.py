import argparse
import datetime

from tqdm import tqdm

from active_learning import configure_al_agent
from training_utils import *
from utils import *

dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    description="Joint Extraction of Entities and Relations"
)
parser.add_argument(
    "-W", "--window", type=int, help="size of window acquired each time in words. set -1 for full sentence",
    required=True
)
parser.add_argument(
    "-A", "--acquisition", type=str, help="acquisition function used by agent. choose from 'rand' and 'lc'",
    required=True
)
parser.add_argument(
    "-I", "--initprop", type=float, help="proportion of sentences of training set labelled before first round. [0,1]",
    required=True
)
parser.add_argument(
    "-R", "--roundsize", type=int, help="number of acquisitions made per round (unitless)", required=True
)
parser.add_argument(
    "--earlystopping", type=int, help="number of epochs of F1 decrease before early stopping", default=3
)
# parser.add_argument(
#     "--labelthres", type=float, help="proportion of sentence that must be manually labelled before it is used for training", required = True
# )

parser.add_argument(
    "--batch_size", type=int, default=32, metavar="N", help="batch size (default: 32)"
)
parser.add_argument("--cuda", default=True, action="store_false", help="use CUDA (default: True)")
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
    "--lr", type=float, default=4, help="initial learning rate (default: 4)"
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
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

char_channels = [args.emsize] + [args.char_nhid] * args.char_layers


def make_root_dir(args):
    rn = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    root_dir = os.path.join(dir_path, f"record-{rn}")
    os.mkdir(root_dir)

    with open(os.path.join(root_dir, "config.txt"), "w") as config_file:
        config_file.write(f"Run started at: {rn} \n")
        config_file.write(f"\n")
        config_file.write(f"Number of train sentences {len(train_data)} \n")
        config_file.write(f"Number of test sentences {len(test_data)} \n")
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


def make_model():
    return Model(
        charset_size=len(charset),
        char_embedding_size=args.emsize,
        char_channels=char_channels,
        char_padding_idx=charset["<pad>"],
        char_kernel_size=args.char_kernel_size,
        weight=word_embeddings,
        word_embedding_size=word_embedding_size,
        word_channels=word_channels,
        word_kernel_size=args.word_kernel_size,
        num_tag=len(tag_set),
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
    ).to(device)


word_channels = [word_embedding_size + args.char_nhid] + [
    args.word_nhid
] * args.word_layers

weight = [args.weight] * len(tag_set)
weight[tag_set["O"]] = 1
weight = torch.tensor(weight).to(device)
criterion = nn.NLLLoss(weight, size_average=False)


def early_stopping_original(f1_list, num):
    if len(f1_list) < num:
        return False
    elif sorted(f1_list[-num:], reverse=True) == f1_list[-num:]:
        return True
    else:
        return False


def early_stopping(f1_list, num=3):
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


def train(model, al_agent):
    model.train()
    total_loss = 0
    count = 0
    sampler = al_agent.labelled_batch_indices

    for idx, batch_indices in enumerate(sampler):
        sentences, tokens, targets, lengths = get_batch(
            batch_indices, al_agent.autolabelled_data, device
        )

        # [vocab[int(a)] for a in sentences[10]] reveals sentence
        # [tag_set[int(a)] for a in targets[10]] reveals corresponding taglist

        optimizer.zero_grad()
        output = model(sentences, tokens)
        # output in shape [batch_size, length_of_sentence, num_tags (193)]

        output = pack_padded_sequence(output, lengths.cpu(), batch_first=True).data
        targets = pack_padded_sequence(targets, lengths.cpu(), batch_first=True).data
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
                    lr,
                    cur_loss,
                )
            )
            total_loss = 0
            count = 0


# NOT CHANGED BY AL
def evaluate(model, data_groups):
    model.eval()
    total_loss = 0
    count = 0
    TP = 0
    TP_FP = 0
    TP_FN = 0
    with torch.no_grad():
        print("Beginning evaluation")
        for batch_indices in tqdm(
                GroupBatchRandomSampler(data_groups, args.batch_size, drop_last=False)
        ):
            sentences, tokens, targets, lengths = get_batch(
                batch_indices, train_data, device
            )

            output = model(sentences, tokens)
            tp, tp_fp, tp_fn = measure(output, targets, lengths)
            # tp, tp_fp, tp_fn = 5,5,5

            TP += tp
            TP_FP += tp_fp
            TP_FN += tp_fn
            output = pack_padded_sequence(output, lengths.cpu(), batch_first=True).data
            targets = pack_padded_sequence(targets, lengths.cpu(), batch_first=True).data
            loss = criterion(output, targets)
            total_loss += loss.item()

            count += len(targets)
    if TP_FP == 0:
        TP_FP = 1
    if TP_FN == 0:
        TP_FN = 1
    return total_loss / count, TP / TP_FP, TP / TP_FN, 2 * TP / (TP_FP + TP_FN)


if __name__ == "__main__":

    model = make_model()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

    agent = configure_al_agent(args, device, model, train_data, test_data)
    root_dir = make_root_dir(args)
    round = 0

    while agent.budget > 0:

        best_val_loss = None
        lr = args.lr
        all_val_loss = []
        all_precision = []
        all_recall = []
        all_f1 = []

        start_time = time.time()
        print("-" * 118)

        word_inds = [v for k, v in agent.labelled_idx.items() if v]
        num_sentences = len(word_inds)
        num_words = sum([len(b) for b in word_inds])

        print(
            f"Starting training with {num_words} words labelled in {num_sentences} sentences"
        )
        for epoch in range(1, args.epochs + 1):

            if early_stopping(all_f1, args.earlystopping):
                break

            train(model=model, al_agent=agent)

            val_loss, precision, recall, f1 = evaluate(model, val_data_groups)

            elapsed = time.time() - start_time
            print(
                "| End of Epoch {:2d} | Elapsed Time {:s} | Validation Loss {:5.3f} | Precision {:5.3f} "
                "| Recall {:5.3f} | F1 {:5.3f} |".format(
                    epoch, time_display(elapsed), val_loss, precision, recall, f1
                )
            )

            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr = lr / 4.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            all_val_loss.append(val_loss)
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

        # Run on test data
        test_loss, precision, recall, f1 = evaluate(model, test_data_groups)
        print("=" * 118)
        print(
            "| End of Training | Test Loss {:5.3f} | Precision {:5.3f} "
            "| Recall {:5.3f} | F1 {:5.3f} |".format(test_loss, precision, recall, f1)
        )
        print("=" * 118)

        round_dir = os.path.join(root_dir, f"round-{round}")
        os.mkdir(round_dir)
        print(f"Logging round {round}")

        with open(
                os.path.join(round_dir, f"record.tsv"), "wt", encoding="utf-8"
        ) as f:
            f.write(
                f"{num_words} words in {num_sentences} sentences. Total {len(train_data)} sentences.\n\n"
            )
            f.write("Epoch\tLOSS\tPREC\tRECL\tF1\n")
            for idx in range(len(all_val_loss)):
                f.write(
                    "{:d}\t{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n".format(
                        idx + 1,
                        all_val_loss[idx],
                        all_precision[idx],
                        all_recall[idx],
                        all_f1[idx],
                    )
                )
            f.write(
                "\n{:5.3f}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n".format(
                    test_loss, precision, recall, f1
                )
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
                num_unlabelled = len(agent.unlabelled_idx[sentence_idx])
                prop = num_labelled / (num_labelled + num_unlabelled)
                f.write(str(prop))
                i += 1
                if i % 10 == 0:
                    f.write("\n")
                    i = 0
                else:
                    f.write("\t")

        with open(
                os.path.join(round_dir, f"sentence_labels-{round}"), "wt", encoding="utf-8"
        ) as f:
            f.write("This file shows sentences used in training for this round.\n")
            f.write("For each sentence, first row represents the words in the sentence.\n")
            f.write(
                "\tUPPERCASE are words labelled by the annotator, lowercase are the words automatically labelled by the model.\n")
            f.write("The second row is the ground truth labels.\n")
            f.write("The third row is the labels used by the model in training.\n")
            f.write("\n")
            f.write(
                "For full sentence labelling all words will be uppercase. For oracles the second and third row will be the same for uppercase words\n")
            f.write("\n")

            for batch_indices in agent.labelled_batch_indices:

                sentences, _, used_targets, _ = get_batch(
                    batch_indices, agent.autolabelled_data, device
                )
                _, _, real_targets, _ = get_batch(
                    batch_indices, train_data, device
                )

                for j, batch_idx in enumerate(batch_indices):
                    labelled_idx = agent.labelled_idx[batch_idx]
                    f.write("\n")
                    f.write(f"Sentence {batch_idx}\n")
                    f.write(
                        "\t".join([
                            vocab[int(v)].upper() if k in labelled_idx else vocab[int(v)].upper() for k, v in
                            enumerate(sentences[j])
                        ])
                    )
                    f.write("\n")
                    f.write(
                        "\t".join([
                            tag_set[int(v)] for v in real_targets[j]
                        ])
                    )
                    f.write("\n")
                    f.write(
                        "\t".join([
                            tag_set[int(v)] for v in used_targets[j]
                        ])
                    )
                    f.write("\n")

        print(f"Finished logging round {round}")

        model.eval()
        agent.update_indices(model)
        agent.update_datasets(model)
        round += 1
