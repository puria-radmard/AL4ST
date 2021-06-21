import logging, datetime, os, sys, json
from tqdm import tqdm

def configure_logger():
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger(__name__).addHandler(TqdmLoggingHandler())


def make_root_dir(args, indices):
    rn = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    root_dir = os.path.join(
        '.',
        f"{'-'.join(sys.argv[1:])}--{rn}".replace("/", "")
    )
    os.mkdir(root_dir)

    with open(os.path.join(root_dir, "config.txt"), "w") as config_file:
        config_file.write(f"Run started at: {rn} \n")
        config_file.write(f"\n")
        # config_file.write(f"Number of train sentences {len(train_data)} \n")
        # config_file.write(f"Number of test sentences {len(test_data)} \n")
        config_file.write(f"AFTER 17/12/2020 LOGGING CHANGE - ALL UNUSED WINDOWS INCLUDED\n")
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

    with open(os.path.join(root_dir, "dataset_indices.json"), "w") as jfile:
        json.dump(indices, jfile)

    return root_dir


def log_round(root_dir, round_results, agent, test_loss, test_precision, test_recall, test_f1, round_num):

    logging.info(
        "| end of training | test loss {:5.4f} | prec {:5.4f} "
        "| rec {:5.4f} | f1 {:5.4f} |".format(test_loss, test_precision, test_recall, test_f1)
    )

    num_words = round_results["num_words"]
    num_sentences = round_results["num_sentences"]

    round_dir = os.path.join(root_dir, f"round-{round_num}")
    os.mkdir(round_dir)
    logging.info(f"logging round {round_num}")

    agent.save(round_dir)

    with open(os.path.join(round_dir, f"record.tsv"), "wt", encoding="utf-8") as f:
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
            f"\n{num_words} words in {num_sentences} sentences. Total {len(agent.train_set)} sentences.\n\n"
        )

    with open(
            os.path.join(round_dir, f"sentence_prop-{round_num}.tsv"), "wt", encoding="utf-8"
    ) as f:

        f.write("sent_length\tnum_labelled\tnum_temp_labelled\n")

        i = 0
        for sentence_idx, labelled_idx in agent.train_set.index.labelled_idx.items(): # not used in final

            num_unlabelled = len(agent.train_set.index.unlabelled_idx[sentence_idx])
            num_temp_labelled = len(agent.train_set.index.temp_labelled_idx[sentence_idx])
            f.write(str(len(labelled_idx) + num_unlabelled + num_temp_labelled))
            f.write("\t")
            f.write(str(len(labelled_idx)))
            f.write("\t")
            f.write(str(num_temp_labelled))
            f.write("\n")

    logging.info(f"finished logging round {round_num}")