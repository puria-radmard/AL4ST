from torch import nn, optim
import torch

from utils import time_display
from .training_utils import EarlyStopper, GroupBatchRandomSampler
import time, logging


def train_epoch(model, device, agent, start_time, epoch, optimizer, criterion, args):
    model.train()
    total_loss = 0
    count = 0

    sampler = agent.labelled_set
    for idx, batch_indices in enumerate(sampler):

        model.eval()
        sentences, targets, lengths, self_supervision_mask = [
            a.to(device)
            for a in agent.train_set.get_batch(batch_indices, labels_important=True)
        ]
        model.train()

        optimizer.zero_grad()
        output = model(sentences)["last_preds"]
        # output in shape [batch_size, length_of_sentence, num_tags (193)]

        # output = pack_padded_sequence(output, lengths.cpu(), batch_first=True).data
        # targets = pack_padded_sequence(targets, lengths.cpu(), batch_first=True).data

        loss = criterion(output, targets, self_supervision_mask)
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
            logging.info(
                "| epoch {:2d}/{:2d} | batch {:5d}/{:5d} | elapsed time {:s} | remaining time {:s} | "
                "lr {:4.2e} | train_loss {:5.3f} |".format(
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
def evaluate(model, data_sampler, dataset, helper, criterion, device):
    model.eval()
    total_loss = 0
    count = 0
    tp_total = 0
    tp_fp_total = 0
    tp_fn_total = 0
    with torch.no_grad():
        for batch_indices in data_sampler:

            sentences, targets, lengths, _ = [
                a.to(device)
                for a in dataset.get_batch(batch_indices, labels_important=True)
            ]  # Labels important here?

            output = model(sentences)["last_preds"]
            tp, tp_fp, tp_fn = helper.measure(output, targets, lengths)

            tp_total += tp
            tp_fp_total += tp_fp
            tp_fn_total += tp_fn

            loss = criterion(output, targets, 1)
            total_loss += loss.item()

            count += len(targets)
    if tp_fp_total == 0:
        tp_fp_total = 1
    if tp_fn_total == 0:
        tp_fn_total = 1
    if count == 0:
        count = 1
    return (
        total_loss / count,
        tp_total / tp_fp_total,
        tp_total / tp_fn_total,
        2 * tp_total / (tp_fp_total + tp_fn_total),
    )


def train_full(
    model, device, agent, helper, val_set, val_data_groups, original_lr, criterion, args
):
    lr = args.lr
    early_stopper = EarlyStopper(patience=args.earlystopping, model=model)

    all_val_loss = []
    all_val_precision = []
    all_val_recall = []
    all_val_f1 = []

    all_train_loss = []
    all_train_precision = []
    all_train_recall = []
    all_train_f1 = []

    start_time = time.time()

    num_sentences = agent.num_instances()
    num_words = agent.budget_spent()

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=original_lr)

    logging.info(
        f"Starting training with {num_words} words labelled in {num_sentences} sentences"
    )
    for epoch in range(1, args.epochs + 1):

        train_epoch(model, device, agent, start_time, epoch, optimizer, criterion, args)

        logging.info("beginning evaluation")
        val_loss, val_precision, val_recall, val_f1 = evaluate(
            model,
            GroupBatchRandomSampler(val_data_groups, args.batch_size, drop_last=False),
            val_set,
            helper,
            criterion,
            device,
        )
        train_loss, train_precision, train_recall, train_f1 = evaluate(
            model, agent.labelled_set, agent.train_set, helper, criterion, device
        )

        elapsed = time.time() - start_time
        logging.info(
            "| epoch {:2d} | elapsed time {:s} | train_loss {:5.4f} | val_loss {:5.4f} | prec {:5.4f} "
            "| rec {:5.4f} | f1 {:5.4f} |".format(
                epoch,
                time_display(elapsed),
                train_loss,
                val_loss,
                val_precision,
                val_recall,
                val_f1,
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

        if early_stopper.is_overfitting(val_loss):
            break

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


# TODO: Make this properly, maybe a config file in the dataset path
def get_measure_type(path):
    if "NYT_CoType" in path:
        return "relations"
    elif "OntoNotes-5.0" in path or "conll" in path:
        return "entities"
    else:
        raise NotImplementedError(path)


def initialise_model(word_embeddings, args, helper, device, model_class, loss_type):

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
    criterion = loss_type(weight)

    model = model_class(
        charset_size=len(helper.charset),
        char_embedding_size=args.emsize,
        char_channels=char_channels,
        char_padding_idx=helper.charset["<pad>"],
        char_kernel_size=args.char_kernel_size,
        char_set=helper.charset,
        vocab=helper.vocab,
        weight=word_embeddings,
        word_embedding_size=word_embedding_size,
        word_channels=word_channels,
        word_kernel_size=args.word_kernel_size,
        num_tag=len(helper.tag_set),
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        T=args.temperature,
    )._to(device)

    return model, criterion
