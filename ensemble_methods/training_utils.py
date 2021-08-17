import torch
import random

def train_epoch(model, agent, optimiser, criterion, transform_train):
    model.train()
    total_loss = 0
    num_instances = 0
    sampler = sorted(agent.labelled_set, key=lambda k: random.random())
    for batch_indices in sampler:
        inputs, targets, _, self_supervision_mask = agent.train_set.get_batch(batch_indices, True)

        readable_images = np.stack([inputs[:, 0, :, :], inputs[:, 1, :, :], inputs[:, 2, :, :]], axis=-1)
        inputs = torch.stack(
            [transform_train(Image.fromarray(im)) for im in readable_images]
        ).to('cuda')

        optimiser.zero_grad()
        output = model(inputs.float().to('cuda'), anneal=False)['last_preds']
        loss = criterion(output, targets.to('cuda'))
        loss.backward()
        total_loss += loss.item()
        optimiser.step()
        num_instances += len(batch_indices)
    return total_loss / num_instances


def val_epoch(model, val_set, criterion, transform_test):
    model.eval()
    num_instances = 0
    num_correct = 0
    total_loss = 0
    for i in range((len(val_set) // 128)):
        batch_indices = range(i * 128, (i + 1) * 128)
        inputs = val_set.data.attr[batch_indices]

        readable_images = np.stack([inputs[:, 0, :, :], inputs[:, 1, :, :], inputs[:, 2, :, :]], axis=-1)
        inputs = torch.stack(
            [transform_test(Image.fromarray(im)) for im in readable_images]
        ).to('cuda')

        targets = np.array([int(val_set.labels.attr[b]) for b in batch_indices])
        num_instances += len(batch_indices)
        output = model(inputs.float().to('cuda'), anneal=False)['last_preds']
        total_loss += criterion(output, torch.tensor(targets).to('cuda')).detach().cpu().item()
        preds = output.argmax(dim=-1).cpu().numpy()
        num_correct += sum(preds == targets)
    return num_correct / num_instances, total_loss / num_instances


def log_epoch(agent, log_dir, round_num, statement, indices):
    log_file_path = os.path.join(log_dir, "model_perfomance.log")
    if os.path.exists(log_file_path):
        mode = 'a'  # append if already exists
    else:
        mode = 'w'  # make a new file if not
    with open(log_file_path, mode) as f:
        f.write(f"{statement}\n")

    indices_file_path = os.path.join(log_dir, f"indices_{round_num}.log")
    with open(indices_file_path, 'w') as f2:
        for ind in indices:
            f2.write(f"{int(ind)}\n")


def train_full(model, agent, optimiser, scheduler, criterion, early_stopper, transform_train, transform_test, val_set,
               args):
    log_dir_original = f"{args.log_dir}_reinit{args.retrain}"
    ilog = 0
    while True:
        try:
            log_dir_mod = f"{log_dir_original}-{ilog}"
            os.mkdir(log_dir_mod)
            break
        except:
            ilog += 1
        if ilog > 40:
            break
    args['log_dir'] = log_dir_mod
    print('\n')
    print(log_dir_mod)
    for k, v in args.items():
        print(k, '\t', v)

    round_number = 0
    if args['reinit_weights']:
        original_state_dict = model.state_dict()

    agent.init(3000)
    current_labelled = {k for k, v in agent.train_set.index.labelled_idx.items() if v}
    log_epoch(agent, args['log_dir'], round_number, 'Initialising', current_labelled)

    for _ in agent:
        round_number += 1
        # os.mkdir(os.path.join(args['log_dir'], f"round-{round_number}"))
        val_set_accuracy = []
        val_loss_history = []
        current_labelled = {k for k, v in agent.train_set.index.labelled_idx.items() if v}

        for e in range(args['num_epochs']):
            training_loss = train_epoch(model, agent, optimiser, criterion, transform_train)
            epoch_accuracy, val_loss = val_epoch(model, val_set, criterion, transform_test)
            # scheduler.step(val_loss)
            val_set_accuracy.append(epoch_accuracy)
            val_loss_history.append(val_loss)
            statement = "Round {} || Epoch {} || Training Loss {} || Val Loss {} || Val Accuracy {}".format(
                round_number, e, training_loss, val_loss, epoch_accuracy
            )
            log_epoch(agent, args['log_dir'], round_number, statement, current_labelled)
            print(statement, '\n')

            if early_stopper.check_stop([-a for a in val_set_accuracy]):
                print("latest eval accuracies:", [v for v in val_set_accuracy[-4:]], '\t so early stopping\n')
                break

        if args['reinit_weights']:
            model.load_state_dict(original_state_dict)