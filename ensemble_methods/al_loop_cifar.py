import sys
import os
import numpy as np
from .densenet import SelfDirDenseNet, DenseNet, EvalDenseNet
from torchvision import transforms
from .training_utils import *
import argparse

sys.path.append(os.path.dirname(os.getcwd()))
import active_learning as al

parser = argparse.ArgumentParser(description = 'Active Learning Loop for CIFAR 100')

parser.add_argument('--model_type', choices=['densenet'])
parser.add_argument('--log_dir', type=str)
parser.add_argument('--augment', default=1, type = int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--retrain', default=0, type=int)



def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_data(data_path):
    train_images = np.vstack(
        [
            unpickle(os.path.join(data_path, f"data_batch_{i}"))[b"data"]
            for i in range(1, 6)
        ]
    ).reshape(-1, 3, 32, 32)
    eval_images = unpickle(os.path.join(data_path, "test_batch"))[b"data"].reshape(
        -1, 3, 32, 32
    )
    train_labels = []
    for i in range(1, 6):
        train_labels.extend(
            unpickle(os.path.join(data_path, f"data_batch_{i}"))[b"labels"]
        )
    eval_labels = unpickle(os.path.join(data_path, "test_batch"))[b"labels"]
    return train_images, train_labels, eval_images, eval_labels


if __name__ == "__main__":

    args = parser.parse_args()
    data_dir = "/home/alta/BLTSpeaking/exp-pr450/cifar-10"
    train_images, train_labels, eval_images, eval_labels = get_data(data_dir)

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dropout_rate = 0
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dropout_rate = 0.2

    if args.model_type == 'densenet':
        model = DenseNet(depth = 100, growthRate=12, dropRate=dropout_rate, compressionRate=2, num_classes=100)
        al_attr = []

    elif args.model_type == 'selfdir':
        model = SelfDirDenseNet()
        al_attr = [
            al.util_classes.ALAttribute('log_alphas', [None for _ in range(train_images)]),
            al.util_classes.StochasticAttribute('teacher_branch_prediction', [None for _ in range(train_images)], M=5),
            al.util_classes.StochasticAttribute('student_branch_target', [None for _ in range(train_images)], M=5),
        ]

    elif args.model_type == 'eval':
        model = EvalDenseNet()
        al_attr = [al.util_classes.ALAttribute('log_alphas', [None for _ in range(train_images)])]

    else:
        raise ValueError(args.model_type)

    train_set = al.util_classes.ImageClassificationDataset(
        data=train_images,
        labels=train_labels,
        index_class=al.util_classes.DimensionlessIndex,
        al_attributes=al_attr,
        semi_supervision_multiplier=0,
    )
    val_set = al.util_classes.ImageClassificationDataset(eval_images, eval_labels, al.util_classes.DimensionlessIndex, 0)

    selector = al.selector.DimensionlessSelector(
        round_size=500,
        acquisition=al.acquisition.MaximumEntropyAcquisition(train_set),
        window_class=al.util_classes.DimensionlessAnnotationUnit,
        diversity_policy=al.batch_querying.NoPolicyBatchQuerying()
    )

    agent = al.agent.ActiveLearningAgent(train_set, 128, selector, model, 'cuda', 1.0)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)d
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = al.util_classes.EarlyStopper(model, 5)

    train_full(model, agent, optimiser, scheduler, criterion, early_stopper, transform_train, transform_test, val_set,
               args)

