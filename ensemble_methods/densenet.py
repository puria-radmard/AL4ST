from __future__ import print_function

import sys, os

sys.path.insert(0, "/home/alta/sequence_ensemble_distillation/local-code/")
sys.path.insert(0, "/home/alta/sequence_ensemble_distillation/local-code/seq")
sys.path.insert(0, "/home/alta/sequence_ensemble_distillation/local-code/seq/cvision")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.autograd import Variable

import numpy as np
import math

sys.path.append(os.path.dirname(os.getcwd()))
from dropout import GaussianDropoutLayer

__all__ = [
    "densenet",
    "evaldensenet",
    "selfdirdensenet",
]


def check_device(use_gpu):
    """
    Return device, either cpu or gpu
    """
    available_gpu = use_gpu and torch.cuda.is_available()
    return torch.device("cuda") if available_gpu else torch.device("cpu")


class Bottleneck(nn.Module):
    def __init__(
        self, inplanes, expansion=4, growthRate=12, dropRate=0, evaluation=False
    ):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate
        self.evaluation = evaluation
        self.dropUse = True

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(
                out,
                p=self.dropRate,
                training= self.training #self.dropUse if self.evaluation else self.training,
            )

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(
        self, inplanes, expansion=1, growthRate=12, dropRate=0, evaluation=False
    ):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(
            inplanes, growthRate, kernel_size=3, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate
        self.evaluation = evaluation
        self.dropUse = True

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(
                out,
                p=self.dropRate,
                training=self.dropUse if self.evaluation else self.training,
            )

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        depth=22,
        block=Bottleneck,
        dropRate=0,
        num_classes=10,
        growthRate=12,
        compressionRate=2,
    ):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, "depth should be 3n+4"
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(
                block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate)
            )
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return {
            "last_preds": F.log_softmax(x, dim=-1)
        }


class EvalDenseNet(nn.Module):
    def __init__(
        self,
        depth=22,
        block=Bottleneck,
        dropRate=0,
        num_classes=10,
        growthRate=12,
        compressionRate=2,
    ):
        super(EvalDenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, "depth should be 3n+4"
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate
        self.dropUse = True

        # self.inplanes is a global variable used across multiple helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def activateDrop(self):
        for layer in self.dense1.children():
            layer.dropUse = True
        for layer in self.dense2.children():
            layer.dropUse = True
        for layer in self.dense3.children():
            layer.dropUse = True

    def deactivateDrop(self):
        for layer in self.dense1.children():
            layer.dropUse = False
        for layer in self.dense2.children():
            layer.dropUse = False
        for layer in self.dense3.children():
            layer.dropUse = False

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(
                block(
                    self.inplanes,
                    growthRate=self.growthRate,
                    dropRate=self.dropRate,
                    evaluation=True,
                )
            )
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return {
            'last_preds': F.log_softmax(x, dim=-1),
            'log_alphas': x
        }


class SelfDirDenseNet(nn.Module):
    def __init__(
        self,
        depth=22,
        block=Bottleneck,
        dropRate=0,
        num_classes=10,
        growthRate=12,
        compressionRate=2,
        dropout_a=0.10,
        dropout_b=0.70,
        temperature=1.0,
        niter=10,
    ):
        super(SelfDirDenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, "depth should be 3n+4"
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Temperature for second branch
        self.temperature = temperature
        self.niter = niter

        self.dp = GaussianDropoutLayer(
            dropout_a, dropout_b, method="uniform", use_gpu=True
        )
        self.device = check_device(True)

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(
                block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate)
            )
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward(self, x, passes=5):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Get dropout predictions
        dp_x = [self.dp(x).to(self.device) for _ in range(passes)]
        dp_x = [self.fc(q) for q in dp_x]
        dp_t = [q.clone().detach() / self.temperature for q in dp_x]

        # Get standard prediction
        x = self.fc(x)

        return {
            'last_preds': None,
            'log_alphas': x,
            'teacher_branch_prediction': dp_x, # M logits of size b*k
            'student_branch_target': dp_t # M logits of size b*k
        }

    @staticmethod
    def calc_expectation_of_entropy(log_alphas: torch.Tensor):
        """
        Calculates per token expectation of entropy.
        This can be seen as data uncertainty.
        """
        # Dimension b x len x dim
        alphas = torch.exp(log_alphas)

        # Dimension b x len
        alpha0 = torch.sum(alphas, dim=-1)

        expectation_of_entropy = torch.digamma(alpha0 + 1)
        expectation_of_entropy -= (
            torch.sum(alphas * torch.digamma(alphas + 1), dim=-1) / alpha0
        )

        # Dimension b
        return expectation_of_entropy

    @staticmethod
    def calc_entropy_of_expectation(log_alphas: torch.Tensor):
        """
        Calculates per token entropy of expectation.
        This can be seen as total uncertainty.
        """
        # Dimension b x dim
        expectation_of_prob = torch.softmax(log_alphas, dim=-1)

        # Dimension b
        entropy_of_expectation = Categorical(probs=expectation_of_prob).entropy()

        return entropy_of_expectation

    @staticmethod
    def calc_expectation_of_entropy_ens(log_probs: torch.Tensor):
        """
        Calculates per token expectation of entropy.
        This can be seen as data uncertainty.
        """
        # Dimension b x N
        entropy = Categorical(probs=torch.exp(log_probs)).entropy()

        # Dimension b
        expectation_of_entropy = torch.mean(entropy, dim=-1)

        return expectation_of_entropy.type(torch.FloatTensor)

    @staticmethod
    def calc_entropy_of_expectation_ens(log_probs: torch.Tensor):
        """
        Calculates per token entropy of expectation.
        This can be seen as total uncertainty.
        """
        # Dimension b x dim
        expectation_of_prob = torch.mean(torch.exp(log_probs), dim=1)

        # Dimension b
        entropy_of_expectation = Categorical(probs=expectation_of_prob).entropy()

        return entropy_of_expectation

    @staticmethod
    def estimate_dirichlet(log_probs, niter=10, enforce_soft_constraint=True):
        def init(log_p, temperature=1.0):
            """
            Initialises the mean and scale of dirichlet.
            log_p: log probabilites of each model
                dimension: batch x len x n_models x vocab
            """
            b, s, m, v = log_p.size()

            log_expected_prob = torch.logsumexp(log_p, dim=2) - np.log(m)
            log_expected_sq_prob = torch.logsumexp(2 * log_p, dim=2) - np.log(m)
            log_expected_prob_sq = 2 * log_expected_prob

            div = torch.exp(log_expected_prob - log_expected_sq_prob) - 1
            div = div / (
                1 - torch.exp(log_expected_prob_sq - log_expected_sq_prob) + 1e-3
            )
            div = torch.log(div[:, :, :-1] + 1e-3)

            alpha0 = torch.mean(div, dim=-1, keepdim=True)
            alpha0 = torch.exp(alpha0 / temperature)

            info = {"log_expected_prob": log_expected_prob}
            return torch.exp(log_expected_prob), alpha0, info

        def step(expected_log_p, p, s):
            """
            Performs an update step of the scale
            """
            v = p.size(-1)

            new_s = torch.digamma(s * p) - expected_log_p
            new_s = (new_s * p).sum(dim=-1, keepdim=True)
            new_s += (v - 1) / (s + 1e-6) - torch.digamma(
                s + 1e-6
            )  # Adding errors to ensure it works
            new_s = (v - 1) / (new_s + 2e-6)  # Adding errors to ensure it works
            return F.softplus(new_s, beta=5.0)

        # Initialise
        probs, scale, info = init(log_probs)

        # Geometric mean needed for updates
        expected_log_probs = log_probs.mean(dim=2)

        for i in range(niter):
            scale = step(expected_log_probs, probs, scale)

        # Remove all nans
        mask = torch.isnan(scale)
        scale[mask] = scale[~mask].mean()

        # Ensure alphas are larger than 1
        if enforce_soft_constraint:
            log_alphas_est = torch.log(torch.exp(info["log_expected_prob"]) * scale + 1)
            return log_alphas_est.clone().detach()

        log_alphas_est = F.softplus(
            info["log_expected_prob"] + torch.log(scale), beta=5.0
        )
        return log_alphas_est.clone().detach()

    def compute_loss(
        self,
        log_alphas,
        log_probs,
        loss_type="standard",
        return_target=False,
        reduction="sum",
    ):
        """
        Estimates a dirichlet and computes the kl-divergence between the predicted and estimated.
        log_alphas: b x vocab
        log_probs: [b x vocab] x models
        """

        with torch.no_grad():
            log_alphas_est = self.estimate_dirichlet(
                torch.stack(log_probs, dim=1).unsqueeze(1), niter=self.niter
            ).squeeze(1)

        alphas_est = torch.exp(log_alphas_est)
        target = Dirichlet(alphas_est)

        alphas = torch.exp(log_alphas)
        predic = Dirichlet(alphas)

        # Dimension b x len
        if loss_type == "reversekl":
            # Using reverse KL for possibly more stable criteria
            loss = torch.distributions.kl.kl_divergence(predic, target)
        else:
            loss = torch.distributions.kl.kl_divergence(target, predic)

        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()

        if return_target:
            return loss, log_alphas_est
        return loss



def selfdirdensenet(**kwargs):
    """
    Constructs a Self Dir DenseNet model.
    """
    return SelfDirDenseNet(**kwargs)


def evaldensenet(**kwargs):
    """
    Constructs a DenseNet model.
    """
    return EvalDenseNet(**kwargs)


def densenet(**kwargs):
    """
    Constructs a DenseNet model.
    """
    return DenseNet(**kwargs)
