import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta
from torch.distributions.gamma import Gamma


def check_device(use_gpu):
    """
    Return device, either cpu or gpu
    """
    available_gpu = use_gpu and torch.cuda.is_available()
    return torch.device("cuda") if available_gpu else torch.device("cpu")


#
# Bernoulli Dropout based layers
#


class SamplingDropoutLayer(nn.Module):
    def __init__(self, a, b, method="delta", use_gpu=True):
        super(SamplingDropoutLayer, self).__init__()
        device = check_device(use_gpu)

        if method == "delta":
            self.dp = DeltaDropoutLayer(device, a, b)
        elif method == "uniform":
            self.dp = UniformDropoutLayer(device, a, b)
        elif method == "beta" or method == "gamma":
            self.dp = BetaDropoutLayer(device, a, b)
        else:
            msg = "The specified method {} is not implemented. \n "
            msg += "PS Make sure method is lowercase."
            msg = msg.format(method)
            raise ValueError(msg)

        self.dp = self.dp.to(device=check_device(use_gpu))

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        return self.dp(x)


class SamplingDropoutClass(nn.Module):
    def __init__(self, device, *args):
        super(SamplingDropoutClass, self).__init__()
        self.sampler = None
        self.device = device

    def __sample_p(self):
        return self.sampler.sample().item()

    def __call__(self, x: torch.Tensor, level=True, *args, **kwargs):
        if level:
            y = torch.zeros(x.size()).to(self.device)
            for i, xx in enumerate(x):
                y[i] = F.dropout(xx, p=self.__sample_p(), training=self.training).to(
                    self.device
                )
            return y

        return F.dropout(x, p=self.__sample_p(), training=self.training).to(self.device)


class DeltaDropoutLayer(SamplingDropoutClass):
    def __init__(self, device, a, b=None):
        super(DeltaDropoutLayer, self).__init__(device)
        self.dropout = nn.Dropout(p=a).to(device)

    def __call__(self, x: torch.Tensor, level=True, *args, **kwargs):
        return self.dropout(x).to(self.device)


class UniformDropoutLayer(SamplingDropoutClass):
    def __init__(self, device, a, b):
        super(UniformDropoutLayer, self).__init__(device)
        self.sampler = Uniform(torch.tensor([a]), torch.tensor([b]))


class BetaDropoutLayer(SamplingDropoutClass):
    def __init__(self, device, a, b):
        super(BetaDropoutLayer, self).__init__(device)
        self.sampler = Beta(torch.tensor([a]), torch.tensor([b]))


#
# Gaussian Dropout based layers
#


class GaussianDropoutClass(nn.Module):
    def __init__(self, device, *args):
        super(GaussianDropoutClass, self).__init__()
        self.sampler = None
        self.device = device

    def __sample_p(self):
        return self.sampler.sample().item()

    def generate_noise(self, x: torch.Tensor, level=True):
        if level:
            y = torch.zeros(x.size()).to(self.device)
            for i, xx in enumerate(x):
                s = self.__sample_p()
                y[i] = torch.randn(xx.size()).to(self.device) * s + 1
            return y

        # Samples a standard deviation of gaussian
        s = self.__sample_p()

        # Gaussian noise with mean 1 and stddev s
        return torch.randn(x.size()).to(self.device) * s + 1

    def __call__(self, x: torch.Tensor, level=True, *args, **kwargs):
        noise = self.generate_noise(x, level=True)
        return x * noise


class DeltaGaussianLayer(GaussianDropoutClass):
    def __init__(self, device, a, b=None):
        super(DeltaGaussianLayer, self).__init__(device)
        self.s = a

    def generate_noise(self, x: torch.Tensor, level=True):
        # Gaussian noise with mean 1 and stddev s
        return torch.randn(x.size()).to(self.device) * self.s + 1


class UniformGaussianLayer(GaussianDropoutClass):
    def __init__(self, device, a, b):
        super(UniformGaussianLayer, self).__init__(device)
        self.sampler = Uniform(torch.tensor([a]), torch.tensor([b]))


class GammaGaussianLayer(GaussianDropoutClass):
    def __init__(self, device, a, b):
        super(GammaGaussianLayer, self).__init__(device)
        self.sampler = Gamma(torch.tensor([a]), torch.tensor([b]))


class GaussianDropoutLayer(nn.Module):
    def __init__(self, a, b, method="delta", use_gpu=True):
        device = check_device(use_gpu)

        super(GaussianDropoutLayer, self).__init__()
        if method == "delta":
            self.dp = DeltaGaussianLayer(device, a, b)
        elif method == "uniform":
            self.dp = UniformGaussianLayer(device, a, b)
        elif method == "beta" or method == "gamma":
            self.dp = GammaGaussianLayer(device, a, b)
        else:
            msg = "The specified method {} is not implemented. \n "
            msg += "PS Make sure method is lowercase."
            msg = msg.format(method)
            raise ValueError(msg)

        self.dp = self.dp.to(device=check_device(use_gpu))

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        return self.dp(x)

