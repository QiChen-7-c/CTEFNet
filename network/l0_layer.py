import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class L0Dense(Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""
    def __init__(self, channels, height, width, droprate_init=0.5, temperature=2./3., lamba=1., local_rep=False, **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution  # beta
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Dense, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.size = height*width*channels
        # print(torch.Tensor(input_shape).shape)
        self.qz_loga = Parameter(torch.Tensor(channels, height, width), requires_grad=True)  # log alpha
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.local_rep = local_rep

        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

    def constrain_parameters(self, **kwargs):
        # 限制 qz_log a 在 0.01~100
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw = torch.sum((1 - self.cdf_qz(0)) * self.lamba)
        return logpw

    def regularization(self):
        return self._reg_w()/self.size

    def get_mask(self):
        z = F.sigmoid(self.qz_loga)
        z = z * (limit_b - limit_a) + limit_a
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.size
        expected_l0 = ppos * self.size
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def forward(self, input):
        if self.training:
            if self.local_rep:
                eps = self.get_eps(self.floatTensor(input.size(0), self.channels, self.height, self.width))
                z = self.quantile_concrete(eps)
                mask = F.hardtanh(z, min_val=0, max_val=1)
            else:
                eps = self.get_eps(self.floatTensor(self.channels, self.height, self.width))
                z = self.quantile_concrete(eps).expand(input.size(0), self.channels, self.height, self.width)
                mask = F.hardtanh(z, min_val=0, max_val=1)
        else:
            pi = F.sigmoid(self.qz_loga).view(1, self.channels, self.height, self.width).expand(input.size(0),
                                                                                                self.channels,
                                                                                                self.height, self.width)
            mask = F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
        return input * mask

    def __repr__(self):
        s = ('{name}({channels}, {height}, {width}, droprate_init={droprate_init}, '
             'lamba={lamba}, temperature={temperature})')
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)



if __name__ == '__main__':
    layer = L0Dense(3, 120, 180,).cuda()
    input = torch.rand(8, 3, 120, 180).cuda()
    res = layer(input)
    print(res.shape)
    print(layer.get_mask())

    import matplotlib.pyplot as plt

    weight = layer.get_mask().detach().cpu().numpy().mean(axis=0)
    plt.imshow(weight, cmap='OrRd',interpolation='bicubic', vmin=0, vmax=1)
    plt.title('regularization weight')
    plt.colorbar()
    plt.show()