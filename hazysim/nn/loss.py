import torch.nn.functional as F
import torch
from .base import Base
from .functional import *
from .interpolator import Interpolator

class _Loss(Base):
    def __init__(self, size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(_WeightedLoss, self).__init__(size_average, reduce)
        self.register_buffer('weight', weight)

class CrossEntropyLoss(_WeightedLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 2` for the `K`-dimensional case (described later).

    This criterion expects a class index (0 to `C-1`) as the
    `target` for each value of a 1D tensor of size `minibatch`

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the `weight` argument being specified:

    .. math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch.

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
           If given, has to be a Tensor of size `C`
        size_average (bool, optional): By default, the losses are averaged
           over each loss element in the batch. Note that for some losses, there
           multiple elements per sample. If the field size_average is set to
           ``False``, the losses are instead summed for each minibatch. Ignored
           when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When `size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on `size_average`. When reduce
            is ``False``, returns a loss per batch instead and ignores
            size_average. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
            :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`
            in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
            :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 2` in the case of
            K-dimensional loss.
        - Output: scalar. If reduce is ``False``, then the same size
            as the target: :math:`(N)`, or
            :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 2` in the case
            of K-dimensional loss.

    Examples::

        >>> loss = nn.CrossEntropyLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce)
        self.ignore_index = ignore_index
        self.register_precision()

    def forward(self, input, target):
        torch.nn.modules.loss._assert_no_grad(target)
        input.quantize_(self.n_exponent_bits, self.n_mantissa_bits)

        return F.cross_entropy(input, target, self.weight, self.size_average,
                               self.ignore_index, self.reduce)

class ICrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(ICrossEntropyLoss, self).__init__(weight, size_average, reduce)
        
        self.iexp = Interpolator(torch.exp)
        self.iexp.chunk(min = -100, max = 70, num_points = 100)
        #self.iexp = torch.exp

        self.ilog = Interpolator(torch.log, kind="linear")
        self.ilog.chunk(min = 1e-30, max = 0.9, num_points = 1e6)
        #self.ilog = torch.log

        self.ignore_index = ignore_index
        self.register_precision()

    def forward(self, input, target):
        torch.nn.modules.loss._assert_no_grad(target)
        assert(len(input.size()) == 2)
        assert(len(target.size()) == 1)

        """ 
        Specialized 2d Cross Entropy loss that only works on 
        dimension 1. TODO: Replace torch.log and torch.exp with
        interpolated versions of each function.
        """
        dimension = 1
        dimensions = input.size()
        softmax = input.clone()

        for i in range(dimensions[0]):
            exp = self.iexp(input[i, :])
            softmax[i, :] = (exp/exp.sum())

        assert(torch.isnan(softmax).sum() == 0)
        logsm = -self.ilog(softmax)
 
        assert(torch.isnan(logsm).sum() == 0)

        loss = 0
        for i in range(dimensions[0]):
            loss += logsm[i, target[i]]
        loss = loss / dimensions[0]

        return loss 
