import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class FInterpolator(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, fn, kind='linear'):
        """
        Initializes the function we will interpolate.
        """
        self.fn = fn
        self.kind = kind
        self.forward_fn_interpolate = None
        self.backward_fn_interpolate = None   

    def __eval_forward(self, inp):
        result = self.fn(torch.tensor(inp, dtype=torch.float64))
        # Just return the python value if it is a scalar.
        if result.size() == torch.Size([]):
            result = result.numpy().item(0)
        return result

    def __eval_backward(self, inp):
        dummy_in = torch.tensor(inp, dtype=torch.float64, requires_grad=True)
        dummy_out = self.fn(dummy_in)
        # Backwards must be called on a scalar. So if our input/output is not
        # a scalar sum the tensor before calling backward (will force grads of
        # 1 during evaluation).
        if dummy_in.size() != torch.Size([]):
            dummy_out = dummy_out.sum()
        dummy_out.backward()
        result = dummy_in.grad
        # Just return the python value if it is a scalar.
        if result.size() == torch.Size([]):
            result = result.numpy().item(0)
        return result 

    def __recurse_adapt(self, xL, xR, delta, hmin, x, y, fn):
        if xR - xL <= hmin:
            x.append(xL)
            y.append(fn(xL))
        else:
            mid = (xL + xR) / 2
            fmid = fn(mid)
            fL = fn(xL)
            fR = fn(xR)
            if abs( ( (fL + fR) / 2) - fmid) <= delta:
                # Subinterval accepted
                x.append(xL)
                y.append(fn(xL))
            else:
                self.__recurse_adapt(xL, mid, delta, hmin, x, y, fn)
                self.__recurse_adapt(mid, xR, delta, hmin, x, y, fn)

    def naive_chunking(self, min, max, num_points):
        assert(max > min)
        assert(num_points > 0)
        diff = max - min
        chunk_size = diff/num_points

        self.xin = torch.arange(float(min), 
                                 float(max), 
                                 step=chunk_size)
        self.yin = self.__eval_forward(self.xin)
        self.forward_fn_interpolate = interpolate.interp1d(self.xin, 
                                                           self.yin, 
                                                           kind=self.kind)


        self.xin_grad = self.xin
        self.yin_grad = self.__eval_backward(self.xin_grad)
        self.backward_fn_interpolate = interpolate.interp1d(self.xin_grad, 
                                                             self.yin_grad, 
                                                             kind=self.kind)
        print("done")


    def adapt_linear(self, start, end, delta, hmin):
        """
        Adaptive Piecewise Linear Interpolation from Section 3.1.4. 
        http://www.cs.cornell.edu/courses/cs4210/2014fa/CVLBook/CVL3.PDF
        """

        # Forwards
        x = []
        y = []
        self.__recurse_adapt(start, end, delta, hmin, x, y, 
                             self.__eval_forward)
        x.append(end) # Stitch the end value on.
        y.append(self.__eval_forward(end)) # Stitch the end value on.
        self.xin = torch.tensor(x, dtype=torch.float64)
        self.yin = torch.tensor(y, dtype=torch.float64)
        self.forward_fn_interpolate = interpolate.interp1d(self.xin, 
                                                           self.yin, 
                                                           kind=self.kind)

        # Backwards
        x_grad = []
        y_grad = []
        self.__recurse_adapt(start, end, delta, hmin, x_grad, y_grad, 
                             self.__eval_backward)
        x_grad.append(end) # Stitch the end value on.
        y_grad.append(self.__eval_backward(end)) # Stitch the end value on.
        self.xin_grad = torch.tensor(x_grad, dtype=torch.float64)
        self.yin_grad = torch.tensor(y_grad, dtype=torch.float64)
        self.backward_fn_interpolate = interpolate.interp1d(self.xin_grad, 
                                                             self.yin_grad, 
                                                             kind=self.kind)

    def fit_naive(self, start, end, num_points):
        print("fitting naive")

    def __plot_vals(self, xin, yin, fn):
        minval = xin.numpy()[0]
        maxval = xin.numpy()[len(xin)-1]

        # Create a plot for the 'actual' function by mapping small 
        # granularities.
        xnew = torch.from_numpy(np.arange(minval, maxval, 1/3600))
        ynew = fn(xnew)

        print("Memory Required: " + str(len(xin)*2*8) + " bytes")

        plt.plot(xin.numpy(), yin.numpy(), 'o',
                 xnew.numpy(), ynew.numpy(), '-')
        plt.show()

    def plot(self):
        """
        Plot the forwards interpolation first, then the backwards interpolation.
        """   
        # Forwards plot
        self.__plot_vals(self.xin, self.yin, self.__eval_forward)
        # Backwards plot
        self.__plot_vals(self.xin_grad, self.yin_grad, self.__eval_backward)

    def __nearest(self, xout, x0, x1, y0, y1):
        """
        Simply finds and returns the y value ('y0' or 'y1') whose associated x 
        value ('x0' or 'x1') is closer to 'xout'. For x values that are equally 
        distant this method rounds up.  
        """
        return np.where(np.abs(xout - x0) < np.abs(xout - x1), y0, y1)

    @staticmethod
    def forward(ctx, input, self):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward 
        method.
        """
        inp_clone = input.clone().detach()
        forward = self.forward_fn_interpolate(inp_clone.numpy())
        forward = torch.tensor(forward)

        ctx.self = self
        ctx.save_for_backward(inp_clone)

        return forward

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the 
        loss with respect to the output, and we need to compute the gradient of 
        the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = ctx.self.backward_fn_interpolate(input.numpy())
        grad_input = torch.tensor(grad_input)
        return (grad_input*grad_output, None)
