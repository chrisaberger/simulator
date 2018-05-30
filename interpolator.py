import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Interpolator(torch.autograd.Function):
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

    def interp1d(self, xin):
        """
        Stores the static grid of points ('xin') we will use to perform 
        interpolation.
        """
        self.xin, indices = torch.sort(xin)
        self.yin = self.__eval_forward(self.xin)        

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

    def adapt(self, start, end, delta, hmin):
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

        # Backwards
        x_grad = []
        y_grad = []
        self.__recurse_adapt(start, end, delta, hmin, x_grad, y_grad, 
                             self.__eval_backward)
        x.append(end) # Stitch the end value on.
        y.append(self.__eval_backward(end)) # Stitch the end value on.
        self.xin_grad = torch.tensor(x_grad, dtype=torch.float64)
        self.yin_grad = torch.tensor(y_grad, dtype=torch.float64)

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

    def __linear(self, xout, x0, x1, y0, y1):
        """
        Calls the point slope formula to return a y value on the line connecting
        the points [(x0, y0), (x1, y1)].
        """
        m = (y1 - y0) /  (x1 - x0) # Calculate slope.
        return (xout - x0) * m + y0 # Point slope formula.

    def __get_points(self, xout, xin, yin):
        """
        Given a x value 'xout' to predict, this method finds the coordinates of 
        the two nearest points [(x0, y0), (x1, y1)] we have in our grid of 
        'self.xin' and 'self.yin' values
        """
        # Change this method to accept a list of points. 
        lenxin = len(xin)

        # Search for the nearest index to 'xout'. This is the larger index
        # from which we will grid a line between two points (e.g. larger point).
        i1 = np.searchsorted(xin, xout)

        i1 = 1 if i1 == 0 else i1 # Corner case for start of array.
        i1 = lenxin-1 if i1 == lenxin else i1 # Corner case for end of array.

        x0 = xin[i1-1]
        x1 = xin[i1]
        y0 = yin[i1-1]
        y1 = yin[i1]
        return x0, x1, y0, y1

    def __get_single_point(self, xout, backwards=False):
        """
        Interprets a single y value for a given x value ('xout').
        """
        x0, x1, y0, y1 = self.__get_points(xout, self.xin, self.yin)
        if backwards:
            x0, x1, y0, y1 = self.__get_points(xout, self.xin_grad, self.yin_grad)
        if self.kind == 'nearest':
            return self.__nearest(xout, x0, x1, y0, y1)
        elif self.kind == 'linear':
            return self.__linear(xout, x0, x1, y0, y1)
        else:
            raise ValueError('Invalid interpolation kind: %s' % self.kind)

    @staticmethod
    def forward(ctx, input, self):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward 
        method.
        """
        #ctx.save_for_backward(input)

        inp_clone = input.clone().detach()
        inp_clone = torch.tensor(inp_clone.map_(inp_clone, 
                        lambda a,b: self.__get_single_point(a)),
                        requires_grad=True)
 
        # TODO: Can we fix this? Right now I basically have to do backwards 
        # in the forwards because I cannot access 'self' in the backwards pass.
        # 'save_for_backward' can only store tensors.
        grad_clone = input.clone().detach()
        grad_clone = torch.tensor(grad_clone.map_(grad_clone, 
                        lambda a,b: self.__get_single_point(a, backwards=True)))
        
        ctx.save_for_backward(input, grad_clone)

        return inp_clone

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the 
        loss with respect to the output, and we need to compute the gradient of 
        the loss with respect to the input.
        """
        # TODO: Is it right to just do 'grad_input' * 'grad_output'. I think
        # so...check up on yo chain rule.
        input, grad_input = ctx.saved_tensors
        return (grad_input*grad_output, None)
