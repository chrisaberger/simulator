import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class Interpolator:
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
        result = self.fn(inp)
        # Just return the python value if it is a scalar.
        #if result.size() == torch.Size([]):
        #    result = result.numpy().item(0)
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

    def adapt_linear(self, start, end, delta, hmin):
        """
        Adaptive Piecewise Linear Interpolation from Section 3.1.4. 
        http://www.cs.cornell.edu/courses/cs4210/2014fa/CVLBook/CVL3.PDF
        """

        fill_value = self.fn(np.array([float(start), float(end)]))
        fill_value = (fill_value[0],fill_value[1])

        # Forwards
        x = []
        y = []
        self.__recurse_adapt(start, end, delta, hmin, x, y, 
                             self.__eval_forward)
        x.append(end) # Stitch the end value on.
        y.append(self.__eval_forward(end)) # Stitch the end value on.
        self.xin = np.array(x, dtype=np.float32)
        self.yin = np.array(y, dtype=np.float32)
        self.forward_fn_interpolate = interpolate.interp1d(self.xin, 
                                                           self.yin, 
                                                           kind=self.kind,
                                                           bounds_error=False,
                                                           fill_value=fill_value)
        print("Memory Required: " + str(len(self.xin)*2*8) + " bytes")

    def chunk(self, min, max, num_points):
        assert(max > min)
        assert(num_points > 0)
        diff = max - min
        chunk_size = diff/num_points

        fill_value = self.fn(np.array([float(min), float(max)]))
        fill_value = (fill_value[0],fill_value[1])
        self.xin = np.arange(float(min), 
                                 float(max), 
                                 step=chunk_size)
        self.yin = self.__eval_forward(self.xin)
        self.forward_fn_interpolate = interpolate.interp1d(self.xin, 
                                                           self.yin, 
                                                           kind=self.kind,
                                                           fill_value=fill_value,
                                                           bounds_error=False)
        print("Memory Required: " + str(len(self.xin)*2*8) + " bytes")


    def __plot_vals(self, xin, yin, fn):
        minval = xin[0]
        maxval = xin[len(xin)-1]

        # Create a plot for the 'actual' function by mapping small 
        # granularities.
        xnew = np.arange(minval, maxval, 1/3600)
        ynew = fn(xnew)

        print("Memory Required: " + str(len(xin)*2*8) + " bytes")

        plt.plot(xin, yin, 'o',
                 xnew, ynew, '-')
        plt.show()

    def plot(self):
        """
        Plot the forwards interpolation first, then the backwards interpolation.
        """   
        # Forwards plot
        self.__plot_vals(self.xin, self.yin, self.__eval_forward)

    def forward(self, inp):
        return self.forward_fn_interpolate(inp)
