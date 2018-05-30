import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F

class HazyInterpolate:
    """
    Initializes the function we will interpolate.
    """
    def __init__(self, fn):
        self.fn = fn
        self.kind = 'linear'

    """
    Stores the grid of points we will use to perform interpolation.
    """
    def interp1d(self, xin, kind='linear'):
        self.kind = kind
        self.xin, indices = torch.sort(xin)
        self.yin = self.fn(self.xin)

    def __recurse_adapt(self, xL, xR, delta, hmin, x, y):
        # Yikes, see if you can eliminate this weird double casting to tensors.
        if xR - xL <= hmin:
            print("less than hmin")
            x.append(xL)
            #x.append(xR)
            y.append(self.fn(torch.DoubleTensor([xL])).numpy()[0])
            #y.append(self.fn(torch.DoubleTensor([xR])).numpy()[0])

        else:
            mid = (xL + xR) / 2
            fmid = self.fn(torch.DoubleTensor([mid])).numpy()[0]
            fL = self.fn(torch.DoubleTensor([xL])).numpy()[0]
            fR = self.fn(torch.DoubleTensor([xR])).numpy()[0]
            print("fl: " + str(fL) + " fR: " + str(fR) + " fmid: " + str(fmid))
            if abs( ( (fL + fR) / 2) - fmid) <= delta:
                print("subinterval accepted")
                # Subinterval accepted
                x.append(xL)
                #x.append(xR)
                y.append(self.fn(torch.DoubleTensor([xL])).numpy()[0])
                #y.append(self.fn(torch.DoubleTensor([xR])).numpy()[0])
            else:
                self.__recurse_adapt(xL, mid, delta, hmin, x, y)
                self.__recurse_adapt(mid, xR, delta, hmin, x, y)

    def adapt(self, start, end, delta, hmin):
        x = []
        y = []
        self.__recurse_adapt(start, end, delta, hmin, x, y)
        x.append(end)
        y.append(self.fn(torch.DoubleTensor([end])).numpy()[0])
        self.xin = torch.DoubleTensor(x)
        self.yin = torch.DoubleTensor(y)

    """
    Simply finds and returns the y value ('y0' or 'y1') whose associated x value 
    ('x0' or 'x1') is closer to 'xout'. For x values that are equally distant
    this method rounds up.  
    """
    def __nearest(self, xout, x0, x1, y0, y1):
        return np.where(np.abs(xout - x0) < np.abs(xout - x1), y0, y1)

    """
    Calls the point slope formula to return a y value on the line connecting
    the points [(x0, y0), (x1, y1)].
    """
    def __linear(self, xout, x0, x1, y0, y1):
        m = (y1 - y0) /  (x1 - x0) # Calculate slope.
        return (xout - x0) * m + y0 # Point slope formula.

    """
    Given a x value 'xout' to predict, this method finds the coordinates of 
    the two nearest points [(x0, y0), (x1, y1)] we have in our grid of 
    'self.xin' and 'self.yin' values
    """
    def __get_points(self, xout):
        # Change this method to accept a list of points. 
        lenxin = len(self.xin)

        # Search for the nearest index to 'xout'. This is the larger index
        # from which we will grid a line between two points (e.g. larger point).
        i1 = np.searchsorted(self.xin, xout)

        i1 = 1 if i1 == 0 else i1 # Corner case for start of array.
        i1 = lenxin-1 if i1 == lenxin else i1 # Corner case for end of array.

        x0 = self.xin[i1-1]
        x1 = self.xin[i1]
        y0 = self.yin[i1-1]
        y1 = self.yin[i1]
        return x0, x1, y0, y1

    """
    Interprets a single y value for a given x value ('xout').
    """
    def __get_single_point(self, xout):
        x0, x1, y0, y1 = self.__get_points(xout)
        if self.kind == 'nearest':
            return self.__nearest(xout, x0, x1, y0, y1)
        elif self.kind == 'linear':
            return self.__linear(xout, x0, x1, y0, y1)
        else:
            raise ValueError('Invalid interpolation kind: %s' % self.kind)

    """
    Performs interpolation on a tensor of input x values ('xout').
    """
    def __call__(self, xout):
      return xout.map_(xout, lambda a,b: self.__get_single_point(a))

    def plot(self):
        minval = self.xin.numpy()[0]
        maxval = self.xin.numpy()[len(self.xin)-1]

        # Create a plot for the 'actual' function by mapping small 
        # granularities.
        xnew = torch.from_numpy(np.arange(minval, maxval, 1/3600))
        ynew = self.fn(xnew)

        #print(self.xin)
        #print(self.yin)
        print("new")
        print(xnew)
        print(ynew)

        print("Memory Required: " + str(len(self.xin)*2*8) + " bytes")

        plt.plot(self.xin.numpy(), self.yin.numpy(), 'o',
                 xnew.numpy(), ynew.numpy(), '-')
        plt.show()
