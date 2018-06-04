from graphviz import Digraph
import torch
from torch.autograd import Variable
from collections import namedtuple
import logging
import os

class SizeUsage:
    def __init__(self):
        self.weights_bytes = 0
        self.grad_weights_bytes = 0
        self.cached_bytes = 0

    def __repr__(self):
        return \
            f"""Total Weights Bytes: {self.weights_bytes}\n"""\
            f"""Total Grad Weights Bytes: {self.grad_weights_bytes}"""

def traverse(var, params, bytes_per_elem=4, log="traversal"):
    """ 
    Traverses a pytorch autograd graph by going down backwards from the
    'var' tensor given as input.

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if not os.path.exists("log"):
        os.makedirs("log")

    handler = logging.FileHandler("log/"+log+".log")  
    logger = logging.getLogger(log)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    seen = set()
    sizes = SizeUsage()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        global weights_bytes, grad_weights_bytes
        if var not in seen:
            # Check to see if this is an input tensor?
            if torch.is_tensor(var):
                logger.info(str(type(var).__name__))
                logger.warn("Got a tensor but am not accounting for this size.")
            # Check to see if it is part of the model?
            elif hasattr(var, 'variable'):
                u = var.variable
                w_b = u.data.numel() * bytes_per_elem
                g_w_b = 0
                name = param_map[id(u)] if params is not None else ''
                if u.grad is not None:
                    g_w_b = u.grad.data.numel() * bytes_per_elem
                    sizes.grad_weights_bytes += g_w_b
                sizes.weights_bytes += w_b
                logger.info(str(name) + ": bytes(" + str(w_b) +")" \
                                              " grad_bytes(" + str(g_w_b) +")" )
            # These are the function nodes.
            else:
                logger.info(str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    add_nodes(t)
    
    add_nodes(var.grad_fn)

    return sizes
