import torch
import torchvision
import logging

# The model we will analyze!
model = torchvision.models.resnet50()

"""
Part 1: Compute the storage cost of the model (a.k.a. weights a.k.a. 
parameters).

Loop over the model and print out the size of each weight. The total number of 
bytes is summed into the 'weights_bytes' variable. If the 'verbose_weights'
flag is set the name of the weight (or parameter) and the number of bytes
it consumes is printed for each layer.
"""
def get_num_bytes_from_type(dtype):
    """
    Helper function to return the number of bytes given a torch type.
    TODO: fill out for all types.
    """
    if dtype == torch.float32:
        return 4
    elif dtype == torch.float64:
        return 8
    elif dtype == torch.int:
        return 4
    elif dtype == torch.long:
        return 8
    else:
        raise ValueError("Type not yet supported: " + str(dtype))

def get_num_bytes(tensor):
    """
    Helper function to return the number of bytes that a given tensor 'tensor'
    consumes.
    """
    return tensor.numel()*get_num_bytes_from_type(tensor.dtype)

weights_bytes = 0
verbose_weights = False
for name, param in model.named_parameters():
    if param.requires_grad:
        param_bytes = get_num_bytes(param)
        weights_bytes += param_bytes
        if verbose_weights:
            print(name, param_bytes)

print("Total Weights Bytes:\t" + str(weights_bytes) + " bytes")

"""
Part 2: Calculate how much memory is used to store the gradients. This is summed
into 'grad_storage'. This gets a litte more intense :). 
"""
grad_storage = 0
def hookFunc(module, gradInput, gradOutput):
    global grad_storage
    """
    For a given layer there will be multiple gradients flowing into it in the 
    backwards pass. These are 'gradInput'. There will be one output gradient
    that is computed at this layer in the backwards pass. This is 'gradOuput'.

    To sum this properly and avoid double counting we only count the size of the 
    'gradOuput'. This output could then be sent as input to many other nodes
    in the backwards (so if you count both you are double/triple/... counting).
    """
    for v in gradInput:
        if v is not None:
            grad_storage += get_num_bytes(v)

def find_leaves(layer, name=None):
    """
    This function traverses down to the base modules in the module (nested).
    At the base or leaf modules we register a backwards hook that will account
    for the size of the gradients that can be stored.s
    """

    # I could not find a clean way to find if you had a child or not. There 
    # I used this 'is_leaf' hack. Works good enough.
    is_leaf = True 
    for child_name, child in layer.named_children():
        is_leaf = False
        find_leaves(child, child_name)
    if is_leaf:
        layer.register_backward_hook(hookFunc)

find_leaves(model)

# Create a dummy model and dummy loss (here just a sum). Simulate a forward
# and backward through the model. TODO: Add in something like using different
# strides or batch sizes.
x = torch.randn(1, 3, 224, 224)
out = model(x)
out.sum().backward()

print("Gradient Bytes:\t\t" + str(grad_storage) + " bytes")

"""
Part 3: Calculate extra memory that might be used in the optimizer.
Look at: https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
"""
optimizer_bytes = 0
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# The only term we need to worry about is the momentum term. Everything else
# appears to be a scalar.
for group in optimizer.param_groups:
    momentum = group['momentum']
    for p in group['params']:
        if momentum != 0:
            param_state = optimizer.state[p]
            optimizer_bytes += get_num_bytes(torch.zeros_like(p.data))



print("Optimizer Bytes:\t" + str(optimizer_bytes) + " bytes")




