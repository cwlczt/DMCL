import os

import torch

def print_model(model):
    for name, parameters in model.named_parameters(): # if there is no parameter in the model, this will raise an error
        print(name, ':', parameters.size())

def print_model_weights(model):

    for param in model.parameters():
        print(torch.sum(param.grad.data))

def check_model_updates(model):
    for ae in model:
        print(torch.sum(ae.enc_1.weight))


def save_model(opts, model, optimizer, current_epoch):
    out = os.path.join(opts.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)