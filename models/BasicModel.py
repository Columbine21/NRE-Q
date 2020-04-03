import time
import torch
import torch.nn as nn


class BasicModel(nn.Module):
    """ provide save & load method."""
    def __init__(self):
        super(BasicModel, self).__init__()
        self.model_name = str(type(self))   # model name

    def load(self, path):
        """ load model from the path. """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """ save the model with default name model name + time """
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            # month day_hours:min:sec.pth
            name = name + time.strftime('%m%d_%H:%M:%S.pth')
        else:
            name = prefix + name + '.pth'
        torch.save(self.state_dict(), name)

        return name


