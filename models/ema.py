import copy
import mlx.nn as nn
import mlx.core as mx


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.parameters().items():
            if param.requires_grad:
                self.shadow[name] = mx.array(param)

    def update(self, module):
        for name, param in module.parameters().items():
            if param.requires_grad:
                self.shadow[name] = (1. - self.mu) * param + self.mu * self.shadow[name]

    def ema(self, module):
        for name, param in module.parameters().items():
            if param.requires_grad:
                param = self.shadow[name]

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
