#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Delay Head."

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

__author__ = "Ruo-Ze Liu"

debug = False


def checkNaNandInf(val, name):
    if torch.isnan(val).any():
        print(name, 'Find nan:', val)
    if torch.isinf(val).any():
        print(name, 'Find inf:', val)


class DelayHead(nn.Module):
    '''
    Inputs: autoregressive_embedding
    Outputs:
        delay_logits - The logits corresponding to the probabilities of each delay
        delay - The sampled delay
        autoregressive_embedding - Embedding that combines information from `lstm_output` and all previous sampled arguments. 
    '''

    def __init__(self, autoregressive_embedding_size=AHP.autoregressive_embedding_size, 
                 original_256=AHP.original_256, max_delay=SFS.last_delay):
        super().__init__()
        self.fc_1 = nn.Linear(autoregressive_embedding_size, original_256)  # with relu
        self.fc_2 = nn.Linear(original_256, original_256)  # with relu
        self.max_delay = max_delay

        self.embed_fc = nn.Linear(original_256, max_delay)  # no relu

        self.fc_3 = nn.Linear(max_delay, original_256)  # with relu
        self.fc_4 = nn.Linear(original_256, original_256)  # with relu
        self.project = nn.Linear(original_256, autoregressive_embedding_size)

        self.softmax = nn.Softmax(dim=-1)

    def preprocess(self):
        pass

    def forward(self, autoregressive_embedding):
        checkNaNandInf(autoregressive_embedding, 'autoregressive_embedding')

        # AlphaStar: `autoregressive_embedding` is decoded using a 2-layer (each with size 256) 
        # AlphaStar: linear network with ReLUs,
        x = F.relu(self.fc_1(autoregressive_embedding))
        print("x.shape:", x.shape) if debug else None
        checkNaNandInf(x, 'x')
        x = F.relu(self.fc_2(x))
        checkNaNandInf(x, 'x')

        # AlphaStar: before being embedded into `delay_logits` that has size 128 (one for each 
        # AlphaStar: possible requested delay in game steps).
        # note: no temperature used here
        delay_logits = self.embed_fc(x)
        checkNaNandInf(delay_logits, 'delay_logits')

        # AlphaStar: `delay` is sampled from `delay_logits` using a multinomial, though unlike all other arguments,
        # AlphaStar: no temperature is applied to `delay_logits` before sampling.
        delay_probs = self.softmax(delay_logits)
        delay = torch.multinomial(delay_probs, 1)
        checkNaNandInf(delay, 'delay')
        print("delay:", delay) if debug else None

        # AlphaStar: Similar to `action_type`, `delay` is projected to a 1D tensor of size 1024 through 
        # AlphaStar: a 2-layer (each with size 256) linear network with ReLUs, and added to `autoregressive_embedding`
        # similar to action_type here, change it to one_hot version
        delay_one_hot = L.one_hot_embedding(delay, self.max_delay)
        # to make the dim of delay_one_hot as delay
        delay_one_hot = delay_one_hot.squeeze(-2)
        print("delay_one_hot:", delay_one_hot) if debug else None
        print("delay_one_hot.shape:", delay_one_hot.shape) if debug else None
        z = F.relu(self.fc_3(delay_one_hot))
        z = F.relu(self.fc_4(z))
        t = self.project(z)
        # the operation may auto broadcasting, so we need a test
        y = autoregressive_embedding + t
        # make sure autoregressive_embedding has the same shape as y, prevent the auto broadcasting
        assert autoregressive_embedding.shape == y.shape
        autoregressive_embedding = y

        return delay_logits, delay, autoregressive_embedding


def test():
    batch_size = 2
    autoregressive_embedding = torch.randn(batch_size, AHP.autoregressive_embedding_size)
    delay_head = DelayHead()

    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    delay_logits, delay, autoregressive_embedding = delay_head.forward(autoregressive_embedding)

    print("delay_logits:", delay_logits) if debug else None
    print("delay_logits.shape:", delay_logits.shape) if debug else None
    print("delay:", delay) if debug else None
    print("delay.shape:", delay.shape) if debug else None
    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
