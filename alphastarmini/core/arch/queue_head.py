#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Queue Head."

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

__author__ = "Ruo-Ze Liu"

debug = False


class QueueHead(nn.Module):
    '''
    Inputs: autoregressive_embedding, action_type, embedded_entity
    Outputs:
        queued_logits - The logits corresponding to the probabilities of queueing and not queueing
        queued - Whether or no to queue this action.
        autoregressive_embedding - Embedding that combines information from `lstm_output` and all previous sampled arguments. 
    '''

    def __init__(self, input_size=AHP.autoregressive_embedding_size, 
                 original_256=AHP.original_256,
                 max_queue=SFS.last_repeat_queued, is_sl_training=True, temperature=0.8):
        super().__init__()
        self.is_sl_training = is_sl_training
        if not self.is_sl_training:
            self.temperature = temperature
        else:
            self.temperature = 1.0

        self.fc_1 = nn.Linear(input_size, original_256)  # with relu
        self.fc_2 = nn.Linear(original_256, original_256)  # with relu
        self.max_queue = max_queue

        self.embed_fc = nn.Linear(original_256, max_queue)  # no relu

        self.fc_3 = nn.Linear(max_queue, original_256)  # with relu
        self.fc_4 = nn.Linear(original_256, original_256)  # with relu
        self.project = nn.Linear(original_256, AHP.autoregressive_embedding_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def preprocess(self):
        pass

    # QUESTION: It is similar to delay head. But how did it use the embedded_entity?
    def forward(self, autoregressive_embedding, action_type, embedded_entity=None):
        # AlphaStar: Queued Head is similar to the delay head except a temperature of 0.8 
        # AlphaStar: is applied to the logits before sampling,
        x = self.fc_1(autoregressive_embedding)
        print("x.shape:", x.shape) if debug else None
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        # note: temperature is used here, compared to delay head
        queue_logits = self.embed_fc(x).div(self.temperature)
        queue_probs = self.softmax(queue_logits)
        # AlphaStar: the size of `queued_logits` is 2 (for queueing and not queueing),
        queue = torch.multinomial(queue_probs, 1)

        # similar to action_type here, change it to one_hot version
        queue_one_hot = L.one_hot_embedding(queue, self.max_queue)
        # to make the dim of queue_one_hot as queue
        queue_one_hot = queue_one_hot.squeeze(-2)

        z = self.relu(self.fc_3(queue_one_hot))
        z = self.relu(self.fc_4(z))
        t = self.project(z)
        # make sure autoregressive_embedding has the same shape as y, prevent the auto broadcasting
        assert autoregressive_embedding.shape == t.shape

        # AlphaStar: and the projected `queued` is not added to `autoregressive_embedding` 
        # AlphaStar: if queuing is not possible for the chosen `action_type`
        # note: projected `queued` is not added to `autoregressive_embedding` if queuing is not 
        # possible for the chosen `action_type`

        assert action_type.shape[0] == autoregressive_embedding.shape[0]
        mask = L.action_can_be_queued_mask(action_type).float()
        print("mask:", mask) if debug else None
        autoregressive_embedding = autoregressive_embedding + mask * t

        ''' # below code only consider the cases when action_type is scalar
        if L.action_can_be_queued(action_type):
            autoregressive_embedding = autoregressive_embedding + t
        else:
            print("None add to autoregressive_embedding!") if debug else None
        '''

        return queue_logits, queue, autoregressive_embedding


def test():
    batch_size = 2
    autoregressive_embedding = torch.randn(batch_size, AHP.autoregressive_embedding_size)
    action_type = torch.randint(low=0, high=SFS.available_actions, size=(batch_size, 1))
    queue_head = QueueHead()

    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    queue_logits, queue, autoregressive_embedding = queue_head.forward(autoregressive_embedding, action_type)

    print("queue_logits:", queue_logits) if debug else None
    print("queue_logits.shape:", queue_logits.shape) if debug else None
    print("queue:", queue) if debug else None
    print("queue.shape:", queue.shape) if debug else None
    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
