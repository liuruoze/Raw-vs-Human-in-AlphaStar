#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Selected Units Head."

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

__author__ = "Ruo-Ze Liu"

debug = False


class SelectedUnitsHead(nn.Module):
    '''
    Inputs: autoregressive_embedding, action_type, entity_embeddings
    Outputs:
        units_logits - The logits corresponding to the probabilities of selecting each unit, 
            repeated for each of the possible 64 unit selections
        units - The units selected for this action.
        autoregressive_embedding - Embedding that combines information from `lstm_output` and all previous sampled arguments.
    '''

    def __init__(self, embedding_size=AHP.entity_embedding_size, 
                 max_number_of_unit_types=SCHP.max_unit_type, is_sl_training=True, 
                 temperature=0.8, max_selected=AHP.max_selected,
                 original_256=AHP.original_256, original_32=AHP.original_32,
                 autoregressive_embedding_size=AHP.autoregressive_embedding_size):
        super().__init__()
        self.is_sl_training = is_sl_training
        if not self.is_sl_training:
            self.temperature = temperature
        else:
            self.temperature = 1.0

        self.max_number_of_unit_types = max_number_of_unit_types
        self.func_embed = nn.Linear(max_number_of_unit_types, original_256)  # with relu

        self.conv_1 = nn.Conv1d(in_channels=embedding_size, 
                                out_channels=original_32, kernel_size=1, stride=1,
                                padding=0, bias=True)

        self.fc_1 = nn.Linear(autoregressive_embedding_size, original_256)
        self.fc_2 = nn.Linear(original_256, original_32)

        self.small_lstm = nn.LSTM(input_size=original_32, hidden_size=original_32, num_layers=1, 
                                  dropout=0.0, batch_first=True)

        self.max_selected = max_selected

        self.max_entities = AHP.max_entities

        self.fc_3 = nn.Linear(autoregressive_embedding_size, self.max_entities)

        # self.fc_4 = nn.Linear(autoregressive_embedding_size, original_32)
        # self.layer_norm_k = nn.LayerNorm([self.max_entities, original_32], eps=1e-09, elementwise_affine=False)
        # self.layer_norm_q = nn.LayerNorm(original_32, eps=1e-09, elementwise_affine=False)

        self.project = nn.Linear(original_32, autoregressive_embedding_size)
        self.softmax = nn.Softmax(dim=-1)

    def preprocess(self):
        pass

    def forward(self, autoregressive_embedding, action_type, entity_embeddings):
        '''
        Inputs:
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
            action_type: [batch_size x 1]
            entity_embeddings: [batch_size x entity_size x embedding_size]
        Output:
            units_logits: [batch_size x max_selected x entity_size]
            units: [batch_size x max_selected x 1]
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
        '''

        batch_size = entity_embeddings.shape[0]
        entity_size = entity_embeddings.shape[-2]
        unit_types_one_hot = L.action_can_apply_to_entity_types_mask(action_type)

        device = next(self.parameters()).device
        print("device:", device) if debug else None
        # note! to(device) on tensor is not in-place operation, so don't only use unit_types_one_hot.to(device)
        unit_types_one_hot = unit_types_one_hot.to(device)

        the_func_embed = F.relu(self.func_embed(unit_types_one_hot))
        key = self.conv_1(entity_embeddings.transpose(-1, -2)).transpose(-1, -2)
        # key shape [batch_size, entity_size, embedd_size]   
        print("key:", key) if debug else None

        # key = self.layer_norm_k(key)
        # print("key:", key) if debug else None

        units_logits = []
        units = []

        select_unit_mask = L.action_involve_selecting_units_mask(action_type)

        if action_type.shape[0] == 1:
            print("action_type:", action_type.item()) if 1 else None

        # x = self.fc_1(autoregressive_embedding)
        # z_1 = the_func_embed + x
        # z_2 = self.fc_2(F.relu(z_1)).unsqueeze(1)
        # print("z_2:", z_2) if 1 else None
        # y = torch.bmm(key, z_2.transpose(-1, -2)).squeeze(-1)

        # print("key.shape:", key.shape) if debug else None
        # print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

        # q = self.fc_4(autoregressive_embedding)
        # # shape [batch_size, embedd_size]
        # print("q:", q) if debug else None
        # q = self.layer_norm_q(q)
        # print("q:", q) if debug else None
        # y = torch.matmul(key, q.unsqueeze(-1)).squeeze(-1) / (AHP.original_32 ** 0.5)

        #y_1 = self.fc_3(autoregressive_embedding)
        #y = self.fc_4(F.relu(y_1))

        # y = torch.bmm(key, x.transpose(-1, -2))
        # print("y:", y) if 1 else None
        # y = y.squeeze(-1)

        y = self.fc_3(autoregressive_embedding)
        print("y.shape:", y.shape) if debug else None

        entity_logits = y.div(self.temperature)
        # entity_logits shape: [batch_size x entity_size]
        print("entity_logits:", entity_logits) if debug else None
        print("entity_logits.shape:", entity_logits.shape) if debug else None

        entity_probs = self.softmax(entity_logits)
        # entity_probs shape: [batch_size x entity_size]
        print("entity_probs:", entity_probs) if 1 else None
        print("entity_probs.shape:", entity_probs.shape) if debug else None

        entity_id = torch.multinomial(entity_probs, 1)
        # TODO: Wenhai: If this entity_id is a EOF, end the selection

        # entity_id shape: [batch_size x 1]
        print("entity_id:", entity_id) if debug else None
        print("entity_id.shape:", entity_id.shape) if debug else None

        # note, we add a dimension where is in the seq_one to help
        # we concat to the one : [batch_size x max_selected x ?]
        units_logits.append(entity_logits.unsqueeze(-2))
        units.append(entity_id.unsqueeze(-2))

        # AlphaStar: The one-hot position of the selected entity is multiplied by the keys,
        entity_one_hot = L.to_one_hot(entity_id, entity_size).squeeze(-2)
        # entity_one_hot shape: [batch_size x entity_size]
        print("entity_one_hot:", entity_one_hot) if debug else None
        print("entity_one_hot.shape:", entity_one_hot.shape) if debug else None

        entity_one_hot_unsqueeze = entity_one_hot.unsqueeze(-2) 
        # entity_one_hot_unsqueeze shape: [batch_size x seq_len x entity_size], note seq_len =1 
        # key_shape: [batch_size x entity_size x key_size], note key_size = 32
        out = torch.bmm(entity_one_hot_unsqueeze, key).squeeze(-2)
        # out shape: [batch_size x key_size]
        print("out:", out) if debug else None
        print("out.shape:", out.shape) if debug else None

        # AlphaStar: reduced by the mean across the entities,
        # Wenhai: should be key mean
        # Ruo-Ze: should be out mean
        # mean = torch.mean(entity_one_hot, dim=-1, keepdim=True)
        mean = torch.mean(out, dim=-1, keepdim=True)
        # mean shape: [batch_size x 1]

        out = out - mean
        print("out:", out) if debug else None
        print("out.shape:", out.shape) if debug else None

        # AlphaStar: passed through a linear layer of size 1024,
        t = self.project(out)
        print("t:", t) if debug else None
        print("t.shape:", t.shape) if debug else None

        # AlphaStar: and added to `autoregressive_embedding` for subsequent iterations.
        assert autoregressive_embedding.shape == t.shape
        autoregressive_embedding = autoregressive_embedding + t * select_unit_mask.float()
        print("autoregressive_embedding:",
              autoregressive_embedding) if debug else None
        print("autoregressive_embedding.shape:",
              autoregressive_embedding.shape) if debug else None

        # QUESTION: When to break?

        units_logits = torch.cat(units_logits, dim=1)
        print("units_logits:", units_logits) if debug else None
        print("units_logits.shape:", units_logits.shape) if debug else None

        # units_logits: [batch_size x max_selected x entity_size]
        units = torch.cat(units, dim=1)
        # units: [batch_size x max_selected x 1]

        # autoregressive_embedding: [batch_size x autoregressive_embedding_size]
        return units_logits, units, autoregressive_embedding


def test():
    batch_size = 2
    autoregressive_embedding = torch.randn(batch_size, AHP.autoregressive_embedding_size)
    action_type = torch.randint(low=0, high=SFS.available_actions, size=(batch_size, 1))
    entity_embeddings = torch.randn(batch_size, AHP.max_entities, AHP.entity_embedding_size)

    selected_units_head = SelectedUnitsHead()

    print("autoregressive_embedding:",
          autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:",
          autoregressive_embedding.shape) if debug else None

    units_logits, units, autoregressive_embedding = \
        selected_units_head.forward(
            autoregressive_embedding, action_type, entity_embeddings)

    if units_logits is not None:
        print("units_logits:", units_logits) if debug else None
        print("units_logits.shape:", units_logits.shape) if debug else None
    else:
        print("units_logits is None!")

    if units is not None:
        print("units:", units) if debug else None
        print("units.shape:", units.shape) if debug else None
    else:
        print("units is None!")

    print("autoregressive_embedding:",
          autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:",
          autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
