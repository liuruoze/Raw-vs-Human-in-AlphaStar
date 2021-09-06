#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Agent."

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysc2.lib import actions as A

from alphastarmini.core.arch.arch_model import ArchModel
from alphastarmini.core.arch.entity_encoder import Entity

from alphastarmini.core.rl.action import ArgsAction
from alphastarmini.core.rl.state import MsState

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label

from alphastarmini.lib import utils as L

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import MiniStar_Arch_Hyper_Parameters as MAHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP
from alphastarmini.lib.hyper_parameters import Scalar_Feature_Size as SFS

from pysc2.lib.units import get_unit_type

__author__ = "Ruo-Ze Liu"

debug = False


class Agent(object):
    def __init__(self, weights=None):
        self.model = ArchModel()
        self.hidden_state = None

        if weights is not None:
            self.set_weights(weights)

    def init_hidden_state(self):
        if self.model is not None:
            return self.model.init_hidden_state()
        else:
            return None

    def device(self):
        device = next(self.model.parameters()).device
        return device

    def to(self, DEVICE):
        self.model.to(DEVICE)

    def unroll(self, one_traj):
        action_output = []
        for traj_step in one_traj:
            (feature, label, is_final) = traj_step
            state = Feature.feature2state(feature)
            action_logits_prdict, self.hidden_state = self.action_logits_by_state(state, self.hidden_state)
            action_output.append(action_logits_prdict)
            if is_final:
                self.hidden_state = self.init_hidden_state()

        return action_output

    def preprocess_state_all(self, obs, build_order=None):
        batch_entities_tensor = self.preprocess_state_entity(obs)
        scalar_list = self.preprocess_state_scalar(obs, build_order=build_order)
        map_data = self.preprocess_state_spatial(obs)
        state = MsState(entity_state=batch_entities_tensor, 
                        statistical_state=scalar_list, map_state=map_data)

        return state

    def get_state_and_action_from_pickle(self, obs):
        batch_entities_tensor = self.preprocess_state_entity(obs)
        scalar_list = self.preprocess_state_scalar(obs)
        map_data = self.preprocess_state_spatial(obs)
        state = MsState(entity_state=batch_entities_tensor, 
                        statistical_state=scalar_list, map_state=map_data)

        return state

    def get_scalar_list(self, obs, build_order=None):
        scalar_list = []

        # implement the agent_statistics
        player = obs["player"]
        player_statistics = player[1:]
        agent_statistics = torch.tensor(player_statistics, dtype=torch.float).reshape(1, -1)
        print('agent_statistics:', agent_statistics) if debug else None

        # implement the upgrades
        upgrades = torch.zeros(1, SFS.upgrades)
        obs_upgrades = obs["upgrades"]
        print('obs_upgrades:', obs_upgrades) if debug else None
        for u in obs_upgrades:
            assert u >= 0 
            assert u < SFS.upgrades
            upgrades[0, u] = 1

        # implement the unit_counts_bow
        unit_counts_bow = L.calculate_unit_counts_bow(obs)
        print('unit_counts_bow:', unit_counts_bow) if debug else None
        print('torch.sum(unit_counts_bow):', torch.sum(unit_counts_bow)) if debug else None

        # TODO: implement the units_buildings
        units_buildings = torch.randn(1, SFS.units_buildings)

        # implement the effects
        effects = torch.zeros(1, SFS.effects)
        # we now use feature_effects to represent it
        feature_effects = obs["feature_effects"]
        print('feature_effects:', feature_effects) if debug else None
        for effect in feature_effects:
            e = effect.effect_id
            assert e >= 0 
            assert e < SFS.effects
            effects[0, e] = 1
        # the raw effects are reserved for use
        raw_effects = obs["raw_effects"]
        print('raw_effects:', raw_effects) if debug else None

        # now we simplely make upgrade the same as upgrades
        upgrade = upgrades

        # implement the build order
        # TODO: add the pos of buildings
        beginning_build_order = torch.zeros(1, SCHP.count_beginning_build_order, int(SFS.beginning_build_order / SCHP.count_beginning_build_order))
        print('beginning_build_order.shape:', beginning_build_order.shape) if debug else None
        if build_order is not None:
            # implement the beginning_build_order               
            for i, bo in enumerate(build_order):
                if i < 20:
                    assert bo < SFS.unit_counts_bow
                    beginning_build_order[0, i, bo] = 1
            print("beginning_build_order:", beginning_build_order) if debug else None
            print("sum(beginning_build_order):", torch.sum(beginning_build_order).item()) if debug else None

        scalar_list.append(agent_statistics)
        scalar_list.append(upgrades)
        scalar_list.append(unit_counts_bow)
        scalar_list.append(units_buildings)
        scalar_list.append(effects)
        scalar_list.append(upgrade)
        scalar_list.append(beginning_build_order)

        return scalar_list

    def preprocess_baseline_state(self, home_obs, away_obs, build_order=None):
        batch_size = 1

        agent_scalar_list = self.get_scalar_list(home_obs, build_order)    
        opponenet_scalar_out = self.get_scalar_list(away_obs)  

        return agent_scalar_list, opponenet_scalar_out

    def preprocess_state_scalar(self, obs, build_order=None):
        scalar_list = []

        player = obs["player"]
        print('player:', player) if debug else None

        # The first is player_id, so we don't need it.
        player_statistics = player[1:]
        print('player_statistics:', player_statistics) if debug else None

        # player_statistics = np.log(player_statistics + 1)
        # print('player_statistics:', player_statistics)

        # agent_statistics = torch.ones(1, 10)
        agent_statistics = torch.tensor(player_statistics, dtype=torch.float).reshape(1, -1)
        print('player_statistics:', agent_statistics) if debug else None

        home_race = torch.zeros(1, 5)
        if "home_race_requested" in obs:
            home_race_requested = obs["home_race_requested"].item()
            print('home_race_requested:', home_race_requested) if debug else None
        else:
            home_race_requested = 0
        assert home_race_requested >= 0 and home_race_requested <= 4
        home_race[0, home_race_requested] = 1
        print('home_race:', home_race) if debug else None

        away_race = torch.zeros(1, 5)
        if "away_race_requested" in obs:
            away_race_requested = obs["away_race_requested"].item()
            print('away_race_requested:', away_race_requested) if debug else None
        else:
            away_race_requested = 0
        assert away_race_requested >= 0 and away_race_requested <= 4
        away_race[0, away_race_requested] = 1
        print('away_race:', away_race) if debug else None

        if "action_result" in obs:
            action_result = obs["action_result"]
            print('action_result:', action_result) if debug else None

        if "alerts" in obs:
            alerts = obs["alerts"]
            print('alerts:', alerts) if debug else None

        # implement the upgrades
        upgrades = torch.zeros(1, SFS.upgrades)
        obs_upgrades = obs["upgrades"]
        print('obs_upgrades:', obs_upgrades) if debug else None
        for u in obs_upgrades:
            assert u >= 0 
            assert u < SFS.upgrades
            upgrades[0, u] = 1

        # question: how to know enemy's upgrades?
        enemy_upgrades = torch.zeros(1, SFS.upgrades)

        # time conver to gameloop
        time = torch.zeros(1, SFS.time)
        game_loop = obs["game_loop"]
        print('game_loop:', game_loop) if debug else None

        time_encoding = torch.tensor(L.unpackbits_for_largenumber(game_loop, num_bits=64), dtype = torch.float).reshape(1, -1)
        print('time_encoding:', time_encoding) if debug else None 
        # note, we use binary encoding here for time
        time = time_encoding
        #time[0, 0] = game_loop

        # implement the available_actions
        # note: if we use raw action, this key doesn't exist
        # the_available_actions = obs["available_actions"] 
        # print('the_available_actions:', the_available_actions) if 1 else None
        available_actions = torch.zeros(1, SFS.available_actions)

        # implement the unit_counts_bow
        unit_counts_bow = L.calculate_unit_counts_bow(obs)
        print('unit_counts_bow:', unit_counts_bow) if debug else None
        print('torch.sum(unit_counts_bow):', torch.sum(unit_counts_bow)) if debug else None

        # implement the build order
        beginning_build_order = torch.zeros(1, SCHP.count_beginning_build_order, int(SFS.beginning_build_order / SCHP.count_beginning_build_order))
        print('beginning_build_order.shape:', beginning_build_order.shape) if debug else None
        if build_order is not None:
            # implement the beginning_build_order               
            for i, bo in enumerate(build_order):
                if i < 20:
                    assert bo < SFS.unit_counts_bow
                    beginning_build_order[0, i, bo] = 1
            print("beginning_build_order:", beginning_build_order) if debug else None
            print("sum(beginning_build_order):", torch.sum(beginning_build_order).item()) if debug else None

        mmr = torch.zeros(1, SFS.mmr)
        units_buildings = torch.zeros(1, SFS.units_buildings)

        # implement the effects
        effects = torch.zeros(1, SFS.effects)
        # we now use feature_effects to represent it
        feature_effects = obs["feature_effects"]
        print('feature_effects:', feature_effects) if debug else None
        for effect in feature_effects:
            e = effect.effect_id
            assert e >= 0 
            assert e < SFS.effects
            effects[0, e] = 1
        # the raw effects are reserved for use
        raw_effects = obs["raw_effects"]
        print('raw_effects:', raw_effects) if debug else None

        # implement the upgrade
        upgrade = torch.zeros(1, SFS.upgrades)
        for u in obs_upgrades:
            assert u >= 0 
            assert u < SFS.upgrades
            upgrade[0, u] = 1

        last_delay = torch.zeros(1, SFS.last_delay)

        # implement the last action
        # note: if we use raw action, this property is always empty
        last_actions = obs["last_actions"]
        print('last_actions:', last_actions) if debug else None
        last_action_type = torch.zeros(1, SFS.last_action_type)

        last_repeat_queued = torch.zeros(1, SFS.last_repeat_queued)

        scalar_list.append(agent_statistics)
        scalar_list.append(home_race)
        scalar_list.append(away_race)
        scalar_list.append(upgrades)
        scalar_list.append(enemy_upgrades)
        scalar_list.append(time)

        scalar_list.append(available_actions)
        scalar_list.append(unit_counts_bow)
        scalar_list.append(mmr)
        scalar_list.append(units_buildings)
        scalar_list.append(effects)
        scalar_list.append(upgrade)

        scalar_list.append(beginning_build_order)
        scalar_list.append(last_delay)
        scalar_list.append(last_action_type)
        scalar_list.append(last_repeat_queued)

        return scalar_list

    def preprocess_state_entity(self, obs, return_tag_list = False):
        raw_units = obs["raw_units"]

        our_unit_list = []
        nexus_list = []
        probe_list = []
        idle_probe_list = []
        for u in raw_units:
            # only include the units we have
            if u.alliance == 1:
                # our_unit_list.append(u)
                if u.unit_type == 59:
                    nexus_list.append(u)
                if u.unit_type == 84:
                    probe_list.append(u)
                    if u.order_length == 0:
                        idle_probe_list.append(u)

        our_unit_list.extend(nexus_list)

        def myFunc(e):
            return e.tag
        # note we must ensure the index is the same across state -> action
        probe_list.sort(reverse=False, key=myFunc)
        our_unit_list.extend(probe_list)

        #print("preprocess: our_unit_list", our_unit_list) if 1 else None
        print("preprocess: len(our_unit_list)", len(our_unit_list)) if debug else None
        for u in our_unit_list:
            print(u.tag) if debug else None

        e_list = []
        tag_list = []

        index = 0
        for i, raw_unit in enumerate(our_unit_list):
            unit_type = raw_unit.unit_type
            alliance = raw_unit.alliance
            tag = raw_unit.tag
            # note: wo only consider the entities not beyond the max number
            if index < AHP.max_entities:
                e = Entity()
                e.unit_type = raw_unit.unit_type
                # e.unit_attributes = None
                e.alliance = raw_unit.alliance
                e.health = raw_unit.health
                e.shield = raw_unit.shield
                e.energy = raw_unit.energy
                e.cargo_space_taken = raw_unit.cargo_space_taken
                e.cargo_space_max = raw_unit.cargo_space_max
                e.build_progress = raw_unit.build_progress
                e.current_health_ratio = raw_unit.health_ratio
                e.current_shield_ratio = raw_unit.shield_ratio
                e.current_energy_ratio = raw_unit.energy_ratio
                # e.health_max = None
                # e.shield_max = None
                # e.energy_max = None
                e.display_type = raw_unit.display_type
                e.x = raw_unit.x
                e.y = raw_unit.y
                e.is_cloaked = raw_unit.cloak
                e.is_powered = raw_unit.is_powered
                e.is_hallucination = raw_unit.hallucination
                e.is_active = raw_unit.active
                e.is_on_screen = raw_unit.is_on_screen
                e.is_in_cargo = raw_unit.is_in_cargo
                e.current_minerals = raw_unit.mineral_contents
                e.current_vespene = raw_unit.vespene_contents
                # e.mined_minerals = None
                # e.mined_vespene = None
                e.assigned_harvesters = raw_unit.assigned_harvesters
                e.ideal_harvesters = raw_unit.ideal_harvesters
                e.weapon_cooldown = raw_unit.weapon_cooldown
                e.order_length = raw_unit.order_length
                e.order_1 = raw_unit.order_id_0
                e.order_2 = raw_unit.order_id_1
                e.order_3 = raw_unit.order_id_2
                e.order_4 = raw_unit.order_id_3
                e.order_progress_1 = raw_unit.order_progress_0
                e.order_progress_2 = raw_unit.order_progress_1
                e.buff_id_1 = raw_unit.buff_id_0
                e.buff_id_2 = raw_unit.buff_id_1
                e.addon_unit_type = raw_unit.addon_unit_type
                e.attack_upgrade_level = raw_unit.attack_upgrade_level
                e.armor_upgrade_level = raw_unit.armor_upgrade_level
                e.shield_upgrade_level = raw_unit.shield_upgrade_level
                e.is_selected = raw_unit.is_selected
                # e.is_targeted = None 

                # add tag
                # note: we use tag to find the right index of entity
                e.tag = raw_unit.tag
                e_list.append(e)

                # note: the unit_tags and target_unit_tag in pysc2 actions
                # are actually index in _raw_tags!
                # so they are different from the tags really used by a SC2-action!
                # thus we only need to append index, not sc2 tags          
                # tag_list.append(e.tag)
                tag_list.append(i)  # we only need to append index, not sc2 tags 
            else:
                break

            index += 1

        print("len(e_list)", len(e_list)) if debug else None

        # preprocess entity list
        entities_tensor = self.model.preprocess_entity(e_list)

        print("entities_tensor: len(e_list)", len(e_list)) if debug else None
        for u in e_list:
            print(u.tag) if debug else None

        batch_entities_tensor = torch.unsqueeze(entities_tensor, dim = 0) 

        if return_tag_list:
            return batch_entities_tensor, tag_list

        return batch_entities_tensor

    def preprocess_state_spatial(self, obs):
        map_data = self.model.preprocess_spatial(obs)

        return map_data

    def action_by_obs(self, obs):
        state = self.preprocess_state_all(obs)
        action = self.model.forward(state)
        return action

    def action_by_state(self, state):
        action = self.model.forward(state)
        return action

    def action_logits_by_state(self, state, hidden_state = None, single_inference = False):
        batch_size = 1 if single_inference else None
        sequence_length = 1 if single_inference else None

        action_logits, actions, new_state = self.model.forward(state, batch_size = batch_size,
                                                               sequence_length = sequence_length,
                                                               hidden_state = hidden_state, 
                                                               return_logits = True)
        return action_logits, actions, new_state

    def state_by_obs(self, obs, return_tag_list = False):
        state, tag_list = self.preprocess_state_all(obs, return_tag_list)

        if tag_list and return_tag_list:
            return state, tag_list

        return state, None

    def func_call_to_action(self, func_call, obs = None):
        # note: this is a pysc2 action, and the 
        # unit_tags and target_unit_tag are actually index in _raw_tags!
        # so they are different from the tags really used by a SC2-action!
        func = func_call.function
        args = func_call.arguments

        print('function:', func) if debug else None
        print('function value:', func.value) if debug else None
        print('arguments:', args) if debug else None

        args_action = ArgsAction(use_tag = True)
        args_action.action_type = func.value

        # use a non-smart method to calculate the args of the action
        need_args = A.RAW_FUNCTIONS[func].args
        i = 0
        for arg in need_args:
            print("arg:", arg) if debug else None
            if arg.name == 'queued':
                args_action.queue = args[i][0].value
                i = i + 1
            elif arg.name == 'unit_tags':
                args_action.units = args[i]
                i = i + 1
            elif arg.name == 'target_unit_tag':
                args_action.target_unit = args[i][0]
                i = i + 1
            elif arg.name == 'world':
                scale_factor = 0.5 if SCHP.world_size == 128 else 1
                print('args[i]:', args[i]) if debug else None
                args_action.target_location = [int(x * scale_factor) for x in args[i]]
                print('args_action.target_location:', args_action.target_location) if debug else None
                i = i + 1

        if obs is not None:
            units = args_action.units
            print('units index:', units)
            if units is not None:
                unit_type = get_unit_type(obs["raw_units"][units[0]].unit_type)
                print('selected unit is:', unit_type)

        print('args_action:', args_action) if debug else None
        return args_action

    def action_to_func_call_RAS(self, action, action_spec, obs, use_random_args = False):
        # assert the action is single
        print('action:', action) if debug else None
        print('action.get_shape():', action.get_shape()) if debug else None

        action_id = action.action_type.item()
        print('action_id:', action_id) if debug else None

        function_id = L.action_type_index_map_raw(action_id)
        # function_id = 35  # TODO: Only in test, when training delete it!

        print('action_id:', action_id) if debug else None
        print('function_id:', function_id) if debug else None

        delay = action.delay.item()
        print('delay:', delay) if debug else None

        queue = action.queue.item()
        print('queue:', queue) if debug else None

        # we assume single inference
        units = action.units.cpu().detach().reshape(-1).numpy().tolist()
        print('units:', units) if debug else None

        target_unit = action.target_unit.item()  
        print('target_unit:', target_unit) if debug else None

        # we assume single inference
        target_location = action.target_location.cpu().detach().reshape(-1).numpy().tolist()
        print('target_location:', target_location) if debug else None

        need_args = action_spec.functions[function_id].args
        args = []

        def to_list(i):
            return [i]

        for unit in obs["raw_units"]:
            if unit.unit_type == 59:
                pass

        units_args = []    
        if not use_random_args:
            for arg in need_args:
                print("arg:", arg) if debug else None
                rand = [np.random.randint(0, size) for size in arg.sizes]

                if arg.name == 'queued':
                    size = arg.sizes[0]
                    if queue < 0 or queue > size - 1:
                        args.append(rand)
                        print("argument queue beyond the size!") if debug else None
                    else:
                        args.append(to_list(queue))
                elif arg.name == 'unit_tags':
                    # the unit_tags size is actually the max selected number
                    size = arg.sizes[0]

                    for unit_index in units:
                        if unit_index < 0 or unit_index > size - 1:
                            units_args.append(np.random.randint(0, size))
                            print("argument unit_index beyond the size!") if debug else None
                        else:
                            units_args.append(unit_index)

                    args.append(units_args)

                elif arg.name == 'target_unit_tag':
                    size = arg.sizes[0]
                    target = None
                    if target_unit < 0 or target_unit > size - 1:
                        target = rand
                        print("argument target_unit beyond the size!") if debug else None
                    else:
                        target = to_list(target_unit)

                    args.append(target)

                elif arg.name == 'world':
                    world_args = []

                    for val, size in zip(target_location, arg.sizes):
                        if val < 0 or val > size - 1:
                            world_args.append(np.random.randint(0, size))
                            print("argument world beyond the size!") if debug else None
                        elif val == 0:
                            world_args.append(np.random.randint(0, size))
                            print("argument world is 0, we change it to random!") if debug else None
                        else:
                            world_args.append(val)                        
                    print('world_args:', world_args) if debug else None

                    # for move camera, select fixed postion
                    if function_id == 168:
                        world_args = [40, 50]
                        print('world_args for move camera:', world_args) if debug else None

                    if function_id == 35:
                        # for Build_Pylon_pt, select randomly from some predined postion
                        rand_x = random.randint(-5, 5)
                        rand_y = random.randint(-5, 5)

                        world_args = [40 + rand_x * 2, 50 + rand_y * 2]
                        print('world_args for build pylon:', world_args) if debug else None

                    args.append(world_args)
        else:
            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in action_spec.functions[function_id].args]

        print('args:', args) if debug else None

        # AlphaStar use the raw actions
        func_call = A.FunctionCall.init_with_validation(function=function_id, arguments=args, raw=True)

        return func_call, units_args

    def action_to_func_call_HAS(self, action, action_spec, obs, use_random_args = False):
        # assert the action is single
        print('action_to_func_call_HAS:') if debug else None
        print('action:', action) if debug else None
        print('action.get_shape():', action.get_shape()) if debug else None

        action_id = action.action_type.item()
        print('action_id:', action_id) if debug else None

        function_id = L.action_type_index_map_human(action_id)

        #function_id = -1
        print('function_id:', function_id) if debug else None
        print("available_actions:", obs.available_actions) if debug else None
        print("function_id is available:", function_id in (obs.available_actions)) if debug else None

        if not function_id in (obs.available_actions):
            function_id = 0  # np.random.choice(obs.available_actions)

        #print('stop', stop)

        delay = action.delay.item()
        print('delay:', delay) if debug else None

        queue = action.queue.item()
        print('queue:', queue) if debug else None

        # we assume single inference
        units = action.units.cpu().detach().reshape(-1).numpy().tolist()
        print('units:', units) if debug else None

        target_unit = action.target_unit.item()  
        print('target_unit:', target_unit) if debug else None

        # we assume single inference
        target_location = action.target_location.cpu().detach().reshape(-1).numpy().tolist()
        print('target_location:', target_location) if debug else None

        need_args = action_spec.functions[function_id].args
        args = []

        def to_list(i):
            return [i]

        units_args = []    
        if not use_random_args:
            for arg in need_args:
                print("arg:", arg) if debug else None
                rand = [np.random.randint(0, size) for size in arg.sizes]

                if arg.name == 'queued':
                    size = arg.sizes[0]
                    if queue < 0 or queue > size - 1:
                        args.append(rand)
                        print("argument queue beyond the size!") if debug else None
                    else:
                        args.append(to_list(queue))

                elif arg.name == 'screen':
                    screen_args = []

                    # note target location for world is 256 x 256
                    # but in human space, we defualt set the screen size to 64 x 64
                    for val, size in zip(target_location, arg.sizes):
                        # so we default divide it by 4
                        val = int(val / 4)
                        if val < 0 or val > size - 1:
                            screen_args.append(np.random.randint(0, size))
                            print("argument world beyond the size!") if debug else None
                        elif val == 0:
                            screen_args.append(np.random.randint(0, size))
                            print("argument world is 0, we change it to random!") if debug else None
                        else:
                            screen_args.append(val)                        
                    print('screen_args:', screen_args) if debug else None

                    args.append(screen_args)

                elif arg.name == 'minimap':
                    minimap_args = []

                    # note target location for world is 256 x 256
                    # but in human space, we defualt set the minimap size to 64 x 64
                    for val, size in zip(target_location, arg.sizes):
                         # so we default divide it by 4
                        val = int(val / 4) 
                        if val < 0 or val > size - 1:
                            minimap_args.append(np.random.randint(0, size))
                            print("argument world beyond the size!") if debug else None
                        elif val == 0:
                            minimap_args.append(np.random.randint(0, size))
                            print("argument world is 0, we change it to random!") if debug else None
                        else:
                            minimap_args.append(val)                        
                    print('minimap_args:', minimap_args) if debug else None

                    args.append(minimap_args)
                else:
                    args.append([np.random.randint(0, size) for size in arg.sizes])

        else:
            print('use_random_args!') if debug else None
            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in action_spec.functions[function_id].args]

        print('args:', args) if debug else None

        # AlphaStar use the raw actions
        func_call = A.FunctionCall.init_with_validation(function=function_id, arguments=args, raw=False)

        return func_call, units_args

    def unroll_traj(self, state_all, initial_state, baseline_state=None, baseline_opponent_state=None):
        baseline_value_list, action_logits, _, _ = self.model.forward(state_all, batch_size=None, sequence_length=None, 
                                                                      hidden_state=initial_state, return_logits=True,
                                                                      baseline_state=baseline_state, 
                                                                      baseline_opponent_state=baseline_opponent_state,
                                                                      return_baseline=True)
        return baseline_value_list, action_logits

    def get_weights(self):
        if self.model is not None:
            return self.model.state_dict()
        else:
            return None

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
        return


def test():

    agent = Agent()

    batch_size = AHP.batch_size * AHP.sequence_length
    # dummy scalar list
    scalar_list = []

    agent_statistics = torch.ones(batch_size, SFS.agent_statistics)
    home_race = torch.randn(batch_size, SFS.home_race)
    away_race = torch.randn(batch_size, SFS.away_race)
    upgrades = torch.randn(batch_size, SFS.upgrades)
    enemy_upgrades = torch.randn(batch_size, SFS.upgrades)
    time = torch.randn(batch_size, SFS.time)

    available_actions = torch.randn(batch_size, SFS.available_actions)
    unit_counts_bow = torch.randn(batch_size, SFS.unit_counts_bow)
    mmr = torch.randn(batch_size, SFS.mmr)
    units_buildings = torch.randn(batch_size, SFS.units_buildings)
    effects = torch.randn(batch_size, SFS.effects)
    upgrade = torch.randn(batch_size, SFS.upgrade)

    beginning_build_order = torch.randn(batch_size, SCHP.count_beginning_build_order, 
                                        int(SFS.beginning_build_order / SCHP.count_beginning_build_order))
    last_delay = torch.randn(batch_size, SFS.last_delay)
    last_action_type = torch.randn(batch_size, SFS.last_action_type)
    last_repeat_queued = torch.randn(batch_size, SFS.last_repeat_queued)

    scalar_list.append(agent_statistics)
    scalar_list.append(home_race)
    scalar_list.append(away_race)
    scalar_list.append(upgrades)
    scalar_list.append(enemy_upgrades)
    scalar_list.append(time)

    scalar_list.append(available_actions)
    scalar_list.append(unit_counts_bow)
    scalar_list.append(mmr)
    scalar_list.append(units_buildings)
    scalar_list.append(effects)
    scalar_list.append(upgrade)

    scalar_list.append(beginning_build_order)
    scalar_list.append(last_delay)
    scalar_list.append(last_action_type)
    scalar_list.append(last_repeat_queued)

    # dummy entity list
    e_list = []
    e1 = Entity(115, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], 0, 100, 60, 50, 4, 8, 95, 0.2, 0.0, 0.0, 140, 60, 100,
                1, 123, 218, 3, True, False, True, True, False, 0, 0, 0, 0, 0, 0, 3.0, [2, 3], 2, 1, 0, True, False)
    e2 = Entity(1908, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], 2, 1500, 0, 200, 0, 4, 15, 0.5, 0.8, 0.5, 1500, 0, 250,
                2, 69, 7, 3, True, False, False, True, False, 0, 0, 0, 0, 10, 16, 0.0, [1], 1, 1, 0, False, False)
    e_list.append(e1)
    e_list.append(e2)

    # preprocess entity list
    entities_tensor = agent.model.preprocess_entity(e_list)
    print('entities_tensor.shape:', entities_tensor.shape) if debug else None
    batch_entities_tensor = torch.unsqueeze(entities_tensor, dim=0)
    batch_entities_list = []
    for i in range(batch_size):
        batch_entities_tensor_copy = batch_entities_tensor.detach().clone()
        batch_entities_list.append(batch_entities_tensor_copy)

    batch_entities_tensor = torch.cat(batch_entities_list, dim=0)
    print('batch_entities_tensor.shape:', batch_entities_tensor.shape) if debug else None

    # dummy map list
    map_list = []
    map_data_1 = torch.zeros(batch_size, 1, AHP.minimap_size, AHP.minimap_size)
    map_data_1_one_hot = L.to_one_hot(map_data_1, 2)
    print('map_data_1_one_hot.shape:', map_data_1_one_hot.shape) if debug else None

    map_list.append(map_data_1)
    map_data_2 = torch.zeros(batch_size, 17, AHP.minimap_size, AHP.minimap_size)
    map_list.append(map_data_2)
    map_data = torch.cat(map_list, dim=1)

    state = MsState(entity_state=batch_entities_tensor, statistical_state=scalar_list, map_state=map_data)

    print("Multi-source state:", state) if 1 else None

    action = agent.action_by_state(state)
    print("action is:", action) if 1 else None

    action_logits, actions, hidden_state = agent.action_logits_by_state(state)
    print("action_logits is:", action_logits) if 1 else None
    print("actions is:", actions) if 1 else None

    if debug:
        print("This is a test!")


if __name__ == '__main__':
    test()
