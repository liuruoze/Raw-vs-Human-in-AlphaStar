#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the actor in the actor-learner mode in the IMPALA architecture "

# modified from AlphaStar pseudo-code
import traceback
from time import time, sleep, strftime, localtime
import threading
import random
from datetime import datetime

import torch
from torch.optim import Adam

from tensorboardX import SummaryWriter

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat
from pysc2.env.sc2_env import Agent, Race, Bot, Difficulty, BotBuild
from pysc2.lib import features as F
from pysc2.lib import actions as A

from alphastarmini.core.rl.env_utils import SC2Environment, get_env_outcome
from alphastarmini.core.rl.utils import Trajectory, get_supervised_agent
from alphastarmini.core.rl.learner import Learner
from alphastarmini.core.rl import utils as U

from alphastarmini.lib import utils as L

# below packages are for test
from alphastarmini.core.ma.league import League
from alphastarmini.core.ma.coordinator import Coordinator

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import Training_Races as TR
from alphastarmini.lib.hyper_parameters import AlphaStar_Raw_Interface_Format_Params as ARIFP

__author__ = "Ruo-Ze Liu"

debug = False

STEP_MUL = 8   # 1
GAME_STEPS_PER_EPISODE = 4500  # 4500    # 18000
MAX_EPISODES = 100     # 100   

INTERVAL = 8

# gpu setting
ON_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")
#torch.backends.cudnn.enabled = False


class ActorLoopRAS:
    """A single actor loop that generates trajectories.

    We don't use batched inference here, but it was used in practice.
    TODO: implement the batched version
    """

    def __init__(self, player, coordinator, max_time_for_training = 60 * 60 * 24,
                 max_time_per_one_opponent=60 * 60 * 2,
                 max_frames_per_episode=22.4 * 60 * 15, max_frames=22.4 * 60 * 60 * 24, 
                 max_episodes=MAX_EPISODES, use_replay_expert_reward=True,
                 replay_path="data/Replays/filtered_replays_1/", replay_version='3.16.1',
                 record=False):

        self.player = player
        self.player.add_actor(self)
        if ON_GPU:
            self.player.agent.agent_nn.to(DEVICE)

        # below code is not used because we only can create the env when we know the opponnet information (e.g., race)
        # AlphaStar: self.environment = SC2Environment()

        self.coordinator = coordinator
        self.max_time_for_training = max_time_for_training
        self.max_time_per_one_opponent = max_time_per_one_opponent
        self.max_frames_per_episode = max_frames_per_episode
        self.max_frames = max_frames
        self.max_episodes = max_episodes

        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True                            # Daemonize thread

        self.is_running = True
        self.is_start = False

        now = datetime.now()
        summary_path = "./log/" + now.strftime("%Y%m%d-%H%M%S") + "_reward/"
        self.writer = SummaryWriter(summary_path)
        self.batch_iter = 0
        self.record = record

    def start(self):
        self.is_start = True
        self.thread.start()

    def get_func(self, env, home_obs, player_memory, action_spec):
        # run_loop: actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        player_step = self.player.agent.step_logits_RAS(home_obs, player_memory)
        function_call, select_units, player_action, player_logits, player_new_memory = player_step
        print("function:", function_call.function) if debug else None

        obs = home_obs.observation
        raw_units = obs["raw_units"]

        our_unit_list = []
        mineral_unit_list = []
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
            # include the units of Neutral   
            if u.alliance == 3:
                if u.display_type == 1:
                    if u.x < 40 and u.y < 50:
                        if u.mineral_contents > 0:
                            mineral_unit_list.append(u)

        our_unit_list.extend(nexus_list)

        def myFunc(e):
            return e.tag
        probe_list.sort(reverse=False, key=myFunc)
        our_unit_list.extend(probe_list)

        random_index = random.randint(0, len(our_unit_list) - 1)
        if len(select_units) > 0:
            predict_index = select_units[0]
        else:
            predict_index = -1

        if len(mineral_unit_list) > 0:
            max_mineral_contents = mineral_unit_list[0].mineral_contents
            max_mineral_tag = mineral_unit_list[0].tag

            for u in mineral_unit_list:
                if u.mineral_contents > max_mineral_contents:
                    max_mineral_contents = u.mineral_contents
                    max_mineral_tag = u.tag

        if predict_index >= len(our_unit_list) or predict_index < 0:
            unit_index = random_index
        else:
            unit_index = predict_index

        the_tag = our_unit_list[unit_index].tag

        # we change pysc2 action to sc2 action, for replace the unit tag
        sc2_action = env._features[0].transform_action(obs, function_call)                         
        print("sc2_action before transformed:", sc2_action) if debug else None

        # if len(nexus_list) > 0:
        #     nexus_tag = nexus_list[0].tag
        #     print("nexus_tag", nexus_tag) if debug else None
        #     if function_call.function == 64:
        #         the_tag = nexus_tag

        # if len(idle_probe_list) > 0:
        #     idle_probe_tag = idle_probe_list[0].tag
        #     print("idle_probe_tag", idle_probe_tag) if debug else None
        #     if function_call.function == 35:
        #         the_tag = idle_probe_tag
        # elif len(probe_list) > 0:
        #     probe_tag = probe_list[0].tag
        #     print("probe_tag", probe_tag) if debug else None
        #     if function_call.function == 35:
        #         the_tag = probe_tag

        if sc2_action.HasField("action_raw"):
            raw_act = sc2_action.action_raw
            if raw_act.HasField("unit_command"):
                uc = raw_act.unit_command
                # to judge a repteated field whether has 
                # use the following way
                if len(uc.unit_tags) != 0:
                    # can not assign, must use unit_tags[:]=[xx tag]
                    print("the_tag", the_tag) if debug else None
                    uc.unit_tags[:] = [the_tag]
                # we use fixed target unit tag only for Harvest_Gather_unit action
                if uc.HasField("target_unit_tag"):
                    uc.target_unit_tag = max_mineral_tag

        print("sc2_action after transformed:", sc2_action) if debug else None

        env_actions = [sc2_action]

        player_action_spec = action_spec[0]
        action_masks = U.get_mask_raw(player_action, player_action_spec)
        print('player_action', player_action, 'action_masks', action_masks) if debug else None

        return env_actions, player_action, player_logits, action_masks, player_new_memory

    def get_reward(self, home_obs, home_next_obs):
        r1 = 0  # home_obs.observation.score_cumulative.collection_rate_minerals  # total_value_units
        r2 = home_obs.observation.player.food_workers  # food_cap  # food_workers  # minerals                   
        r = r2
        print("r", r) if debug else None

        # .minerals  # food_workers
        next_r1 = 0  # home_next_obs.observation.score_cumulative.collection_rate_minerals  # total_value_units
        next_r2 = home_next_obs.observation.player.food_workers
        next_r = next_r2
        print("next_r", next_r) if debug else None

        diff = next_r - r
        print("diff", diff) if debug else None

        # we use the change of minerals as reward 
        reward = float(diff)  # home_next_obs.reward
        print("reward: ", reward) if debug else None
        return reward, next_r

    # background
    def run(self):
        try:
            self.is_running = True
            """A run loop to have agents and an environment interact."""
            total_frames = 0
            total_episodes = 0
            results = []

            start_time = time()
            print("start_time before training:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_time)))

            while time() - start_time < self.max_time_for_training:
                agents = [self.player]

                with self.create_env_one_player(self.player) as env:

                    # set the obs and action spec
                    observation_spec = env.observation_spec()
                    action_spec = env.action_spec()

                    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
                        agent.setup(obs_spec, act_spec)

                    print('player:', self.player) if debug else None

                    trajectory = []
                    start_time = time()  # in seconds.
                    print("start_time before reset:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_time)))

                    # one opponent match (may include several games) defaultly lasts for no more than 2 hour
                    while time() - start_time < self.max_time_per_one_opponent:

                        # Note: the pysc2 environment don't return z

                        # AlphaStar: home_observation, away_observation, is_final, z = env.reset()
                        total_episodes += 1
                        print("total_episodes:", total_episodes)

                        timesteps = env.reset()
                        for a in agents:
                            a.reset()

                        [home_obs] = timesteps
                        is_final = home_obs.last()

                        player_memory = self.player.agent.initial_state()

                        episode_frames = 0

                        # default outcome is 0 (means draw)
                        outcome = 0

                        # in one episode (game)
                        start_episode_time = time()  # in seconds.
                        print("start_episode_time before is_final:", strftime("%Y-%m-%d %H:%M:%S", localtime(start_episode_time)))

                        interval = INTERVAL
                        agent_last_obs = None
                        while not is_final:
                            total_frames += 1
                            episode_frames += 1

                            if episode_frames % interval == 0:
                                agent_obs = home_obs
                                func_result = self.get_func(env, home_obs, player_memory, action_spec)
                                env_actions, player_action, player_logits, action_masks, player_new_memory = func_result
                            else:
                                func_noop = A.FunctionCall.init_with_validation(function=0, arguments=[], raw=True)
                                env_actions = [func_noop]

                            timesteps = env.step(env_actions)
                            [home_next_obs] = timesteps

                            game_loop = home_obs.observation.game_loop[0]
                            is_final = home_next_obs.last()

                            if episode_frames % interval == 0:   
                                if agent_last_obs is not None:

                                    reward, next_r = self.get_reward(agent_last_obs, agent_obs)

                                    rl_reward = reward

                                    traj_step = Trajectory(
                                        observation=home_obs.observation,
                                        opponent_observation=home_obs.observation,
                                        memory=player_memory,
                                        z=None,
                                        masks=action_masks,
                                        action=player_action,
                                        behavior_logits=player_logits,
                                        teacher_logits=None,      
                                        is_final=is_final,                                          
                                        reward=rl_reward,
                                        build_order=[],
                                        z_build_order=[],  # we change it to the sampled build order
                                        unit_counts=[],
                                        z_unit_counts=[],  # we change it to the sampled unit counts
                                        game_loop=game_loop,
                                    )
                                    trajectory.append(traj_step)

                                player_memory = tuple(h.detach() for h in player_new_memory)

                            home_obs = home_next_obs

                            if episode_frames % interval == 0:
                                agent_last_obs = agent_obs

                            # if is_final:
                            #     final_return = next_r
                            #     print("final_return: ", final_return) if 1 else None

                            #     self.writer.add_scalar('Return', final_return, self.batch_iter)
                            #     self.batch_iter += 1

                            #     print("Return: {:.6f}.".format(final_return)) if debug else None
                            #     results.append(final_return)

                            if len(trajectory) >= AHP.sequence_length:                    
                                trajectories = U.stack_namedtuple(trajectory)

                                if self.player.learner is not None:
                                    if self.player.learner.is_running:
                                        print("Learner send_trajectory!") if debug else None
                                        self.player.learner.send_trajectory(trajectories)
                                        trajectory = []
                                    else:
                                        print("Learner stops!")

                                        print("Actor also stops!")
                                        return

                            # use max_frames to end the loop
                            # whether to stop the run
                            if self.max_frames and total_frames >= self.max_frames:
                                print("Beyond the max_frames, return!")
                                return

                            # use max_frames_per_episode to end the episode
                            if self.max_frames_per_episode and episode_frames >= self.max_frames_per_episode:
                                print("Beyond the max_frames_per_episode, break!")
                                break

                        if is_final:
                            print('append the last!') if 1 else None

                            if agent_last_obs is not None and agent_obs is not None:
                                agent_obs = home_obs

                                func_result = self.get_func(env, home_obs, player_memory, action_spec)
                                env_actions, player_action, player_logits, action_masks, player_new_memory = func_result

                                # units = player_logits.units.reshape(-1).detach().clone()
                                # units_probs = torch.nn.functional.softmax(units)
                                # print('units_probs:', units_probs)
                                # self.writer.add_histogram('Probs/units_probs', units_probs, self.batch_iter, bins='fd')

                                reward, r = self.get_reward(agent_last_obs, agent_obs)

                                rl_reward = reward

                                traj_step = Trajectory(
                                    observation=home_obs.observation,
                                    opponent_observation=home_obs.observation,
                                    memory=player_memory,
                                    z=None,
                                    masks=action_masks,
                                    action=player_action,
                                    behavior_logits=player_logits,
                                    teacher_logits=None,      
                                    is_final=is_final,                                          
                                    reward=rl_reward,
                                    build_order=[],
                                    z_build_order=[],  # we change it to the sampled build order
                                    unit_counts=[],
                                    z_unit_counts=[],  # we change it to the sampled unit counts
                                    game_loop=game_loop,
                                )
                                trajectory.append(traj_step)

                                final_return = next_r
                                print("final_return: ", final_return) if 1 else None

                                if self.record:
                                    self.writer.add_scalar('Return', final_return, self.batch_iter)
                                    self.batch_iter += 1

                                print("Return: {:.6f}.".format(final_return)) if debug else None
                                results.append(final_return)

                        # use max_frames_per_episode to end the episode
                        if self.max_episodes and total_episodes >= self.max_episodes:
                            print("Beyond the max_episodes, return!")

                            if self.record:
                                print("results: ", results) if 1 else None
                                with open("results.txt", "w") as f:
                                    f.write(", ".join(str(item) for item in results))

                            return

                # close the replays

        except Exception as e:
            print("ActorLoop.run() Exception cause return, Detials of the Exception:", e)
            print(traceback.format_exc())

        finally:
            self.is_running = False

    # create env function
    def create_env_one_player(self, player, game_steps_per_episode=GAME_STEPS_PER_EPISODE, 
                              step_mul=STEP_MUL, version=None, 
                              # the map should be the same as in the expert replay
                              map_name="AbyssalReef", random_seed=1):

        player_aif = AgentInterfaceFormat(**ARIFP._asdict())
        agent_interface_format = [player_aif]

        # create env
        print('map name:', map_name) 
        print('player.name:', player.name)
        print('player.race:', player.race)

        sc2_computer = Bot([Race.terran],
                           Difficulty.very_easy,
                           [BotBuild.random])

        env = SC2Env(map_name=map_name,
                     players=[Agent(player.race, player.name),
                              sc2_computer],
                     step_mul=step_mul,
                     game_steps_per_episode=game_steps_per_episode,
                     agent_interface_format=agent_interface_format,
                     version=version,
                     random_seed=random_seed)

        return env
