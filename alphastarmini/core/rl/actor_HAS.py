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
from alphastarmini.lib.hyper_parameters import AlphaStar_Human_Interface_Format_Params as AHIFP

__author__ = "Ruo-Ze Liu"

debug = False

STEP_MUL = 8   # 1
GAME_STEPS_PER_EPISODE = 4500  # 4500    # 18000
MAX_EPISODES = 100     # 100   

# gpu setting
ON_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")
#torch.backends.cudnn.enabled = False


class ActorLoopHAS:
    """A single actor loop that generates trajectories.

    We don't use batched inference here, but it was used in practice.
    TODO: implement the batched version
    """

    def __init__(self, player, coordinator, max_time_for_training = 60 * 60 * 24,
                 max_time_per_one_opponent=60 * 60 * 2,
                 max_frames_per_episode=22.4 * 60 * 15, max_frames=22.4 * 60 * 60 * 24, 
                 max_episodes=MAX_EPISODES, use_replay_expert_reward=True,
                 replay_path="data/Replays/filtered_replays_1/", replay_version='3.16.1'):

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

    def start(self):
        self.is_start = True
        self.thread.start()

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

                        while not is_final:
                            total_frames += 1
                            episode_frames += 1

                            # run_loop: actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
                            player_step = self.player.agent.step_logits_HAS(home_obs, player_memory)
                            function_call, select_units, player_action, player_logits, player_new_memory = player_step
                            print("function_call:", function_call) if 1 else None

                            env_actions = [function_call]

                            player_action_spec = action_spec[0]
                            action_masks = U.get_mask_human(player_action, player_action_spec)
                            z = None

                            timesteps = env.step(env_actions)
                            [home_next_obs] = timesteps

                            # print the observation of the agent
                            # print("home_obs.observation:", home_obs.observation)

                            game_loop = home_obs.observation.game_loop[0]
                            print("game_loop", game_loop) if debug else None

                            '''
                            minerals = home_obs.observation.player.minerals
                            next_minerals = home_next_obs.observation.player.minerals
                            diff_minerals = next_minerals - minerals
                            print("diff_minerals", diff_minerals) if debug else None
                            '''

                            is_final = home_next_obs.last()

                            food_workers = home_obs.observation.player.food_workers
                            next_food_workers = home_next_obs.observation.player.food_workers
                            diff_food_workers = next_food_workers - food_workers
                            print("diff_food_workers", diff_food_workers) if debug else None

                            # we use the change of minerals as reward 
                            reward = diff_food_workers  # home_next_obs.reward
                            print("reward: ", reward) if debug else None

                            # note, original AlphaStar pseudo-code has some mistakes, we modified 
                            # them here
                            traj_step = Trajectory(
                                observation=home_obs.observation,
                                opponent_observation=home_obs.observation,
                                memory=player_memory,
                                z=z,
                                masks=action_masks,
                                action=player_action,
                                behavior_logits=player_logits,
                                teacher_logits=None,      
                                is_final=is_final,                                          
                                reward=reward,
                                build_order=[],
                                z_build_order=[],  # we change it to the sampled build order
                                unit_counts=[],
                                z_unit_counts=[],  # we change it to the sampled unit counts
                                game_loop=game_loop,
                            )
                            trajectory.append(traj_step)

                            player_memory = tuple(h.detach() for h in player_new_memory)

                            home_obs = home_next_obs

                            if is_final:
                                final_return = home_next_obs.observation.player.food_workers
                                print("final_return: ", final_return) if 1 else None

                                self.writer.add_scalar('Return', final_return, self.batch_iter)
                                self.batch_iter += 1

                                print("Return: {:.6f}.".format(final_return))
                                results.append(final_return)

                            if len(trajectory) >= AHP.sequence_length:                    
                                trajectories = U.stack_namedtuple(trajectory)

                                if self.player.learner is not None:
                                    if self.player.learner.is_running:
                                        print("Learner send_trajectory!")
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

                        # use max_frames_per_episode to end the episode
                        if self.max_episodes and total_episodes >= self.max_episodes:
                            print("Beyond the max_episodes, return!")
                            print("results: ", results) if 1 else None

                            with open("minerals.txt", "w") as f:
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

        player_aif = AgentInterfaceFormat(**AHIFP._asdict())
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
