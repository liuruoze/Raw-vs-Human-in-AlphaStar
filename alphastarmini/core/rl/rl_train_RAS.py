#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train for RL by interacting with the environment"

import os

USED_DEVICES = "0"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import traceback
from time import time, sleep
import threading

from pysc2.env.sc2_env import Race

from alphastarmini.core.rl import utils as U
from alphastarmini.core.rl.learner import Learner
from alphastarmini.core.rl.actor_RAS import ActorLoopRAS

# below packages are for test
from alphastarmini.core.ma.league import League
from alphastarmini.core.ma.coordinator import Coordinator

import param as P

__author__ = "Ruo-Ze Liu"

debug = False


def test(on_server=False, replay_path=None):
    # model path
    ACTOR_NUMS = P.actor_nums
    RESTORE = False
    model_name = "rl_21-08-04_11-08-04.pkl"

    league = League(
        initial_agents={
            race: U.get_reinforcement_agent(race, restore=RESTORE, model_name=model_name)
            for race in [Race.protoss]
        },
        main_players=1, 
        main_exploiters=0,
        league_exploiters=0)

    coordinator = Coordinator(league)
    learners = []
    actors = []

    for idx in range(league.get_learning_players_num()):
        player = league.get_learning_player(idx)
        learner = Learner(player, max_time_for_training=60 * 60 * 24)
        learners.append(learner)
        actors.extend([ActorLoopRAS(player, coordinator, replay_path=replay_path, record=z == 0) for z in range(ACTOR_NUMS)])

    threads = []
    for l in learners:
        l.start()
        threads.append(l.thread)
        sleep(1)
    for a in actors:
        a.start()
        threads.append(a.thread)
        sleep(1)

    try: 
        # Wait for training to finish.
        for t in threads:
            t.join()
    except Exception as e: 
        print("Exception Handled in Main, Detials of the Exception:", e)
