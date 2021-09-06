#!/usr/bin/env python
# -*- coding: utf-8 -*-

" The code for the learner in the actor-learner mode in the IMPALA architecture"

# modified from AlphaStar pseudo-code
import os
import traceback
import random
from time import time, sleep, strftime, localtime
import threading
import itertools
from datetime import datetime

import torch
from torch.optim import Adam, RMSprop

from tensorboardX import SummaryWriter

from alphastarmini.core.rl.rl_loss import loss_function

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import RL_Training_Hyper_Parameters as THP

__author__ = "Ruo-Ze Liu"

debug = False


# model path
MODEL = "rl"
MODEL_PATH = "./model/"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + strftime("%y-%m-%d_%H-%M-%S", localtime()))


class Learner:
    """Learner worker that updates agent parameters based on trajectories."""

    def __init__(self, player, max_time_for_training=60 * 3):
        self.player = player
        self.player.set_learner(self)

        self.trajectories = []

        # AlphaStar code
        #self.optimizer = AdamOptimizer(learning_rate=3e-5, beta1=0, beta2=0.99, epsilon=1e-5)

        # PyTorch code
        self.optimizer = Adam(self.get_parameters(), 
                              lr=THP.learning_rate, betas=(THP.beta1, THP.beta2), 
                              eps=THP.epsilon, weight_decay=THP.weight_decay)

        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True                            # Daemonize thread

        self.max_time_for_training = max_time_for_training
        self.is_running = False

        now = datetime.now()
        summary_path = "./log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        self.writer = SummaryWriter(summary_path)

        self.batch_iter = 0

        self.use_random_shuffle = False

        if self.use_random_shuffle:
            self.replay_buffer_weight = 2
        else:
            self.replay_buffer_weight = 1

    def get_parameters(self):
        return self.player.agent.get_parameters()

    def send_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

    def update_parameters(self):
        if self.use_random_shuffle:
            # random shuffle the list
            random.shuffle(self.trajectories)

        trajectories = self.trajectories[:AHP.batch_size]
        self.trajectories = self.trajectories[AHP.batch_size:]

        agent = self.player.agent

        print("begin backward") if debug else None

        # a error: cudnn RNN backward can only be called in training mode
        agent.agent_nn.model.train()
        #torch.backends.cudnn.enabled = False

        self.optimizer.zero_grad()

        loss_all, loss_list = loss_function(agent, trajectories)
        print("loss_all:", loss_all) if 1 else None

        writer = self.writer
        batch_iter = self.batch_iter
        print("One batch loss_all: {:.6f}.".format(loss_all.item()))
        writer.add_scalar('OneBatch/loss_all', loss_all.item(), batch_iter)

        if True:
            [lambda_loss, pg_loss, loss_upgo, loss_ent] = loss_list

            print("One batch lambda_loss: {:.6f}.".format(lambda_loss.item()))
            writer.add_scalar('OneBatch/lambda_loss', lambda_loss.item(), batch_iter)

            print("One batch pg_loss: {:.6f}.".format(pg_loss.item()))
            writer.add_scalar('OneBatch/pg_loss', pg_loss.item(), batch_iter)

            print("One batch loss_upgo: {:.6f}.".format(loss_upgo.item()))
            writer.add_scalar('OneBatch/loss_upgo', loss_upgo.item(), batch_iter)

            print("One batch loss_ent: {:.6f}.".format(loss_ent.item()))
            writer.add_scalar('OneBatch/loss_ent', loss_ent.item(), batch_iter)

        loss_all.backward()

        # print('selected_units_head.conv_1.weight.grad', agent.agent_nn.model.selected_units_head.conv_1.weight.grad) if 0 else None
        # print('selected_units_head.conv_1.weight.grad.shape', agent.agent_nn.model.selected_units_head.conv_1.weight.grad.shape) if 1 else None
        # print('selected_units_head.conv_1.weight.grad.mean', agent.agent_nn.model.selected_units_head.conv_1.weight.grad.mean()) if 1 else None
        # print('selected_units_head.conv_1.weight.grad.std', agent.agent_nn.model.selected_units_head.conv_1.weight.grad.std()) if 1 else None

        # print('selected_units_head.fc_1.weight.grad', agent.agent_nn.model.selected_units_head.fc_1.weight.grad) if 1 else None
        # print('selected_units_head.fc_1.weight.grad.shape', agent.agent_nn.model.selected_units_head.fc_1.weight.grad.shape) if 1 else None
        # print('selected_units_head.fc_1.weight.grad.mean', agent.agent_nn.model.selected_units_head.fc_1.weight.grad.mean()) if 1 else None
        # print('selected_units_head.fc_1.weight.grad.std', agent.agent_nn.model.selected_units_head.fc_1.weight.grad.std()) if 1 else None

        # print('selected_units_head.func_embed.weight.grad', agent.agent_nn.model.selected_units_head.func_embed.weight.grad) if 1 else None
        # print('selected_units_head.func_embed.weight.grad.shape', agent.agent_nn.model.selected_units_head.func_embed.weight.grad.shape) if 1 else None
        # print('selected_units_head.func_embed.weight.grad.mean', agent.agent_nn.model.selected_units_head.func_embed.weight.grad.mean()) if 1 else None
        # print('selected_units_head.func_embed.weight.grad.std', agent.agent_nn.model.selected_units_head.func_embed.weight.grad.std()) if 1 else None

        # print('selected_units_head.fc_2.weight.grad', agent.agent_nn.model.selected_units_head.fc_2.weight.grad) if 0 else None
        # print('selected_units_head.fc_2.weight.grad.shape', agent.agent_nn.model.selected_units_head.fc_2.weight.grad.shape) if 1 else None
        # print('selected_units_head.fc_2.weight.grad.mean', agent.agent_nn.model.selected_units_head.fc_2.weight.grad.mean()) if 1 else None
        # print('selected_units_head.fc_2.weight.grad.std', agent.agent_nn.model.selected_units_head.fc_2.weight.grad.std()) if 1 else None

        # print('selected_units_head.fc_3.weight.grad', agent.agent_nn.model.selected_units_head.fc_3.weight.grad) if 0 else None
        # print('selected_units_head.fc_3.weight.grad.shape', agent.agent_nn.model.selected_units_head.fc_3.weight.grad.shape) if 1 else None
        # print('selected_units_head.fc_3.weight.grad.mean', agent.agent_nn.model.selected_units_head.fc_3.weight.grad.mean()) if 1 else None
        # print('selected_units_head.fc_3.weight.grad.std', agent.agent_nn.model.selected_units_head.fc_3.weight.grad.std()) if 1 else None

        # print('selected_units_head.small_lstm.weight_ih_l0.grad', agent.agent_nn.model.selected_units_head.small_lstm.weight_ih_l0.grad) if 1 else None
        # print('selected_units_head.small_lstm.weight_ih_l0.grad.shape', agent.agent_nn.model.selected_units_head.small_lstm.weight_ih_l0.grad.shape) if 1 else None
        # print('selected_units_head.small_lstm.weight_ih_l0.grad.mean', agent.agent_nn.model.selected_units_head.small_lstm.weight_ih_l0.grad.mean()) if 1 else None
        # print('selected_units_head.small_lstm.weight_ih_l0.grad.std', agent.agent_nn.model.selected_units_head.small_lstm.weight_ih_l0.grad.std()) if 1 else None

        # print('action_type_head.glu_1.fc_2.weight.grad', agent.agent_nn.model.action_type_head.glu_1.fc_2.weight.grad) if 0 else None
        # print('action_type_head.glu_1.fc_2.weight.grad.shape', agent.agent_nn.model.action_type_head.glu_1.fc_2.weight.grad.shape) if 1 else None
        # print('action_type_head.glu_1.fc_2.weight.grad.mean', agent.agent_nn.model.action_type_head.glu_1.fc_2.weight.grad.mean()) if 1 else None
        # print('action_type_head.glu_1.fc_2.weight.grad.std', agent.agent_nn.model.action_type_head.glu_1.fc_2.weight.grad.std()) if 1 else None

        # if self.batch_iter % 10 == 0:
        #     writer.add_histogram('units/tensor', agent.agent_nn.model.selected_units_head.fc_3.weight.grad, batch_iter)
        #     writer.add_scalar('units/mean', agent.agent_nn.model.selected_units_head.fc_3.weight.grad.mean(), batch_iter)
        #     writer.add_scalar('units/std', agent.agent_nn.model.selected_units_head.fc_3.weight.grad.std(), batch_iter)

        #     writer.add_histogram('action/tensor', agent.agent_nn.model.action_type_head.glu_1.fc_2.weight.grad, batch_iter)
        #     writer.add_scalar('action/mean', agent.agent_nn.model.action_type_head.glu_1.fc_2.weight.grad.mean(), batch_iter)
        #     writer.add_scalar('action/std', agent.agent_nn.model.action_type_head.glu_1.fc_2.weight.grad.std(), batch_iter)

        #print("stop", stop)

        self.optimizer.step()

        print("end backward") if debug else None

        torch.save(agent.agent_nn.model, SAVE_PATH + "" + ".pkl")

        agent.steps += AHP.batch_size * AHP.sequence_length  # num_steps(trajectories)

        self.batch_iter += 1

        # self.player.agent.set_weights(self.optimizer.minimize(loss))

    def start(self):
        self.thread.start()

    # background
    def run(self):
        try:
            start_time = time()
            self.is_running = True

            while time() - start_time < self.max_time_for_training:
                try:
                    # if at least one actor is running, the learner would not stop
                    actor_is_running = False
                    if len(self.player.actors) == 0:
                        actor_is_running = True

                    for actor in self.player.actors:
                        if actor.is_start:
                            actor_is_running = actor_is_running | actor.is_running
                        else:
                            actor_is_running = actor_is_running | 1

                    if actor_is_running:
                        print('learner trajectories size:', len(self.trajectories))

                        if len(self.trajectories) >= self.replay_buffer_weight * AHP.batch_size:
                            print("learner begin to update parameters")
                            self.update_parameters()

                        sleep(1)
                    else:
                        print("Actor stops!")

                        print("Learner also stops!")
                        return

                except Exception as e:
                    print("Learner.run() Exception cause break, Detials of the Exception:", e)
                    print(traceback.format_exc())
                    break

        except Exception as e:
            print("Learner.run() Exception cause return, Detials of the Exception:", e)

        finally:
            self.is_running = False


def test(on_server):
    pass
