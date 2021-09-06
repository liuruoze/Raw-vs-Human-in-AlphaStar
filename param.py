# param for some configs, for ease use of changing different servers
# also ease of use for experiments

'''whether is running on server, on server meaning use GPU with larger memoary'''
on_server = False
#on_server = False

'''The replay path'''
replay_path = "data/Replays/filtered_replays_1/"
#replay_path = "/home/liuruoze/data4/mini-AlphaStar/data/filtered_replays_1/"
#replay_path = "/home/liuruoze/mini-AlphaStar/data/filtered_replays_1/"

'''The mini scale used in hyperparameter'''
#Mini_Scale = 4
#Mini_Scale = 8
Mini_Scale = 16 * 4 * 4

actor_nums = 1

restore = False

use_raw_action = False
action_size_raw = 3
action_size_human = 5

if use_raw_action:
    action_size = action_size_raw
else:
    action_size = action_size_human
