
import numpy as np
from time import sleep
from settings import e

#import pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

def setup(agent):
    agent.reward = 0
    agent.reward_dict = {
        'MOVED_LEFT'     :  0,
        'MOVED_RIGHT'    :  0,
        'MOVED_UP'       :  0, 
        'MOVED_DOWN'     :  0,
        'WAITED'         : -1,
        'INTERRUPTED'    : -2,
        'INVALID_ACTION' : -3,

        'BOMB_DROPPED'   :  2,
        'BOMB_EXPLODED'  :  0,

        'CRATE_DESTROYED':  10,
        'COIN_FOUND'     :  2,
        'COIN_COLLECTED' :  50,

        'KILLED_OPPONENT':  100,
        'KILLED_SELF'    : -300,

        'GOT_KILLED'     : -300,
        'OPPONENT_ELIMINATED' : 100,
        'SURVIVED_ROUND' : 1000
    }
    agent.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
    agent.gamma = 0.99


    return

def act(agent):
    agent.logger.info('Pick action according to pressed key')
    agent.next_action = agent.game_state['user_input']

    map = create_np_map(agent.game_state)
    return

def reward_update(agent):
    current_events = np.array(agent.events)
    reward_gained = 0
    for i in current_events:
        reward_gained += agent.reward_dict[e._fields[i]]
        
    agent.reward = agent.gamma * agent.reward + reward_gained

    print('reward = {}'.format(agent.reward))
    return

def create_np_map(state):
    #get all information
    step = state['step']
    arena = state['arena']
    pos = state['self']
    others = state['others']
    bombs = state['bombs']
    explosions = state['explosions']
    coins = state['coins']
    
    #place agent and other events in one array
    arena[pos[0],pos[1]] = 100
    arena = arena - 55*explosions
    for i in range(len(others)):
        arena[others[i][0],others[i][1]] = 10

    for i in range(len(bombs)):
        arena[bombs[i][0],bombs[i][1]] = (-10 * (3 - bombs[i][2]))

    for i in range(len(coins)):
        arena[coins[i][0],coins[i][1]] = 30

    return arena
    

def end_of_episode(agent):
    last_event = np.array(agent.events)
    reward_gained = 0
    for i in last_event:
        reward_gained += agent.reward_dict[e._fields[i]]
        
    agent.reward = 0.9 * agent.reward + reward_gained
    agent.final_reward = agent.reward
    agent.reward = 0.0

    print('final_reward = {}'.format(agent.final_reward))
    return

def learn(agent):
    pass
