import numpy as np
from time import sleep
import math
from settings import e
import random
from settings import settings
from agent_code.AC_geheimagent.model import ActorCritic
from agent_code.AC_geheimagent.ReplayMemory import *
import matplotlib.pyplot as plt

#import pytorch
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as T
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


def setup(agent):
    print('setup...')
    agent.reward = 0
    agent.final_reward = 0
    agent.current_reward = 0
    agent.max_idx = 0
    agent.reward_dict = {
        'MOVED_LEFT'     :  -1,
        'MOVED_RIGHT'    :  -1,
        'MOVED_UP'       :  -1, 
        'MOVED_DOWN'     :  -1,
        'WAITED'         :  -1,
        'INTERRUPTED'    : -100,
        'INVALID_ACTION' : -200,

        'BOMB_DROPPED'   :  -10,
        'BOMB_EXPLODED'  :  0,

        'CRATE_DESTROYED':  10,
        'COIN_FOUND'     :  200,
        'COIN_COLLECTED' :  1000,

        'KILLED_OPPONENT':  100,
        'KILLED_SELF'    :  -500,

        'GOT_KILLED'     : -500,
        'OPPONENT_ELIMINATED' : 2000,
        'SURVIVED_ROUND' : 1000
        }
    # if gpu is to be used
    agent.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.actions = settings['actions']
    agent.N_actions = len(agent.actions)

    agent.INPUT_SHAPE = (13, 17, 17)
    agent.LR = 0.00001
    agent.GAMMA = 0.8
    agent.model = ActorCritic(agent.INPUT_SHAPE)
    agent.optimizer = optim.Adam(agent.model.parameters(),  lr=agent.LR)
    agent.memory = ReplayMemory(10000)
    agent.episode_durations = []
    agent.logprobs = []
    agent.state_values = []
    agent.episode_reward = []
    agent.states = []
    agent.actions_done = []
    agent.reward_received = []
    agent.reward = 0
    agent.rev = 0

    try:
        agent.model.load_state_dict(torch.load('agent_code/AC_geheimagent/AC_geheimagent.pth'))
        agent.model.eval()
        print('model loaded')
    except FileNotFoundError:
        pass

    agent.steps_done = 0
    plt.ion()
    print('AC geheimagent setup done')
    return

def act(agent):
    # get state from state function and convert it to NN format
    state = create_map(agent)
    agent.states.append(state)

    state_value = agent.model.get_state_value(state)
    action_probs = agent.model.get_action_probs(state)
    action_distribution = Categorical(action_probs)
    action = action_distribution.sample()
        
    agent.logprobs.append(action_distribution.log_prob(action))
    agent.state_values.append(state_value)
    agent.next_action = agent.actions[action.item()]
    #print('action', agent.next_action)

    return

def reward_update(agent):
    agent.rev = agent.rev + 1
    if agent.rev == 1:
        return
    current_events = np.array(agent.events)
    reward_gained = 0
    for i in current_events:
        reward_gained += agent.reward_dict[e._fields[i]]
    
    reward = torch.tensor([reward_gained], device=agent.device, dtype=torch.float)
    
    agent.reward_received.append(reward)
    #print('reward', reward)
    return
    

def end_of_episode(agent):
    reward_update(agent)

    final_rew = 0
    for i in range(len(agent.reward_received)):
        final_rew = agent.reward_received[i] + final_rew
    agent.episode_durations.append(agent.game_state['step'] * 100)
    agent.episode_reward.append(final_rew)


    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(agent.episode_durations, dtype=torch.float)
    reward_t = torch.tensor(agent.episode_reward, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    plt.plot(reward_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())        
        means_rew = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        means_rew = torch.cat((torch.zeros(99), means_rew))
        plt.plot(means_rew.numpy())   
    plt.pause(0.001)  # pause a bit so that plots are updated 

    agent.optimizer.zero_grad()
    loss = calcLoss(agent)
    loss.backward()
    agent.optimizer.step() 

    torch.save(agent.model.state_dict(), 'agent_code/AC_geheimagent/AC_geheimagent.pth') 
    print('model saved')
    agent.states = []
    agent.actions_done = []
    agent.reward_received = []
    agent.logprobs = []
    agent.state_values = []
    return

def calcLoss(agent):
    # calculating discounted rewards:
    rewards = []
    policy_losses = []
    value_losses = []
    dis_reward = 0
    for reward in agent.reward_received[::-1]:
        dis_reward = reward + agent.GAMMA * dis_reward
        rewards.insert(0, dis_reward)
            
    # normalizing the rewards:
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std())
    
    loss = 0
    for logprob, value, reward in zip(agent.logprobs, agent.state_values, rewards):
        advantage = reward  - value.item()
        policy_losses.append(-logprob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([reward])))
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    return loss


def create_map(agent):

    # -1 wall
    #  0 free
    #  1 crate
    #  2 self
    #  3 opponents
    #  4 opponents + bomb
    #  5 coin
    #  9 explosion
    #  10 self + bomb
    #  12 bomb (countdown = 2)
    #  13 bomb (countdown = 1)
    #  14 bomb (countdown = 0)
    #  15 bomb (countdown = -1)

    game_state = agent.game_state

    #get all information
    step = game_state['step']
    arena = game_state['arena']
    pos = game_state['self']
    others = game_state['others']
    bombs = game_state['bombs']
    explosions = game_state['explosions']
    coins = game_state['coins']
    

    pos_np = np.zeros_like(arena)    
    #place agent
    pos_np[pos[0],pos[1]] = 1
    others_np = np.zeros_like(arena)
    #add others
    for i in range(len(others)):
        others_np[others[i][0],others[i][1]] = 1

    bomb_4 = np.zeros_like(arena)
    #add bombs
    for i in range(len(bombs)):
        if (bombs[i][2] == 4):
            bomb_4[bombs[i][0],bombs[i][1]] = 1
    bomb_3 = np.zeros_like(arena)
    for i in range(len(bombs)):
        if (bombs[i][2] == 3):
            bomb_3[bombs[i][0],bombs[i][1]] = 1
    bomb_2 = np.zeros_like(arena)
    for i in range(len(bombs)):
        if (bombs[i][2] == 2):
            bomb_2[bombs[i][0],bombs[i][1]] = 1
    bomb_1 = np.zeros_like(arena)
    for i in range(len(bombs)):
        if (bombs[i][2] == 1):
            bomb_1[bombs[i][0],bombs[i][1]] = 1
    bomb_0 = np.zeros_like(arena)
    for i in range(len(bombs)):
        if (bombs[i][2] == 0):
            bomb_0[bombs[i][0],bombs[i][1]] = 1
    #add explosions
    explosion_2 = (explosions == 2).astype(int)
    explosion_1 = (explosions == 1).astype(int)

    #add coins
    coins_np = np.zeros_like(arena)
    for i in range(len(coins)):
        coins_np[coins[i][0],coins[i][1]] = 1

    #add environment
    wall = (arena == -1).astype(int)
    crate = (arena == 1).astype(int)
    free = (arena == 0).astype(int)
    
    karte = np.stack([pos_np, others_np, bomb_4, bomb_3, bomb_2, bomb_1, bomb_0, explosion_2, explosion_1, coins_np, wall, crate, free])

    assert(np.shape(karte) == agent.INPUT_SHAPE)
    stateNN = np.expand_dims(karte,axis = 0)
    return torch.from_numpy(stateNN).float()

def augment_data_transpose(state, action):
    res_state = np.transpose(state)
    res_action = 0
    if action == 0:
        res_action = 2
    elif action == 2:
        res_action = 0
    elif action == 1:
        res_action = 3
    elif action == 3:
        res_action = 1
    else:
        res_action = action
    return res_state, res_action

def augment_data_flipud(state, action):
    res_state = np.flipud(state)
    res_action = 0
    if action == 0:
        res_action = 1
    elif action == 0:
        res_action = 1    
    else:
        res_action = action
    return res_state, res_action

def augment_data_fliplr(state, action):
    res_state = np.fliplr(state)
    res_action = 0
    if action == 2:
        res_action = 3
    if action == 3:
        res_action = 2
    else:
        res_action = action
    return res_state, res_action

def learn(agent):
    pass
