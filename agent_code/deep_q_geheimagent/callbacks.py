import numpy as np
from time import sleep
import math
from settings import e
import random
from settings import settings
from agent_code.deep_q_geheimagent.model import SecretNetwork
from agent_code.deep_q_geheimagent.ReplayMemory import *
import matplotlib.pyplot as plt

#import pytorch
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as T
import torch.optim as optim
import torch.nn.functional as F


def setup(agent):
    print('setup...')
    agent.reward = 0
    agent.final_reward = 0
    agent.current_reward = 0
    agent.max_idx = 0
    agent.reward_dict = {
        'MOVED_LEFT'     :  0,
        'MOVED_RIGHT'    :  0,
        'MOVED_UP'       :  0, 
        'MOVED_DOWN'     :  0,
        'WAITED'         :  0,
        'INTERRUPTED'    : -200,
        'INVALID_ACTION' : -500,

        'BOMB_DROPPED'   :  0,
        'BOMB_EXPLODED'  :  0,

        'CRATE_DESTROYED':  100,
        'COIN_FOUND'     :  200,
        'COIN_COLLECTED' :  500,

        'KILLED_OPPONENT':  1000,
        'KILLED_SELF'    : -5000,

        'GOT_KILLED'     : -500,
        'OPPONENT_ELIMINATED' : 10,
        'SURVIVED_ROUND' : 1000
        }
    # if gpu is to be used
    agent.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.actions = settings['actions']
    agent.INPUT_SHAPE = (1, 17, 17)
    agent.BATCH_SIZE = 16
    agent.GAMMA = 0.5
    agent.EPS_START = 0.9
    agent.EPS_END = 0.05
    agent.EPS_DECAY = 200
    agent.TARGET_UPDATE = 10
    agent.episodes = 0
    agent.policy_net = SecretNetwork(agent.INPUT_SHAPE)
    agent.target_net = SecretNetwork(agent.INPUT_SHAPE)
    agent.criterion = nn.MSELoss()
    agent.optimizer = optim.RMSprop(agent.policy_net.parameters())
    agent.memory = ReplayMemory(10000)
    agent.optimizer.zero_grad()  
    agent.episode_durations = []
    agent.states = []
    agent.actions_done = []
    agent.reward_received = []
    try:
        agent.policy_net.load_state_dict(torch.load('agent_code/deep_q_geheimagent/deep_geheimagent.pth'))
        agent.policy_net.eval()
        print('model loaded')
    except FileNotFoundError:
        pass
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.target_net.eval()

    agent.steps_done = 0
    plt.ion()
    print('deep geheimagent setup done')
    return

def act(agent):
    # get state from state function and convert it to NN format
    state = create_map(agent)
    agent.states.append(state)


    sample = random.random()
    eps_threshold = agent.EPS_END + (agent.EPS_START - agent.EPS_END) * math.exp(-1. * agent.steps_done / agent.EPS_DECAY)
    agent.steps_done += 1
    # Pass state through network and get action out
    agent.res = agent.policy_net(state)
    # do mapping and random sample
    max_idx = int(agent.res.max(1)[1])
    if sample < eps_threshold:
        max_idx = np.random.randint(6)

    agent.next_action = agent.actions[max_idx]
    agent.idx_action = max_idx

    action = torch.tensor([[max_idx]], device=agent.device, dtype=torch.long)

    agent.actions_done.append(action)

    return

def reward_update(agent):
    current_events = np.array(agent.events)
    reward_gained = 0
    for i in current_events:
        reward_gained += agent.reward_dict[e._fields[i]]
    
    reward = torch.tensor([reward_gained], device=agent.device, dtype=torch.float)
    
    agent.reward_received.append(reward)

    return

def create_map(agent):
    game_state = agent.game_state

    #get all information
    step = game_state['step']
    arena = game_state['arena']
    pos = game_state['self']
    others = game_state['others']
    bombs = game_state['bombs']
    explosions = game_state['explosions']
    coins = game_state['coins']
    
    #place agent and other events in one array
    arena[pos[0],pos[1]] = 100
    arena = arena - 55*explosions
    for i in range(len(others)):
        arena[others[i][0],others[i][1]] = 10

    for i in range(len(bombs)):
        arena[bombs[i][0],bombs[i][1]] = (-50 * (3 - bombs[i][2]))
        if ((pos[0] == bombs[i][0]) and (pos[0] == bombs[i][0])):
            arena[pos[0],pos[1]] = -1000

    for i in range(len(coins)):
        arena[coins[i][0],coins[i][1]] = 30

    stateNN = np.expand_dims(arena,axis = 0)
    assert(np.shape(stateNN) == agent.INPUT_SHAPE)
    stateNN = np.expand_dims(stateNN,axis = 0)
    #try:
    #    stateNN = torch.from_numpy(stateNN).float()
    #except ValueError:
    #    stateNN = torch.from_numpy(np.flip(stateNN,axis=0).copy()).float()

    return torch.from_numpy(stateNN).float()
    

def end_of_episode(agent):
    agent.episodes += 1
    current_events = np.array(agent.events)
    reward_gained = 0
    for i in current_events:
        reward_gained += agent.reward_dict[e._fields[i]]
    reward = torch.tensor([reward_gained], device=agent.device, dtype=torch.float)

    agent.episode_durations.append(agent.game_state['step'])
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(agent.episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated 

    agent.reward_received.append(reward_gained)   

    n = len(agent.states)

    for i in range(n-1):
        state = agent.states[i]
        next_state = agent.states[i+1]
        reward = agent.reward_received[i]
        action = agent.actions_done[i]
        agent.memory.push(state, action, next_state, reward)

        #state, action = augment_data_transpose(agent.states[i],agent.actions_done[i])
        #next_state, _ = augment_data_transpose(agent.states[i+1],0)
        #reward = agent.reward_received[i]
        #memory.push(state, action, next_state, reward)

        #state, action = augment_data_flipud(agent.states[i],agent.actions_done[i])
        #next_state, _ = augment_data_flipud(agent.states[i+1],0)
        #reward = agent.reward_received[i]
        #memory.push(state, action, next_state, reward)

        #state, action = augment_data_fliplr(agent.states[i],agent.actions_done[i])
        #next_state, _ = augment_data_fliplr(agent.states[i+1],0)
        #reward = agent.reward_received[i]
        #memory.push(state, action, next_state, reward)


    sample_size = int(len(agent.memory) / 2)
    transitions = agent.memory.sample(sample_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = agent.policy_net(state_batch).gather(1, action_batch)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=agent.device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(sample_size, device=agent.device)
    next_state_values[non_final_mask] = agent.target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * agent.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    agent.optimizer.zero_grad()
    loss.backward()
    for param in agent.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    agent.optimizer.step()

    print('final learn')

    # Update the target network, copying all weights and biases in DQN
    if agent.episodes % agent.TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('update target_net')

    #print('final_reward = {}'.format(agent.final_reward))
    torch.save(agent.target_net.state_dict(), 'agent_code/deep_q_geheimagent/deep_geheimagent.pth') 
    agent.states = []
    agent.actions_done = []
    agent.reward_received = []
    print('model saved')
    return

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
