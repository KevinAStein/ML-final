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
        'MOVED_LEFT'     :  -1,
        'MOVED_RIGHT'    :  -1,
        'MOVED_UP'       :  -1, 
        'MOVED_DOWN'     :  -1,
        'WAITED'         :  -10,
        'INTERRUPTED'    : -100,
        'INVALID_ACTION' : -200,

        'BOMB_DROPPED'   :  -2,
        'BOMB_EXPLODED'  :  0,

        'CRATE_DESTROYED':  10,
        'COIN_FOUND'     :  200,
        'COIN_COLLECTED' :  1000,

        'KILLED_OPPONENT':  100,
        'KILLED_SELF'    :  -100,

        'GOT_KILLED'     : -100,
        'OPPONENT_ELIMINATED' : 0,
        'SURVIVED_ROUND' : 1000
        }
    # if gpu is to be used
    agent.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.actions = settings['actions']
    agent.INPUT_SHAPE = (1, 17, 17)
    agent.BATCH_SIZE = 128
    agent.LR = 0.0001
    agent.GAMMA = 0.8
    agent.EPS_START = 0.05
    agent.EPS_END = 0.05
    agent.EPS_DECAY = 1000
    agent.TARGET_UPDATE = 10
    agent.episodes = 0
    agent.policy_net = SecretNetwork(agent.INPUT_SHAPE)
    agent.target_net = SecretNetwork(agent.INPUT_SHAPE)
    agent.optimizer = optim.Adam(agent.policy_net.parameters(),  lr=agent.LR)
    agent.memory = ReplayMemory(10000)
    agent.optimizer.zero_grad()  
    agent.episode_durations = []
    agent.episode_reward = []
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
    # Pass state through network and get action out
    action = agent.policy_net(state).max(1)[1].view(1, 1)

    if sample < eps_threshold:
        action = torch.tensor([[random.randrange(6)]], device=agent.device, dtype=torch.long)
    
    agent.actions_done.append(action)

    agent.next_action = agent.actions[int(action)]

    return

def reward_update(agent):
    current_events = np.array(agent.events)
    reward_gained = 0
    for i in current_events:
        reward_gained += agent.reward_dict[e._fields[i]]
    
    reward = torch.tensor([reward_gained], device=agent.device, dtype=torch.float)
    
    agent.reward_received.append(reward)
    #print('return rew:' , reward)

    return
    

def end_of_episode(agent):
    agent.episodes += 1
    reward_update(agent) 

    final_rew = 0
    for i in range(len(agent.reward_received)):
        final_rew = agent.reward_received[i] + final_rew
    agent.episode_durations.append(agent.game_state['step'])
    agent.episode_reward.append(final_rew )


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
    state = agent.states[n-1]
    next_state = None
    reward = agent.reward_received[n-1]
    action = agent.actions_done[n-1]
    agent.memory.push(state, action, next_state, reward)
 

    sample_size = int(len(agent.memory))
    if sample_size > agent.BATCH_SIZE:

        transitions = agent.memory.sample(agent.BATCH_SIZE)
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
        next_state_values = torch.zeros(agent.BATCH_SIZE, device=agent.device)
        next_state_values[non_final_mask] = agent.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        agent.optimizer.zero_grad()
        expected_state_action_values = (next_state_values * agent.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model

        loss.backward()
        for param in agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        agent.optimizer.step()
        agent.steps_done += 1

        # Update the target network, copying all weights and biases in DQN
        if agent.episodes % agent.TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            agent.target_net.eval()
            torch.save(agent.target_net.state_dict(), 'agent_code/deep_q_geheimagent/deep_geheimagent.pth') 
            print('update target_net')

    #print('final_reward = {}'.format(agent.final_reward))
    agent.states = []
    agent.actions_done = []
    agent.reward_received = []
    print('model saved')
    return


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
    
    #place agent and other events in one array
    arena[pos[0],pos[1]] = 2
    #add explosions
    arena = arena + 9*explosions

    #add bombs
    for i in range(len(bombs)):
        arena[bombs[i][0],bombs[i][1]] = 14 - bombs[i][2]
        #make sure we know if we sit on a bomb
        if ((pos[0] == bombs[i][0]) and (pos[1] == bombs[i][1])):
            arena[pos[0],pos[1]] = 10
    

    #add others
    for i in range(len(others)):
        arena[others[i][0],others[i][1]] = 3
        #add others sitting on a bomb
        for j in range(len(bombs)):
            if ((others[i][0] == bombs[j][0]) and (others[i][1] == bombs[j][1])):
                arena[others[i][0],others[i][1]] = 4
    
    #add coins
    for i in range(len(coins)):
        arena[coins[i][0],coins[i][1]] = 5

    stateNN = np.expand_dims(arena,axis = 0)
    assert(np.shape(stateNN) == agent.INPUT_SHAPE)
    stateNN = np.expand_dims(stateNN,axis = 0)
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
