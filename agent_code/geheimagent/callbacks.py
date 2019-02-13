import numpy as np
from time import sleep
from settings import e
import random
from settings import settings
from agent_code.geheimagent.model import SecretNetwork
from agent_code.geheimagent.ReplayMemory import *

#import pytorch
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms


def setup(agent):
    agent.reward = 0
    agent.final_reward = 0
    agent.current_reward = 0
    agent.max_idx = 0
    agent.reward_dict = {
        'MOVED_LEFT'     :  0,
        'MOVED_RIGHT'    :  0,
        'MOVED_UP'       :  0, 
        'MOVED_DOWN'     :  0,
        'WAITED'         :  -10,
        'INTERRUPTED'    : -200,
        'INVALID_ACTION' : -500,

        'BOMB_DROPPED'   :  0,
        'BOMB_EXPLODED'  :  40,

        'CRATE_DESTROYED':  100,
        'COIN_FOUND'     :  200,
        'COIN_COLLECTED' :  500,

        'KILLED_OPPONENT':  1000,
        'KILLED_SELF'    : -100,

        'GOT_KILLED'     : -500,
        'OPPONENT_ELIMINATED' : 10,
        'SURVIVED_ROUND' : 10
    }
    agent.actions = settings['actions']
    agent.rows = settings['rows']
    agent.cols = settings['cols']
    agent.INPUT_SHAPE = (1, 17, 17)
    agent.gamma = 0.8
    agent.model = SecretNetwork(agent.INPUT_SHAPE)
    agent.criterion = nn.MSELoss()
    agent.optimizer = torch.optim.Adam(agent.model.parameters())
    agent.optimizer.zero_grad()  
    agent.states = []
    agent.actions_done = []
    agent.reward_received = []
    print('model loaded')
    try:
        agent.model.load_state_dict(torch.load('geheimagent.pth'))
        agent.model.eval()
        print('model loaded')
    except FileNotFoundError:
        pass

    return

def act(agent):
    agent.logger.info('Pick action according to pressed key')

    # get state from state function and convert it to NN format
    state = create_np_map(agent.game_state)
    agent.states.append(state)
    stateNN = np.expand_dims(state,axis = 0)
    assert(np.shape(stateNN) == agent.INPUT_SHAPE)
    stateNN = np.expand_dims(stateNN,axis = 0)
    try:
        stateNN = torch.from_numpy(stateNN).float()
    except ValueError:
        stateNN = torch.from_numpy(np.flip(stateNN,axis=0).copy()).float()

    # Gradienten löschen.
    agent.optimizer.zero_grad() 

    # Pass state through network and get action out
    agent.res = agent.model(stateNN)


    # do mapping
    max_idx = int(agent.res.max(1)[1])
    if (np.random.rand() < 0.2):
        max_idx = np.random.randint(6)
        #print('random action: ', max_idx)

    agent.next_action = agent.actions[max_idx]
    agent.idx_action = max_idx
    agent.actions_done.append(max_idx)

    print ('Action: ' ,agent.next_action)

    return

def reward_update(agent):
    current_events = np.array(agent.events)
    reward_gained = 0
    for i in current_events:
        reward_gained += agent.reward_dict[e._fields[i]]
    agent.reward_received.append(reward_gained)   
    rew = torch.zeros(1,6)
    #for i in range(6):
    #    rew[0][i] = agent.res[0][i]
    #rew[0][agent.idx_action] = float(reward_gained)

    # Loss auswerten.
    #loss = agent.criterion(rew, agent.res)
    #print(agent.res)

    # Loss zurück propagieren.
    #loss.backward()

    # Optimieren.
    #agent.optimizer.step()

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
        if ((pos[0] == bombs[i][0]) and (pos[0] == bombs[i][0])):
            arena[pos[0],pos[1]] = -100

    for i in range(len(coins)):
        arena[coins[i][0],coins[i][1]] = 30

    return arena
    

def end_of_episode(agent):
    #print('final_reward = {}'.format(agent.final_reward))
    current_events = np.array(agent.events)
    reward_gained = 0
    for i in current_events:
        reward_gained += agent.reward_dict[e._fields[i]]
    agent.reward_received.append(reward_gained)
   
    n = len(agent.states)
    final_rew = 0
    for i in range(n-1):
        final_rew = final_rew + agent.reward_received[i] 
    print('finale reward: ', final_rew)

    #for i in range(n-1):
    #    agent.reward_received[n-i-2]  = agent.reward_received[n-i-2]  + agent.gamma * agent.reward_received[n-i-1] 

    memory = ReplayMemory(10000)

    for i in range(n-1):
        state = agent.states[i]
        next_state = agent.states[i+1]
        reward = agent.reward_received[i]
        action = agent.actions_done[i]
        memory.push(state, action, next_state, reward)

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


    n = int(len(memory) / 2)
    transitions = memory.sample(n)
    batch = Transition(*zip(*transitions))


    for i in range(n-1):

        # Gradienten löschen.
        agent.optimizer.zero_grad() 

        state = batch.state[i]
        stateNN = np.expand_dims(state,axis = 0)
        assert(np.shape(stateNN) == agent.INPUT_SHAPE)
        stateNN = np.expand_dims(stateNN,axis = 0)
        try:
            stateNN = torch.from_numpy(stateNN).float()
        except ValueError:
            stateNN = torch.from_numpy(np.flip(stateNN,axis=0).copy()).float()

        # Pass state through network and get action out
        res = agent.model(stateNN)

        rew = torch.zeros(1,6)
        for i in range(6):
            rew[0][i] = res[0][i]
        rew[0][batch.action[i]] = float(batch.reward[i])

        # Loss auswerten.
        loss = agent.criterion(rew, res)

        # Loss zurück propagieren.
        loss.backward()

        # Optimieren.
        agent.optimizer.step()

    print('final learn')

    #print('final_reward = {}'.format(agent.final_reward))
    torch.save(agent.model.state_dict(), 'geheimagent.pth') 
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
