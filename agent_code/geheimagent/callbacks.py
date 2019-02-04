
import numpy as np
from time import sleep
from settings import e


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
    return

def reward_update(agent):
    current_events = np.array(agent.events)
    reward_gained = 0
    for i in current_events:
        reward_gained += agent.reward_dict[e._fields[i]]
        
    agent.reward = agent.gamma * agent.reward + reward_gained

    print('reward = {}'.format(agent.reward))
    return

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