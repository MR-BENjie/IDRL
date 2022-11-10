"""Battle
"""

import argparse
import os
import tensorflow as tf
import numpy as np
from collections import Counter

import rlcard
from rlcard.models.model import Model
from .battle_model.algo import spawn_ai


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def _cards2array(cards):
    Card2Column = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7,
                   'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12}

    NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                     1: np.array([1, 0, 0, 0]),
                     2: np.array([1, 1, 0, 0]),
                     3: np.array([1, 1, 1, 0]),
                     4: np.array([1, 1, 1, 1])}
    if cards == 'pass':
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(cards)
    for card, num_times in counter.items():
        if card == 'B':
            jokers[0] = 1
        elif card == 'R':
            jokers[1] = 1
        else:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
    return np.concatenate((matrix.flatten('F'), jokers))

class MFRLModel(Model):
    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('red_10')


        player= (replayDeepAgent(env,model_idx=1999999, use_onnx=False, num_actions=env.num_actions))
        self.agent = player
        self.num_players = env.num_players

    def agents(self,rela_state=None,danger_state=None,role_id=0):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        if rela_state and danger_state:
            return 0.0,0.0,1000,self.agent
        else:
            return [self.agent, self.agent, self.agent, self.agent]
class replayDeepAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, env,model_idx=1999,use_onnx=False,num_actions=0):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw =  True
        self.num_actions = num_actions
        self.use_onnx = use_onnx

        tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        tf_config.gpu_options.allow_growth = True

        handles = [0]
        main_model_dir = os.path.join(BASE_DIR, 'pretrained/')

        sess = tf.Session(config=tf_config)
        models = spawn_ai('mfq', sess, env, handles[0], 'mfq-me', 400)
        sess.run(tf.global_variables_initializer())
        models.load(main_model_dir, step=model_idx)
        self.model = models

    @staticmethod
    def step(self,state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (str): The action predicted (randomly chosen) by the pretrained  agent
        '''
        env_output = state
        former_act_prob = np.zeros((len(env_output['raw_legal_actions']), 1))
        feature_obs = np.zeros((len(env_output['raw_legal_actions']), 54))
        for index, x in enumerate(env_output['raw_legal_actions']):
            feature_obs[index:] = _cards2array(x)

            # take actions for every model
        x_shape = env_output['x_batch'].shape
        state = np.reshape(env_output['x_batch'], (x_shape[0], 1, 11, 54))

        # former_act_prob = np.tile(former_act_prob, (env_output.ndim, 1))
        acts = self.model.act(state=[state, feature_obs], prob=former_act_prob, eps=1.0)
        action = env_output['raw_legal_actions'][acts[0]]

        best_action_confidence = [1,1,1]

        best_action = [action, '1', '1']
        return  action, best_action, best_action_confidence

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        choice_action,best_action, best_action_confidence = self.step(self,state = state)
        info = {}
        info['probs'] = {}
        for action,action_c in zip(best_action,best_action_confidence):
            info['probs'][action] = action_c
        return choice_action, info
