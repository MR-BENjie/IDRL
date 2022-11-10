import numpy as np
import os
import sys
os.environ['CUDA_ENABLE_DEVICES'] = '0'
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(FILE_PATH, ''))
sys.path.append(ROOT_PATH)
from env import get_combinations_nosplit, get_combinations_recursive

import torch
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack
from torch import nn
import rlcard
from rlcard.games.doudizhu.utils import CARD_TYPE, INDEX
from rlcard.models.model import Model
from tensorpack import *
from tensorpack.train.model_desc import ModelDesc
from tensorpack.tfutils import (
    varreplace, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.utils import logger
import tensorflow.compat.v1 as tf
import tensorflow as tf2
from conditional import conditional
from collections import Counter

from CDQN_card import Card, action_space, action_space_onehot60, Category, \
    CardGroup, augment_action_space_onehot60, augment_action_space, clamp_action_idx



weight_path = os.path.join(FILE_PATH,"pretrained/CDQN_pretrained/model-500000")

pretrained_dir = ''

class CDQNModel(Model):
    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('red_10')
        self.agent = {}


        pretrained_dir_ = os.path.join(pretrained_dir, '')
        player= (
                replayDeepAgent(0, pretrained_dir_, use_onnx=False, num_actions=env.num_actions))
        self.agent = player

        self.num_players = env.num_players

    def agents(self,rela_state=None,danger_state=None,role_id=0):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        if rela_state and danger_state:
            return 0.0,0.0,1000,self.agent
        else:
            return [self.agent, self.agent, self.agent, self.agent]
class replayDeepAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, position, pretrained_dir, use_onnx=False,num_actions=0):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw =  True
        self.num_actions = num_actions
        self.use_onnx = use_onnx
        agent_names = ['agent%d'%i for i in range(4)]
        model = CDQNNet(agent_names, (1000, 21, 256 + 256 * 2 + 120), 'Double', (1000, 21), 0.99)
        self.predictor=[]
        for i in range(4):
            self.predictor.append(Predictor(OfflinePredictor(PredictConfig(
                model=model,
                session_init=SaverRestore(weight_path),
                input_names=['agent'+str(i)+'/state', 'agent'+str(i)+'_comb_mask',
                             'agent'+str(i)+'/fine_mask'],
                output_names=['agent'+str(i)+'/Qvalue'])), num_actions=(1000, 21)))

    @staticmethod
    def step(self,state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (str): The action predicted (randomly chosen) by the pretrained  agent
        '''

        raw_obs = state['raw_obs']
        c_handcards = raw_obs['current_hand']
        handcards = []
        if c_handcards != 'pass':
            for x in c_handcards:
                if x == 'T':
                    handcards.append("10")
                else:
                    handcards.append(x)

        last_three_cards = self.get_last_three_cards(raw_obs)
        prob_state = self.get_prob_state(raw_obs)
        intention = self.predictor[raw_obs['self']].predict(handcards, last_three_cards, prob_state)

        best_action_confidence = [1,1,1]

        card_list_red10=[ '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K','A', '2']
        card_rank_red10 = {}
        rank_card_red10 = {}
        for index,i in enumerate(card_list_red10):
            card_rank_red10[i] = index
            rank_card_red10[index] = i
        choice_action = []
        for x in intention:
            if x == '10':
                choice_action.append(card_rank_red10['T'])
            else:
                choice_action.append(card_rank_red10[x])
        if len(choice_action) == 0:
            choice_action = 'pass'
        else:
            final_action = ''
            choice_action.sort()
            for x in choice_action:
                final_action+=rank_card_red10[x]
            choice_action = final_action
        best_action = [choice_action, '1', '1']
        return  choice_action, best_action, best_action_confidence

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



    def get_last_three_cards(self,state):
        last_three_cards = []
        for i in range(3):
            teammate_next_id = (state['self'] - 1 - i) % 4
            teammate_next_action = []
            for i, action, _ in reversed(state['trace']):
                if i == teammate_next_id:
                    teammate_next_action = action
                    break
            t_card = []
            if teammate_next_action != 'pass':
                for l_t_c in teammate_next_action:
                    if l_t_c == 'T':
                        t_card.append("10")
                    else:
                        t_card.append(l_t_c)
            last_three_cards.append(t_card)
        return last_three_cards
    def get_prob_state(self,state):
        raw_other_cards = state['others_hand']
        other_cards = []
        for x in raw_other_cards:
            if x == 'T':
                other_cards.append("10")
            else:
                other_cards.append(x)
        total_num_cards = len(other_cards)
        def char2onehot60(cards):
            counts = Counter(cards)
            onehot = np.zeros(60, dtype=np.int32)
            for x in cards:
                subvec = np.zeros(4)
                subvec[:counts[x]] = 1
                onehot[Card.cards.index(x) * 4:Card.cards.index(x) * 4 + 4] = subvec
            return onehot
        other_cards = char2onehot60(other_cards)
        num_cards_left = state['num_cards_left']
        teammate_prob_state = []
        for i in range(3):
            teammate_next_id = (state['self'] - 1 - i) % 4
            teammate_next_num_cards = num_cards_left[teammate_next_id]
            teammate_prob_state.append(other_cards*(teammate_next_num_cards/float(total_num_cards)))
        prob_state  = np.concatenate([teammate_prob_state[0],teammate_prob_state[1],teammate_prob_state[2]])
        return prob_state

def res_fc_block(inputs, units, stack=3):
    residual = inputs
    for i in range(stack):
        residual = FullyConnected('fc%d' % i, residual, units, activation=tf.nn.relu)
    x = inputs
    # x = FullyConnected('fc', x, units, activation=tf.nn.relu)
    if inputs.shape[1] != units:
        x = FullyConnected('fc', x, units, activation=tf.nn.relu)

    #layer_norma = tf2.keras.layers.LayerNormalization(axis=-1)
    return tf2.contrib.layers.layer_norm(residual + x, scale=False)

class CDQNNet(ModelDesc):
    learning_rate = 1e-4
    cost_weights = [1., 1., 1.]

    def __init__(self, agent_names, state_shape, method, num_actions, gamma):
        self.agent_names = agent_names
        self.state_shape = state_shape
        self.method = method
        self.num_actions = num_actions
        self.gamma = gamma

    def inputs(self):
        # Use a combined state for efficiency.
        # The first h channels are the current state, and the last h channels are the next state.
        res = []
        for name in self.agent_names:
            res.extend([tf.placeholder(tf.float32,
                               (None, 2, self.state_shape[0], self.state_shape[1], self.state_shape[2]),
                               name + '_joint_state'),
                tf.placeholder(tf.int64, (None,), name + '_action'),
                tf.placeholder(tf.float32, (None,), name + '_reward'),
                tf.placeholder(tf.bool, (None,), name + '_isOver'),
                tf.placeholder(tf.bool, (None,), name + '_comb_mask'),
                tf.placeholder(tf.bool, (None, 2, None), name + '_joint_fine_mask')])
        return res

    # input B * C * D
    # output B * C * 1
    @auto_reuse_variable_scope
    def _get_DQN_prediction_comb(self, state):
        shape = state.shape.as_list()
        net = tf.reshape(state, [-1, shape[-1]])
        units = [512, 256, 128]
        for i, unit in enumerate(units):
            with tf.variable_scope('block%i' % i):
                net = res_fc_block(net, unit)
        l = net

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, 1)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, 1)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.reshape(Q, [-1, shape[1], 1])

    # input B * N * D
    # output B * N * 1
    def _get_DQN_prediction_fine(self, state):
        shape = state.shape.as_list()
        net = tf.reshape(state, [-1, shape[-1]])
        units = [512, 256, 128]
        for i, unit in enumerate(units):
            with tf.variable_scope('block%i' % i):
                net = res_fc_block(net, unit)
        l = net

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, 1)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, 1)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.reshape(Q, [-1, shape[1], 1])

    def _get_global_feature(self, joint_state):
        shape = joint_state.shape.as_list()
        net = tf.reshape(joint_state, [-1, shape[-1]])
        units = [256, 512, 1024]
        for i, unit in enumerate(units):
            with tf.variable_scope('block%i' % i):
                net = res_fc_block(net, unit)
        net = tf.reshape(net, [-1, shape[1], shape[2], units[-1]])
        return tf.reduce_max(net, [2])

    # decorate the function
    # output : B * A

    def get_DQN_prediction(self, joint_state, comb_mask, fine_mask):
        with tensorpack.argscope([tensorpack.FullyConnected], kernel_initializer=tf.glorot_uniform_initializer()):
            batch_size = tf.shape(joint_state)[0]
            with tf.variable_scope('dqn_global'):
                global_feature = self.get_global_feature(joint_state)

            comb_mask_idx = tf.cast(tf.where(comb_mask), tf.int32)
            with tf.variable_scope('dqn_comb'):
                q_comb = self._get_DQN_prediction_comb(tf.gather(global_feature, comb_mask_idx[:, 0]))
            q_comb = tf.squeeze(q_comb, -1)
            q_comb = tf.scatter_nd(comb_mask_idx, q_comb, tf.stack([batch_size, q_comb.shape[1]]))

            fine_mask_idx = tf.cast(tf.where(tf.logical_not(comb_mask)), tf.int32)
            state_fine = tf.concat([tf.tile(tf.expand_dims(global_feature, 2), [1, 1, joint_state.shape.as_list()[2], 1]), joint_state], -1)
            state_fine = tf.gather(state_fine[:, 0, :, :], fine_mask_idx[:, 0])
            with tf.variable_scope('dqn_fine'):
                q_fine = self._get_DQN_prediction_fine(state_fine)
            q_fine = tf.squeeze(q_fine, -1)
            q_fine = tf.scatter_nd(fine_mask_idx, q_fine, tf.stack([batch_size, q_fine.shape[1]]))

            larger_dim = max(joint_state.shape.as_list()[1], joint_state.shape.as_list()[2])
            padding_np = np.zeros([1, larger_dim], dtype=np.float32)
            padding_np[0, min(joint_state.shape[1], joint_state.shape[2]):] = -1e5
            padding = tf.convert_to_tensor(padding_np)
            # padding = tf.Variable(initial_value=padding_np, trainable=False, name='padding')
            padding = tf.tile(padding, tf.stack(
                [tf.shape(fine_mask_idx if joint_state.shape[1] > joint_state.shape[2] else comb_mask_idx)[0], 1]))
            padding = tf.scatter_nd(fine_mask_idx if joint_state.shape[1] > joint_state.shape[2] else comb_mask_idx,
                                    padding, tf.stack([batch_size, larger_dim]))
            # padding = tf.Print(padding, [padding], summarize=100)
            q = tf.add(tf.pad(q_comb, [[0, 0], [0, larger_dim - q_comb.shape.as_list()[1]]]) + tf.pad(q_fine, [[0, 0], [0, larger_dim - q_fine.shape.as_list()[ 1]]]),
                          padding)

            # q[tf.where(tf.logical_not(fine_mask))] = -1e5
            return tf.add(q, -tf.cast(tf.logical_not(fine_mask), dtype=tf.float32) * 1e5, name='Qvalue')

    # input :B * COMB * N * D
    # output : B * COMB * D'
    @auto_reuse_variable_scope
    def get_global_feature(self, joint_state):
        return self._get_global_feature(joint_state)

    # joint state: B * 2 * COMB * N * D for now, D = 256
    # dynamic action range
    def build_graph(self, *args):
        costs = []
        for i, name in enumerate(self.agent_names):
            joint_state, action, reward, isOver, comb_mask, joint_fine_mask = args[i * 6:(i + 1) * 6]
            with tf.variable_scope(name):
                with conditional(name is None, varreplace.freeze_variables()):
                    state = tf.identity(joint_state[:, 0, :, :, :], name='state')
                    fine_mask = tf.identity(joint_fine_mask[:, 0, :], name='fine_mask')
                    self.predict_value = self.get_DQN_prediction(state, comb_mask, fine_mask)
                    if not get_current_tower_context().is_training:
                        continue

                    # reward = tf.clip_by_value(reward, -1, 1)
                    next_state = tf.identity(joint_state[:, 1, :, :, :], name='next_state')
                    next_fine_mask = tf.identity(joint_fine_mask[:, 1, :], name='next_fine_mask')
                    action_onehot = tf.one_hot(action, self.num_actions, 1.0, 0.0)

                    pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)  # N,
                    max_pred_reward = tf.reduce_mean(tf.reduce_max(
                        self.predict_value, 1), name='predict_reward')
                    summary.add_moving_summary(max_pred_reward)

                    with tf.variable_scope('target'), varreplace.freeze_variables(skip_collection=True):
                        # we are alternating between comb and fine states
                        targetQ_predict_value = self.get_DQN_prediction(next_state, tf.logical_not(comb_mask), next_fine_mask)    # NxA

                    if self.method != 'Double':
                        # DQN
                        best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
                    else:
                        # Double-DQN
                        next_predict_value = self.get_DQN_prediction(next_state, tf.logical_not(comb_mask), next_fine_mask)
                        self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
                        predict_onehot = tf.one_hot(self.greedy_choice, self.num_actions, 1.0, 0.0)
                        best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

                    target = reward + (1.0 - tf.cast(isOver, tf.float32)) * self.gamma * tf.stop_gradient(best_v)
                    # target = tf.Print(target, [target], summarize=100)
                    # tf.assert_greater(target, -100., message='target error')
                    # tf.assert_greater(pred_action_value, -100., message='pred value error')
                    # pred_action_value = tf.Print(pred_action_value, [pred_action_value], summarize=100)

                    l2_loss = tensorpack.regularize_cost(name + '/dqn.*W{1}', l2_regularizer(1e-3))
                    # cost = tf.losses.mean_squared_error(target, pred_action_value)
                    with tf.control_dependencies([tf.assert_greater(target, -100., message='target error'), tf.assert_greater(pred_action_value, -100., message='pred value error')]):
                        cost = tf.losses.huber_loss(
                                        target, pred_action_value, reduction=tf.losses.Reduction.MEAN)
                    summary.add_param_summary((name + '.*/W', ['histogram', 'rms']))   # monitor all W
                    summary.add_moving_summary(cost)
                    costs.append(cost)
        if not get_current_tower_context().is_training:
            return
        return tf.add_n([costs[i] * self.cost_weights[i] for i in range(3)])

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.learning_rate, trainable=False)
        # opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(
            opt, [
                # gradproc.GlobalNormClip(2.0),
                gradproc.MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5)),
                  gradproc.SummaryGradient()])

    # @staticmethod
    def update_target_param(self):
        # update target network together
        vars = tf.global_variables()
        ops = []
        G = tf.get_default_graph()
        for v in vars:
            target_name = v.op.name
            for name in self.agent_names:
                if target_name.startswith(name + '/target'):
                    new_name = target_name.replace('target/', '')
                    logger.info("Target Network Update: {} <- {}".format(target_name, new_name))
                    ops.append(v.assign(G.get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network_all')

def counter_subset(list1, list2):
    c1, c2 = Counter(list1), Counter(list2)

    for (k, n) in c1.items():
        if n > c2[k]:
            return False
    return True
def get_mask_onehot60(cards, action_space, last_cards):
    # 1 valid; 0 invalid
    mask = np.zeros([len(action_space), 60])
    if cards is None:
        return mask
    if len(cards) == 0:
        return mask
    for j in range(len(action_space)):
        if counter_subset(action_space[j], cards):
            mask[j] = Card.char2onehot60(action_space[j])
    if last_cards is None:
        return mask
    if len(last_cards) > 0:
        for j in range(1, len(action_space)):
            if np.sum(mask[j]) > 0 and not CardGroup.to_cardgroup(action_space[j]).\
                    bigger_than(CardGroup.to_cardgroup(last_cards)):
                mask[j] = np.zeros([60])
    return mask

class Predictor:
    def __init__(self, predictor, num_actions=(100, 21)):
        self.predictor = predictor
        self.num_actions = num_actions
        self.encoding = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../AutoEncoder/encoding.npy'))
        print('predictor loaded')

    def pad_state(self, state):
        # since out net uses max operation, we just dup the last row and keep the result same
        newstates = []
        for s in state:
            assert s.shape[0] <= self.num_actions[1]
            s = np.concatenate([s, np.repeat(s[-1:, :], self.num_actions[1] - s.shape[0], axis=0)], axis=0)
            newstates.append(s)
        newstates = np.stack(newstates, axis=0)
        if len(state) < self.num_actions[0]:
            state = np.concatenate([newstates, np.repeat(newstates[-1:, :, :], self.num_actions[0] - newstates.shape[0], axis=0)], axis=0)
        else:
            state = newstates
        return state

    def pad_fine_mask(self, mask):
        if mask.shape[0] < self.num_actions[0]:
            mask = np.concatenate([mask, np.repeat(mask[-1:], self.num_actions[0] - mask.shape[0], 0)], 0)
        return mask

    def pad_action_space(self, available_actions):
        # print(available_actions)
        for i in range(len(available_actions)):
            available_actions[i] += [available_actions[i][-1]] * (self.num_actions[1] - len(available_actions[i]))
        if len(available_actions) < self.num_actions[0]:
            available_actions.extend([available_actions[-1]] * (self.num_actions[0] - len(available_actions)))

    def get_combinations(self, curr_cards_char, last_cards_char):
        if len(curr_cards_char) > 10:
            card_mask = Card.char2onehot60(curr_cards_char).astype(np.uint8)
            mask = augment_action_space_onehot60
            a = np.expand_dims(1 - card_mask, 0) * mask
            invalid_row_idx = set(np.where(a > 0)[0])
            if len(last_cards_char) == 0:
                invalid_row_idx.add(0)

            valid_row_idx = [i for i in range(len(augment_action_space)) if i not in invalid_row_idx]

            mask = mask[valid_row_idx, :]
            idx_mapping = dict(zip(range(mask.shape[0]), valid_row_idx))

            # augment mask
            # TODO: known issue: 555444666 will not decompose into 5554 and 66644
            combs = get_combinations_nosplit(mask, card_mask)
            combs = [([] if len(last_cards_char) == 0 else [0]) + [clamp_action_idx(idx_mapping[idx]) for idx in comb] for comb in combs]

            if len(last_cards_char) > 0:
                idx_must_be_contained = set(
                    [idx for idx in valid_row_idx if CardGroup.to_cardgroup(augment_action_space[idx]). \
                        bigger_than(CardGroup.to_cardgroup(last_cards_char))])
                combs = [comb for comb in combs if not idx_must_be_contained.isdisjoint(comb)]
                fine_mask = np.zeros([len(combs), self.num_actions[1]], dtype=np.bool)
                for i in range(len(combs)):
                    for j in range(len(combs[i])):
                        if combs[i][j] in idx_must_be_contained:
                            fine_mask[i][j] = True
            else:
                fine_mask = None
        else:
            mask = get_mask_onehot60(curr_cards_char, action_space, None).reshape(len(action_space), 15, 4).sum(-1).astype(
                np.uint8)
            valid = mask.sum(-1) > 0
            cards_target = Card.char2onehot60(curr_cards_char).reshape(-1, 4).sum(-1).astype(np.uint8)
            # do not feed empty to C++, which will cause infinite loop
            combs = get_combinations_recursive(mask[valid, :], cards_target)
            idx_mapping = dict(zip(range(valid.shape[0]), np.where(valid)[0]))

            combs = [([] if len(last_cards_char) == 0 else [0]) + [idx_mapping[idx] for idx in comb] for comb in combs]

            if len(last_cards_char) > 0:
                valid[0] = True
                idx_must_be_contained = set(
                    [idx for idx in range(len(action_space)) if valid[idx] and CardGroup.to_cardgroup(action_space[idx]). \
                        bigger_than(CardGroup.to_cardgroup(last_cards_char))])
                combs = [comb for comb in combs if not idx_must_be_contained.isdisjoint(comb)]
                fine_mask = np.zeros([len(combs), self.num_actions[1]], dtype=np.bool)
                for i in range(len(combs)):
                    for j in range(len(combs[i])):
                        if combs[i][j] in idx_must_be_contained:
                            fine_mask[i][j] = True
            else:
                fine_mask = None
        return combs, fine_mask

    def subsample_combs_masks(self, combs, masks, num_sample):
        if masks is not None:
            assert len(combs) == masks.shape[0]
        idx = np.random.permutation(len(combs))[:num_sample]
        return [combs[i] for i in idx], (masks[idx] if masks is not None else None)

    def get_state_and_action_space(self, is_comb, curr_cards_char=None, last_two_cards_char=None, prob_state=None, cand_state=None, cand_actions=None, action=None, fine_mask=None):
        def cards_char2embedding(cards_char):
            test = (action_space_onehot60 == Card.char2onehot60(cards_char))
            test = np.all(test, axis=1)
            target = np.where(test)[0]
            return self.encoding[target[0]]

        if is_comb:
            last_cards_char = last_two_cards_char[0]
            if not last_cards_char:
                last_cards_char = last_two_cards_char[1]
            if not last_cards_char :
                last_cards_char = last_two_cards_char[2]
            combs, fine_mask = self.get_combinations(curr_cards_char, last_cards_char)
            if len(combs) > self.num_actions[0]:
                combs, fine_mask = self.subsample_combs_masks(combs, fine_mask, self.num_actions[0])
            # TODO: utilize temporal relations to speedup
            available_actions = [[action_space[idx] for idx in comb] for
                                 comb in combs]
            # if fine_mask is not None:
            #     fine_mask = np.concatenate([np.ones([fine_mask.shape[0], 1], dtype=np.bool), fine_mask[:, :20]], axis=1)
            assert len(combs) > 0
            # if len(combs) == 0:
            #     available_actions = [[[]]]
            #     fine_mask = np.zeros([1, num_actions[1]], dtype=np.bool)
            #     fine_mask[0, 0] = True
            if fine_mask is not None:
                fine_mask = self.pad_fine_mask(fine_mask)
            self.pad_action_space(available_actions)
            # if len(combs) < num_actions[0]:
            #     available_actions.extend([available_actions[-1]] * (num_actions[0] - len(combs)))
            state = [np.stack([self.encoding[idx] for idx in comb]) for comb in combs]
            assert len(state) > 0
            # if len(state) == 0:
            #     assert len(combs) == 0
            #     state = [np.array([encoding[0]])]
            # test = (action_space_onehot60 == Card.char2onehot60(last_cards_char))
            # test = np.all(test, axis=1)
            # target = np.where(test)[0]
            # assert target.size == 1
            extra_state = np.concatenate([cards_char2embedding(last_two_cards_char[0]), cards_char2embedding(last_two_cards_char[1]),cards_char2embedding(last_two_cards_char[2]),  prob_state])
            for i in range(len(state)):
                state[i] = np.concatenate([state[i], np.tile(extra_state[None, :], [state[i].shape[0], 1])], axis=-1)
            state = self.pad_state(state)
            assert state.shape[0] == self.num_actions[0] and state.shape[1] == self.num_actions[1]
        else:
            assert action is not None
            if fine_mask is not None:
                fine_mask = fine_mask[action]
            available_actions = cand_actions[action]
            state = cand_state[action:action + 1, :, :]
            state = np.repeat(state, self.num_actions[0], axis=0)
            assert state.shape[0] == self.num_actions[0] and state.shape[1] == self.num_actions[1]
        return state, available_actions, fine_mask

    def predict(self, handcards, last_two_cards, prob_state):
        # print('%s current cards' % ('lord' if role_id == 2 else 'farmer'), curr_cards_char)
        fine_mask_input = np.ones([max(self.num_actions[0], self.num_actions[1])], dtype=np.bool)
        # first hierarchy
        # print(handcards, last_cards)
        state, available_actions, fine_mask = self.get_state_and_action_space(True, curr_cards_char=handcards, last_two_cards_char=last_two_cards, prob_state=prob_state)
        # print(available_actions)
        p_x = min(888,state.shape[2])
        q_values = self.predictor(state[None, :, :, :p_x], np.array([True]), np.array([fine_mask_input]))[0][0]
        action = np.argmax(q_values)
        assert action < self.num_actions[0]
        # clamp action to valid range
        action = min(action, self.num_actions[0] - 1)

        # second hierarchy
        state, available_actions, fine_mask = self.get_state_and_action_space(False, cand_state=state, cand_actions=available_actions, action=action, fine_mask=fine_mask)
        if fine_mask is not None:
            fine_mask_input = fine_mask if fine_mask.shape[0] == max(self.num_actions[0], self.num_actions[1]) \
                else np.pad(fine_mask, (0, max(self.num_actions[0], self.num_actions[1]) - fine_mask.shape[0]), 'constant',
                            constant_values=(0, 0))
        q_values = self.predictor(state[None, :, :, :888], np.array([False]), np.array([fine_mask_input]))[0][0]
        if fine_mask is not None:
            q_values = q_values[:self.num_actions[1]]
            # assert np.all(q_values[np.where(np.logical_not(fine_mask))[0]] < -100)
            q_values[np.where(np.logical_not(fine_mask))[0]] = np.nan
        action = np.nanargmax(q_values)
        assert action < self.num_actions[1]
        # clamp action to valid range
        action = min(action, self.num_actions[1] - 1)
        intention = available_actions[action]
        return intention


