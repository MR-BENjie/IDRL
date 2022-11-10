import random
import math
import numpy as np
from collections import Counter
import tensorflow as tf

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

def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    leftID = random.randint(0, 1)
    rightID = 1 - leftID

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


def play(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    #generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    """
    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]
    """
    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} ".format(n_round, eps))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    """
    
    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        if train:
            models[0].flush_buffer(**buffer)

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
    """


    positions = ['landlord', 'landlord_up', 'landlord_down', 'landlord_front']

    T = 100
    prob_buf = {p: [] for p in positions}
    episode_return_buf = {p: [] for p in positions}
    obs_x_buf = {p: [] for p in positions}
    obs_action_buf = {p: [] for p in positions}
    obs_action_idx_buf = {p : [] for p in positions}

    size = {p: 0 for p in positions}

    _,_,role_id, env_output = env.reset()

    former_act_prob = np.zeros((len(env_output['raw_legal_actions']),1))
    feature_obs = np.zeros((len(env_output['raw_legal_actions']),54))
    for index, x in enumerate(env_output['raw_legal_actions']):
        feature_obs[index:] = _cards2array(x)
    while not env.is_over():

        # take actions for every model
        x_shape = env_output['x_batch'].shape
        state = np.reshape(env_output['x_batch'], (x_shape[0],1, 11, 54))

        #former_act_prob = np.tile(former_act_prob, (env_output.ndim, 1))
        acts = models.act(state=[state,feature_obs], prob=former_act_prob, eps=eps)

        obs_action_idx_buf[positions[role_id]].append(acts)
        actions = env_output['raw_legal_actions'][acts[0]]
        obs_x_buf[positions[role_id]].append([state[acts[0]],feature_obs[acts[0]]])
        obs_action_buf[positions[role_id]].append(np.array([acts]))
        episode_return_buf[positions[role_id]].append(None)
        prob_buf[positions[role_id]].append(former_act_prob[acts[0]:])
        # simulate one step
        _,_,env_output,done = env.step(actions)
        size[positions[role_id]] += 1
        former_act_prob = np.zeros((len(env_output['raw_legal_actions']), 1))
        feature_obs = np.zeros((len(env_output['raw_legal_actions']), 54))
        for index, x in enumerate(env_output['raw_legal_actions']):
            feature_obs[index:] = _cards2array(x)
        role_id = (role_id+1)%4
    landlord = env_output['raw_obs']['landlord']
    winner_id = env.game.winner_id
    if winner_id in landlord:
        winner_team = landlord
    else:
        winner_team = [0,1,2,3]
        for w in landlord:
            winner_team.remove(w)
    winner_p = [positions[index] for index in winner_team]
    for p in range(4):

        # landlord阵营赋值正的，farmer阵营赋值负的
        length = len(obs_x_buf[positions[p]])
        for i in range(length):
            if p in winner_team:
                episode_return = 1
            else:
                episode_return = 0
            buffer = {
                'state': obs_x_buf[positions[p]][i], 'acts': obs_action_buf[positions[p]][i],'rewards':np.array([episode_return]),
                'ids': [p],'alives':np.array([True])
            }
            buffer['prob'] = prob_buf[positions[p]][i]
            if train:
                models.flush_buffer(**buffer)
    if train:
        models.train()

    return obs_action_idx_buf,obs_action_buf,winner_p


def battle(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards
