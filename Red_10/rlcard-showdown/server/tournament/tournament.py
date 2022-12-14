import os
import json
from tqdm import tqdm
import numpy as np

from .rlcard_wrap import rlcard

class Tournament(object):
    
    def launch(self):
        """ Currently for two-player game only
        """
        num_models = len(self.model_ids)
        games_data = []
        payoffs_data = []
        for i in range(num_models):
            for j in range(num_models):
                if j == i:
                    continue
                print(self.game, '-', self.model_ids[i], 'VS', self.model_ids[j])
                if self.game == 'red_10':
                    agent0 = self.models[i].agents()
                    agent1 = self.models[j].agents()
                    agents = [agent0[0], agent0[1], agent1[2],agent1[3]]
                    names = [self.model_ids[i], self.model_ids[i], self.model_ids[j],self.model_ids[j]]
                    data, payoffs, wins = red_10_tournament(self.game, agents, names, self.num_eval_games,[self.models[i],self.models[i],self.models[j],self.models[j]])
                elif self.game == 'doudizhu':
                    agents = [self.models[i].agents[0], self.models[j].agents[1], self.models[j].agents[2]]
                    names = [self.model_ids[i], self.model_ids[j], self.model_ids[j]]
                    data, payoffs, wins = doudizhu_tournament(self.game, agents, names, self.num_eval_games)
                elif self.game == 'leduc-holdem':
                    agents = [self.models[i].agents[0], self.models[j].agents[1]]
                    names = [self.model_ids[i], self.model_ids[j]]
                    data, payoffs, wins = leduc_holdem_tournament(self.game, agents, names, self.num_eval_games)
                mean_payoff = np.mean(payoffs)
                print('Average payoff:', mean_payoff)
                print()

                for k in range(len(data)):
                    game_data = {}
                    game_data['name'] = self.game
                    game_data['index'] = k
                    game_data['agent0'] = self.model_ids[i]
                    game_data['agent1'] = self.model_ids[j]
                    game_data['win'] = wins[k]
                    game_data['replay'] = data[k]
                    game_data['payoff'] = payoffs[k]

                    games_data.append(game_data)

                payoff_data = {}
                payoff_data['name'] = self.game
                payoff_data['agent0'] = self.model_ids[i]
                payoff_data['agent1'] = self.model_ids[j]
                payoff_data['payoff'] = mean_payoff
                payoffs_data.append(payoff_data)
        return games_data, payoffs_data

    def __init__(self, game, model_ids, num_eval_games=100):
        """ Default for two player games
            For Dou Dizhu, the two peasants use the same model
        """
        self.game = game
        self.model_ids = model_ids
        self.num_eval_games = num_eval_games
        # Load the models
        self.models = [rlcard.models.load(model_id) for model_id in model_ids]

def doudizhu_tournament(game, agents, names, num_eval_games):
    env = rlcard.make(game, config={'allow_raw_data': True})
    env.set_agents(agents)
    payoffs = []
    json_data = []
    wins = []
    for _ in tqdm(range(num_eval_games)):
        data = {}
        roles = ['landlord', 'peasant', 'peasant']
        data['playerInfo'] = [{'id': i, 'index': i, 'role': roles[i], 'agentInfo': {'name': names[i]}} for i in range(env.num_players)]
        state, player_id = env.reset()
        perfect = env.get_perfect_information()
        data['initHands'] = perfect['hand_cards_with_suit']
        current_hand_cards = perfect['hand_cards_with_suit'].copy()
        for i in range(len(current_hand_cards)):
            current_hand_cards[i] = current_hand_cards[i].split()
        data['moveHistory'] = []
        while not env.is_over():
            action, info = env.agents[player_id].eval_step(state)
            history = {}
            history['playerIdx'] = player_id
            if env.agents[player_id].use_raw:
                _action = action
            else:
                _action = env._decode_action(action)
            history['move'] = _calculate_doudizhu_move(_action, player_id, current_hand_cards)
            history['info'] = info

            data['moveHistory'].append(history)
            state, player_id = env.step(action, env.agents[player_id].use_raw)
        data = json.dumps(str(data))
        #data = json.dumps(data, indent=2, sort_keys=True)
        json_data.append(data)
        if env.get_payoffs()[0] > 0:
            wins.append(True)
        else:
            wins.append(False)
        payoffs.append(env.get_payoffs()[0])
    return json_data, payoffs, wins

# wins = true landlord win  ; wins = false passent win
def red_10_tournament(game, agents, names, num_eval_games, game_model):
    env = rlcard.make(game, config={'allow_raw_data': True})
    env.set_agents(agents)
    payoffs = []
    json_data = []
    wins = []
    for _ in tqdm(range(num_eval_games)):
        data = {}
        rela_state, danger_state, player_id, red_10_state = env.reset()
        
        landlord_id = rela_state['raw_obs']['landlord']
        lorp_roles = []
        for i in range(env.num_players):
            if i in landlord_id:
                lorp_roles.append('landlord')
            else:
                lorp_roles.append('peasant')
        
        data['playerInfo'] = [{'id': i, 'index': i, 'role': lorp_roles[i], 'agentInfo': {'name': names[i]}} for i in range(env.num_players)]
        perfect = env.get_perfect_information()
        data['initHands'] = perfect['hand_cards_with_suit']
        current_hand_cards = perfect['hand_cards_with_suit'].copy()
        for i in range(len(current_hand_cards)):
            current_hand_cards[i] = current_hand_cards[i].split()
        data['moveHistory'] = []

        rule_num = 0
        while not env.is_over():
            relation_score, dangerou_score,model_numbel, agent = game_model[rule_num].agents(rela_state,danger_state,rule_num)
            action, info = agent.eval_step(red_10_state)

            info['relation_score'] = relation_score
            info['dangerou_score'] = dangerou_score
            info['model_numble'] = model_numbel
            history = {}
            history['playerIdx'] = player_id
            if env.agents[player_id].use_raw:
                _action = action
            else:
                _action = env._decode_action(action)
            history['move'] = _calculate_doudizhu_move(_action, player_id, current_hand_cards)
            history['info'] = info

            data['moveHistory'].append(history)

            current_color = rela_state['raw_obs']["current_color"]
            color = ''
            color_subindex = 0
            if _action != 'pass':
                for action_card in _action:
                    if action_card == 'T':
                        color += current_color[color_subindex]
                        color_subindex += 1
            rela_state, danger_state, red_10_state , game_over= env.step(_action, color)
            player_id = red_10_state['raw_obs']['self']
            #print(_action)
            rule_num+=1
            rule_num = rule_num%4

        data = json.dumps(str(data))
        #data = json.dumps(data, indent=2, sort_keys=True)
        json_data.append(data)

        payoff = env.get_payoffs()
        if payoff[0] > 0:
            wins.append(True)
        else:
            wins.append(False)
        payoffs.append(payoff[0])
    return json_data, payoffs, wins
def _calculate_doudizhu_move(action, player_id, current_hand_cards):
    if action == 'pass':
        return action
    trans = {'B': 'BJ', 'R': 'RJ'}
    cards_with_suit = []
    for card in action:
        if card in trans:
            cards_with_suit.append(trans[card])
            current_hand_cards[player_id].remove(trans[card])
        else:
            for hand_card in current_hand_cards[player_id]:
                if hand_card[1] == card:
                    cards_with_suit.append(hand_card)
                    current_hand_cards[player_id].remove(hand_card)
                    break
    return ' '.join(cards_with_suit)

def leduc_holdem_tournament(game, agents, names, num_eval_games):
    env = rlcard.make(game, config={'allow_raw_data': True})
    env.set_agents(agents)
    payoffs = []
    json_data = []
    wins = []
    for _ in tqdm(range(num_eval_games)):
        data = {}
        data['playerInfo'] = [{'id': i, 'index': i, 'agentInfo': {'name': names[i]}} for i in range(env.num_players)]
        state, player_id = env.reset()
        perfect = env.get_perfect_information()
        data['initHands'] = perfect['hand_cards']
        data['moveHistory'] = []
        round_history = []
        round_id = 0
        while not env.is_over():
            action, info = env.agents[player_id].eval_step(state)
            history = {}
            history['playerIdx'] = player_id
            if env.agents[player_id].use_raw:
                history['move'] = action
            else:
                history['move'] = env._decode_action(action)

            history['info'] = info
            round_history.append(history)
            state, player_id = env.step(action, env.agents[player_id].use_raw)
            perfect = env.get_perfect_information()
            if round_id < perfect['current_round'] or env.is_over():
                round_id = perfect['current_round']
                data['moveHistory'].append(round_history)
                round_history = []
        perfect = env.get_perfect_information()
        data['publicCard'] = perfect['public_card']
        data = json.dumps(str(data))
        #data = json.dumps(data, indent=2, sort_keys=True)
        json_data.append(data)
        if env.get_payoffs()[0] > 0:
            wins.append(True)
        else:
            wins.append(False)
        payoffs.append(env.get_payoffs()[0])
    return json_data, payoffs, wins


if __name__=='__main__':
    game = 'leduc-holdem'
    model_ids = ['leduc-holdem-random', 'leduc-holdem-rule-v1', 'leduc-holdem-cfr']
    t = Tournament(game, model_ids)
    games_data = t.launch()
    print(len(games_data))
    print(games_data[0])
    #root_path = './models'
    #agent1 = LeducHoldemDQNModel1(root_path)
    #agent2 = LeducHoldemRandomModel(root_path)
    #agent3 = LeducHoldemRuleModel()
    #agent4 = LeducHoldemCFRModel(root_path)
    #agent5 = LeducHoldemDQNModel2(root_path)
    #t = Tournament(agent1, agent2, agent3, agent4, agent5, 'leduc-holdem')
    ##t.competition()
    #t.evaluate()
