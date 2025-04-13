# ISMCTS.py
import numpy as np
import copy
from Game import BeloteGame
from Card import BeloteCard12, BeloteCard32

class Node(object):
    def __init__(self, n_cards, team = None, inc_action = None):
        self.total_reward = 0
        self.n_visits = 0
        self.children = []
        self.actions_tried = []
        self.availabilities = np.zeros(n_cards)
        self.team = team
        self.inc_action = inc_action

def best_child(node, game, c):
    values = []
    player = game.players[game.get_next_to_play_idx()]
    actions = [c.id for c in player.get_legal_moves(game)]
    for child in node.children:
        if child.inc_action in actions:
            availability = node.availabilities[child.inc_action]
            values.append(((child.total_reward)/(child.n_visits)) + c*np.sqrt((2*np.log(availability))/(child.n_visits)))
        else:
            values.append(-np.inf)
    best_child = node.children[np.argmax(values)]
    game.step(best_child.inc_action)
    return best_child

def expand(node, game):
    player = game.players[game.get_next_to_play_idx()]
    actions = [c.id for c in player.get_legal_moves(game)]
    new_action = next((a for a in actions if a not in node.actions_tried))
    game.step(new_action)
    new_child = Node(game.n_cards, player.team, new_action) # the "team" of the node is the team of the player making the incoming action
    node.children.append(new_child)
    node.actions_tried.append(new_action)
    return new_child

def is_expandable(node, game):
    player = game.players[game.get_next_to_play_idx()]
    actions = [c.id for c in player.get_legal_moves(game)]
    new_actions = [a for a in actions if a not in node.actions_tried]
    if new_actions:
        return True
    else:
        return False
    
def tree_policy(node, game, search_list, cp=1/np.sqrt(2)):
    terminal = False
    while not terminal:
        player = game.players[game.get_next_to_play_idx()]
        actions = [c.id for c in player.get_legal_moves(game)]
        node.availabilities[actions] += 1
        if is_expandable(node, game):
            node = expand(node, game)
            search_list.append(node)
            node.n_visits += 1
            return node
        else:
            node = best_child(node, game, cp)
            search_list.append(node)
            node.n_visits += 1
        if len(game.attack_tricks) + len(game.defense_tricks) == game.n_cards:
            terminal = True
    return node

def default_policy(game):
    return game.playout() # should work even if s is terminal

def backup(playout, search_list):
    for node in search_list:
        if node.team == "attack":
            node.total_reward += playout["attack reward"]
        elif node.team == "defense":
            node.total_reward += playout["defense reward"]

def ismcts_search(game, max_steps = 5):
    current_player = game.players[game.get_next_to_play_idx()]
    # print("Computing all possible states")
    possible_states = current_player.get_all_possible_states(game)
    # print(f"Computed {len(possible_states)} possible states.")
    root = Node(game.n_cards)
    for i in range(max_steps):
        state_idx = np.random.randint(len(possible_states))
        # print(f"Iteration {i} :")
        root.n_visits += 1
        players = copy.deepcopy(game.players)
        other_players = [p for p in players if p.id != current_player.id]
        for j, player in enumerate(other_players):
            if game.n_cards == 12:
                player.hand = [BeloteCard12(id) for id in possible_states[state_idx][j]]
            elif game.n_cards == 32:
                player.hand = [BeloteCard32(id) for id in possible_states[state_idx][j]]
        attack_tricks = copy.deepcopy(game.attack_tricks)
        defense_tricks = copy.deepcopy(game.defense_tricks)
        cards_on_table = copy.deepcopy(game.cards_on_table)
        simu_game = BeloteGame(
            players=players,
            n_cards = game.n_cards,
            attack_tricks=attack_tricks,
            defense_tricks=defense_tricks,
            cards_on_table=cards_on_table
        )
        
        search_list = []
        node = tree_policy(root, simu_game, search_list)
        playout = default_policy(simu_game)
        backup(playout, search_list)
    
    values = [((child.total_reward)/(child.n_visits)) for child in root.children]
    return root.children[np.argmax(values)].inc_action

def ismcts_search_sampling(game, n_samples, max_steps = 5):
    current_player = game.players[game.get_next_to_play_idx()]
    root = Node(game.n_cards)
    #print(f"Sampling {n_samples} possible states")
    possible_states = current_player.sample_possible_states(n_samples, game)
    #print(f"Computed {len(possible_states)} possible states.")
    for i in range(max_steps):
        # print(f"Iteration {i} :")
        state_idx = i % len(possible_states)
        if i >= len(possible_states):
            possible_states = current_player.sample_possible_states(n_samples, game)
        root.n_visits += 1
        players = copy.deepcopy(game.players)
        other_players = [p for p in players if p.id != current_player.id]
        for j, player in enumerate(other_players):
            if game.n_cards == 12:
                player.hand = [BeloteCard12(id) for id in possible_states[state_idx][j]]
            elif game.n_cards == 32:
                player.hand = [BeloteCard32(id) for id in possible_states[state_idx][j]]
        attack_tricks = copy.deepcopy(game.attack_tricks)
        defense_tricks = copy.deepcopy(game.defense_tricks)
        cards_on_table = copy.deepcopy(game.cards_on_table)
        simu_game = BeloteGame(
            players=players,
            n_cards = game.n_cards,
            attack_tricks=attack_tricks,
            defense_tricks=defense_tricks,
            cards_on_table=cards_on_table
        )
        search_list = []
        node = tree_policy(root, simu_game, search_list)
        playout = default_policy(simu_game)
        backup(playout, search_list)
    
    values = [((child.total_reward)/(child.n_visits)) for child in root.children]
    return root.children[np.argmax(values)].inc_action