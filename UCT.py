# UCT.py
import numpy as np
import copy
from Game import BeloteGame
from Card import BeloteCard12, BeloteCard32

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class Node(object):
    def __init__(self, team = None, inc_action = None):
        self.total_reward = 0
        self.n_visits = 0
        self.children = []
        self.actions_tried = []
        self.team = team
        self.inc_action = inc_action

def best_child(node, game, c):
    values = [((child.total_reward)/(child.n_visits)) + c*np.sqrt((2*np.log(node.n_visits))/(child.n_visits)) for child in node.children]
    best_child = node.children[np.argmax(values)]
    game.step(best_child.inc_action)
    return best_child

def expand(node, game):
    player = game.players[game.get_next_to_play_idx()]
    actions = [c.id for c in player.get_legal_moves(game)]
    new_action = next((a for a in actions if a not in node.actions_tried))
    game.step(new_action)
    new_child = Node(player.team, new_action) # the "team" of the node is the team of the player making the incoming action
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

def uct_search(game, max_steps = 5):
    root = Node()
    for i in range(max_steps):
        # print(f"Iteration {i} :")
        root.n_visits += 1
        players = copy.deepcopy(game.players)
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
    
    # Copy needed to avoid moving the game one step (not very efficient, could be improved)
    players = copy.deepcopy(game.players)
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
    return (best_child(root, simu_game, c=0)).inc_action

def uct_all_possible_worlds(game, max_steps = 5):
    best_actions = []
    current_player = game.players[game.get_next_to_play_idx()]
    possible_states = current_player.get_all_possible_states(game)
    for i in range(len(possible_states)):
        players = copy.deepcopy(game.players)
        other_players = [p for p in players if p.id != current_player.id]
        if game.n_cards == 12:
            for j, player in enumerate(other_players):
                player.hand = [BeloteCard12(id) for id in possible_states[i][j]]
        elif game.n_cards == 32:
            for j, player in enumerate(other_players):
                player.hand = [BeloteCard32(id) for id in possible_states[i][j]]
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
        best_actions.append(uct_search(simu_game, max_steps=max_steps))
    return max(set(best_actions), key=best_actions.count)

def uct_sample_possible_worlds(game, n_samples, max_steps = 5): # Not used
    best_actions = []
    current_player = game.players[game.get_next_to_play_idx()]
    possible_states = current_player.sample_possible_states(n_samples, game)
    for i in range(len(possible_states)):
        state_idx = i % len(possible_states)
        if i >= len(possible_states):
            possible_states = current_player.sample_possible_states(n_samples, game)
        players = copy.deepcopy(game.players)
        other_players = [p for p in players if p.id != current_player.id]
        if game.n_cards == 12:
            for j, player in enumerate(other_players):
                player.hand = [BeloteCard12(id) for id in possible_states[state_idx][j]]
        elif game.n_cards == 32:
            for j, player in enumerate(other_players):
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
        best_actions.append(uct_search(simu_game, max_steps=max_steps))
    return max(set(best_actions), key=best_actions.count)

def simulate_possible_world(possible_state, current_player_id, game, max_steps):
    from Game import BeloteGame
    from Card import BeloteCard12
    import copy

    players = copy.deepcopy(game.players)
    other_players = [p for p in players if p.id != current_player_id]
    if game.n_cards == 12:
        for j, player in enumerate(other_players):
            player.hand = [BeloteCard12(id) for id in possible_state[j]]
    elif game.n_cards == 32:
        for j, player in enumerate(other_players):
            player.hand = [BeloteCard32(id) for id in possible_state[j]]

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
    return uct_search(simu_game, max_steps=max_steps)

def simulate_possible_world_32_cards(possible_state, current_player_id, game, max_steps):
    from Game import BeloteGame
    from Card import BeloteCard32
    import copy

    players = copy.deepcopy(game.players)
    other_players = [p for p in players if p.id != current_player_id]
    if game.n_cards == 12:
        for j, player in enumerate(other_players):
            player.hand = [BeloteCard12(id) for id in possible_state[j]]
    elif game.n_cards == 32:
        for j, player in enumerate(other_players):
            player.hand = [BeloteCard32(id) for id in possible_state[j]]

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
    return uct_search(simu_game, max_steps=max_steps)

def uct_all_possible_worlds_parallel(game, max_steps=5):
    current_player = game.players[game.get_next_to_play_idx()]
    possible_states = current_player.get_all_possible_states(game)
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(simulate_possible_world, state, current_player.id, game, max_steps)
            for state in possible_states
        ]
        best_actions = [f.result() for f in futures]

    return max(set(best_actions), key=best_actions.count)
    
def uct_sample_possible_worlds_parallel(game, n_samples, max_steps=5):
    current_player = game.players[game.get_next_to_play_idx()]
    possible_states = current_player.sample_possible_states(n_samples, game)
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(simulate_possible_world_32_cards, state, current_player.id, game, max_steps)
            for state in possible_states
        ]
        best_actions = [f.result() for f in futures]

    return max(set(best_actions), key=best_actions.count)

def uct_search_total_reward(game, max_steps = 5):
    actions_world_scores = np.zeros(game.n_cards)
    root = Node()
    for i in range(max_steps):
        # print(f"Iteration {i} :")
        root.n_visits += 1
        players = copy.deepcopy(game.players)
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
    
    for child in root.children:
        if child.n_visits > 0:
            actions_world_scores[child.inc_action] = ((child.total_reward)/(child.n_visits))
    return actions_world_scores

def uct_all_possible_worlds_total_reward(game, max_steps = 5):
    actions_total_scores = np.zeros(game.n_cards)
    current_player = game.players[game.get_next_to_play_idx()]
    possible_states = current_player.get_all_possible_states(game)
    for i in range(len(possible_states)):
        players = copy.deepcopy(game.players)
        other_players = [p for p in players if p.id != current_player.id]
        if game.n_cards == 12:
            for j, player in enumerate(other_players):
                player.hand = [BeloteCard12(id) for id in possible_states[i][j]]
        elif game.n_cards == 32:
            for j, player in enumerate(other_players):
                player.hand = [BeloteCard32(id) for id in possible_states[i][j]]
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
        actions_world_scores = uct_search_total_reward(simu_game, max_steps=max_steps)
        actions_total_scores += actions_world_scores
    return np.argmax(actions_total_scores)

def simulate_possible_world_reward(possible_state, current_player_id, game, max_steps):
    from Game import BeloteGame
    from Card import BeloteCard12
    from Card import BeloteCard32
    import copy
    import numpy as np

    players = copy.deepcopy(game.players)
    other_players = [p for p in players if p.id != current_player_id]
    if game.n_cards == 12:
        for j, player in enumerate(other_players):
            player.hand = [BeloteCard12(id) for id in possible_state[j]]
    elif game.n_cards == 32:
        for j, player in enumerate(other_players):
            player.hand = [BeloteCard32(id) for id in possible_state[j]]
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
    return uct_search_total_reward(simu_game, max_steps=max_steps)

def simulate_possible_world_reward_32_cards(possible_state, current_player_id, game, max_steps):
    from Game import BeloteGame
    from Card import BeloteCard32
    import copy
    import numpy as np

    players = copy.deepcopy(game.players)
    other_players = [p for p in players if p.id != current_player_id]
    if game.n_cards == 12:
        for j, player in enumerate(other_players):
            player.hand = [BeloteCard12(id) for id in possible_state[j]]
    elif game.n_cards == 32:
        for j, player in enumerate(other_players):
            player.hand = [BeloteCard32(id) for id in possible_state[j]]
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
    return uct_search_total_reward(simu_game, max_steps=max_steps)

def uct_all_possible_worlds_total_reward_parallel(game, max_steps=5):
    current_player = game.players[game.get_next_to_play_idx()]
    possible_states = current_player.get_all_possible_states(game)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(simulate_possible_world_reward, state, current_player.id, game, max_steps)
            for state in possible_states
        ]
        results = [f.result() for f in futures]

    actions_total_scores = sum(results)
    return np.argmax(actions_total_scores)

def uct_sample_possible_worlds_total_reward_parallel(game, n_samples, max_steps=5):
    current_player = game.players[game.get_next_to_play_idx()]
    possible_states = current_player.sample_possible_states(n_samples, game)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(simulate_possible_world_reward_32_cards, state, current_player.id, game, max_steps)
            for state in possible_states
        ]
        results = [f.result() for f in futures]

    actions_total_scores = sum(results)
    return np.argmax(actions_total_scores)