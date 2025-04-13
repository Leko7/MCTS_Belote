# Evaluate.py
import time
import numpy as np
import pandas as pd

from Player import BelotePlayer
from Game import BeloteGame
from Utils import save_csv, save_png

from UCT import uct_search, uct_all_possible_worlds_parallel, uct_all_possible_worlds_total_reward_parallel, uct_sample_possible_worlds_parallel, uct_sample_possible_worlds_total_reward_parallel
from ISMCTS import ismcts_search, ismcts_search_sampling

def evaluate(max_steps, num_games, mcts_method, method_name, num_mcts_moves, n_cards):
    mcts_scores = []
    mcts_wins = []
    n_ties = 0
    random_scores = []

    start = time.time()
    for s in range(num_games):
        print(f"Game {s}:")
        if s < (num_games // 2): # Switch which team plays first in half of the games, to avoid bias.
            mcts_pair = [0,2]
        else:
            mcts_pair = [1,3]
        
        players = [BelotePlayer(0), BelotePlayer(1), BelotePlayer(2), BelotePlayer(3)]
        game = BeloteGame(players, n_cards)
        game.reset_game()
        game.distribute_cards()

        while len(game.defense_tricks) + len(game.attack_tricks) < game.n_cards:
            player = game.players[game.get_next_to_play_idx()]
            if player.id in mcts_pair:
                if len(player.get_legal_moves(game)) > 1:
                    print(f"MCTS move")
                    action = mcts_method(game, max_steps=max_steps)
                else:
                    print("MCTS player plays the only move available")
                    action = player.random_action(game)
            else:
                print("Random move")
                action = player.random_action(game)
            game.step(action)
        if s < (num_games // 2):
            mcts_scores.append(sum([c.value for c in game.attack_tricks]))
            random_scores.append(sum([c.value for c in game.defense_tricks]))
        else:
            mcts_scores.append(sum([c.value for c in game.defense_tricks]))
            random_scores.append(sum([c.value for c in game.attack_tricks]))
        
        if mcts_scores[-1] > random_scores[-1]:
            mcts_wins.append(1)
        else:
            mcts_wins.append(0)
        
        if mcts_scores[-1] == random_scores[-1]:
            n_ties += 1

    mcts_mean = np.mean(mcts_scores)
    mcts_std = np.std(mcts_scores, ddof=1)

    # random_mean = np.mean(random_scores)
    # random_std = np.std(random_scores, ddof=1)

    win_rate = np.mean(mcts_wins)

    end = time.time()
    elapsed_time = end - start
    time_per_move = elapsed_time / (num_games*num_mcts_moves)

    if n_cards == 12:
        return {
            "Algorithm" : method_name,
            "Average Score" : f"{round(mcts_mean, 2)} (/80)",
            "STD" : f"{round(mcts_std, 2)}",
            "Win Rate" : f"{round(win_rate*100, 2)}%",
            "Draw Rate" : f"{round(n_ties*100/num_games, 2)}%",
            "Time per Move" : f"{round(time_per_move, 2)} seconds"
        }
    
    elif n_cards == 32:
        return {
            "Algorithm" : method_name,
            "Average Score" : f"{round(mcts_mean, 2)} (/152)",
            "STD" : f"{round(mcts_std, 2)}",
            "Win Rate" : f"{round(win_rate*100, 2)}%",
            "Draw Rate" : f"{round(n_ties*100/num_games, 2)}%",
            "Time per Move" : f"{round(time_per_move, 2)} seconds"
        }

def evaluate_sampling_method(max_steps, num_games, mcts_method, method_name, num_mcts_moves, n_cards, n_samples):
    mcts_scores = []
    mcts_wins = []
    n_ties = 0
    random_scores = []

    start = time.time()
    for s in range(num_games):
        print(f"Game {s}:")
        if s < (num_games // 2): # Switch which team plays first in half of the games, to avoid bias.
            mcts_pair = [0,2]
        else:
            mcts_pair = [1,3]
        
        players = [BelotePlayer(0), BelotePlayer(1), BelotePlayer(2), BelotePlayer(3)]
        game = BeloteGame(players, n_cards)
        game.reset_game()
        game.distribute_cards()

        while len(game.defense_tricks) + len(game.attack_tricks) < game.n_cards:
            player = game.players[game.get_next_to_play_idx()]
            if player.id in mcts_pair:
                if len(player.get_legal_moves(game)) > 1:
                    print(f"MCTS move")
                    action = mcts_method(game, n_samples, max_steps=max_steps)
                else:
                    print("MCTS player plays the only move available")
                    action = player.random_action(game)
            else:
                print("Random move")
                action = player.random_action(game)
            game.step(action)
        if s < (num_games // 2):
            mcts_scores.append(sum([c.value for c in game.attack_tricks]))
            random_scores.append(sum([c.value for c in game.defense_tricks]))
        else:
            mcts_scores.append(sum([c.value for c in game.defense_tricks]))
            random_scores.append(sum([c.value for c in game.attack_tricks]))
        
        if mcts_scores[-1] > random_scores[-1]:
            mcts_wins.append(1)
        else:
            mcts_wins.append(0)
        
        if mcts_scores[-1] == random_scores[-1]:
            n_ties += 1

    mcts_mean = np.mean(mcts_scores)
    mcts_std = np.std(mcts_scores, ddof=1)

    # random_mean = np.mean(random_scores)
    # random_std = np.std(random_scores, ddof=1)

    win_rate = np.mean(mcts_wins)

    end = time.time()
    elapsed_time = end - start
    time_per_move = elapsed_time / (num_games*num_mcts_moves)

    return {
        "Algorithm" : method_name,
        "Average Score" : f"{round(mcts_mean, 2)} (/152)",
        "STD" : f"{round(mcts_std, 2)}",
        "Win Rate" : f"{round(win_rate*100, 2)}%",
        "Draw Rate" : f"{round(n_ties*100/num_games, 2)}%",
        "Time per Move" : f"{round(time_per_move, 2)} seconds"
    }

def compare_methods_belote12(n_cards, num_games, png_path, csv_path):
    num_mcts_moves = ((n_cards / 4) - 1)*2 # (number of tricks - 1) x number of mcts players

    results = []

    # Evaluate Cheating UCT
    mcts_method = uct_search
    method_name = 'Cheating UCT'
    max_steps = 1_000
    results.append(evaluate(max_steps, num_games, mcts_method, method_name, num_mcts_moves, n_cards))
    
    # Evaluate All Worlds UCT (max times best action)
    mcts_method = uct_all_possible_worlds_parallel
    method_name = 'All Worlds UCT (max times best action)'
    max_steps = 400
    results.append(evaluate(max_steps, num_games, mcts_method, method_name, num_mcts_moves, n_cards))

    # Evaluate All Worlds UCT (max total reward action)
    mcts_method = uct_all_possible_worlds_total_reward_parallel
    method_name = 'All Worlds UCT (max total reward action)'
    max_steps = 400
    results.append(evaluate(max_steps, num_games, mcts_method, method_name, num_mcts_moves, n_cards))

    # Evaluate SO-ISMCTS
    mcts_method = ismcts_search
    method_name = 'SO-ISMCTS'
    max_steps = 50_000
    results.append(evaluate(max_steps, num_games, mcts_method, method_name, num_mcts_moves, n_cards))

    df = pd.DataFrame(results)
    save_png(df, png_path)
    save_csv(df, csv_path)

def compare_methods_belote32(n_cards, num_games, png_path, csv_path):
    num_mcts_moves = ((n_cards / 4) - 1)*2 # (n_tricks - 1)*number of mcts players

    results = []

    # Evaluate Cheating UCT
    mcts_method = uct_search
    method_name = 'Cheating UCT'
    max_steps = 30_000
    print(f"{method_name} games")
    results.append(evaluate(max_steps, num_games, mcts_method, method_name, num_mcts_moves, n_cards))
    
    # Evaluate All Worlds UCT (max times best action)
    mcts_method = uct_sample_possible_worlds_parallel
    method_name = 'Sampled All Worlds UCT (max times best action)'
    max_steps = 200
    n_samples = 1000
    print(f"{method_name} games")
    results.append(evaluate_sampling_method(max_steps, num_games, mcts_method, method_name, num_mcts_moves, n_cards, n_samples))

    # Evaluate All Worlds UCT (max total reward action)
    mcts_method = uct_sample_possible_worlds_total_reward_parallel
    method_name = 'Sampled All Worlds UCT (max total reward action)'
    max_steps = 200
    n_samples = 1000
    print(f"{method_name} games")
    results.append(evaluate_sampling_method(max_steps, num_games, mcts_method, method_name, num_mcts_moves, n_cards, n_samples))

    # Evaluate SO-ISMCTS with sampling of possible states
    mcts_method = ismcts_search_sampling
    method_name = 'SO-ISMCTS'
    max_steps = 3_000
    n_samples = 1500
    print(f"{method_name} games")
    results.append(evaluate_sampling_method(max_steps, num_games, mcts_method, method_name, num_mcts_moves, n_cards, n_samples))

    df = pd.DataFrame(results)
    save_png(df, png_path)
    save_csv(df, csv_path)

if __name__ == "__main__":
    if False:
        # Evaluate 4-player Belote with only 12 cards, with no sampling needed (/!\ takes 1 HOUR)
        n_cards = 12
        num_games = 60
        png_path = "12_cards_results.png"
        csv_path = "12_cards_results.csv"
        compare_methods_belote12(n_cards, num_games, png_path, csv_path)

    if True:
        # Evaluate 4-player Belote with the full deck of 32 cards, with sampling of possible states (/!\ takes 7+ HOURS)
        n_cards = 32
        num_games = 30
        png_path = "32_cards_results_more_time_cheating_uct.png"
        csv_path = "32_cards_results_more_time_cheating_uct.csv"
        compare_methods_belote32(n_cards, num_games, png_path, csv_path)