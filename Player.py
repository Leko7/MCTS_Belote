# Player.py
import numpy as np
import random
import itertools

class BelotePlayer(object):
    def __init__(self, id : int):
        self.id = id
        self.hand = None
        self.team = None
        self.first_to_play = False

    def get_legal_moves(self, belote_game):
        if belote_game.table_color == None:
            return self.hand
        elif belote_game.table_color in ['carreau', 'coeur', 'trèfle']:
            color_cards = [card for card in self.hand if card.color == belote_game.table_color]
            has_color = len(color_cards)>0
            if has_color:
                return color_cards
            else:
                piques = [card for card in self.hand if card.color == 'pique']
                if piques:
                    # obbligation to raise for Tarot only, not for Belote
                    # piques_in_table = [card.value for card in belote_game.cards_on_table if card.color == 'pique']
                    # if piques_in_table:
                    #     max_in_table = np.max(piques_in_table)
                    #     piques_to_raise = [card for card in piques if card.value > max_in_table]
                    #     if piques_to_raise:
                    #         return piques_to_raise
                    #     else:
                    return piques
                    # else:
                    #     return piques
                else:
                    return self.hand
        elif belote_game.table_color == 'pique':
            piques = [card for card in self.hand if card.color == 'pique']
            if piques:
                # piques_in_table = [card.value for card in belote_game.cards_on_table if card.color == 'pique']
                # max_in_table = np.max(piques_in_table)
                # piques_to_raise = [card for card in piques if card.value > max_in_table]
                # if piques_to_raise:
                #     return piques_to_raise
                # else:
                return piques
            else:
                return self.hand
            
    def play_card(self, card, belote_game):
        self.hand.remove(card)
        belote_game.cards_on_table.append(card)        
        if belote_game.table_color == None:
            belote_game.table_color = card.color

    def naive_move(self, belote_game):
        legal_moves = self.get_legal_moves(belote_game)
        card = np.random.choice(legal_moves, 1, replace=False)[0]
        # print(f"player {self.id} : {card.color}, {card.label}")
        self.play_card(card,belote_game)

    def heuristic_action(self, belote_game):
        legal_moves = self.get_legal_moves(belote_game)
        max_val = max([c.value for c in legal_moves])
        card = next((c for c in legal_moves if c.value == max_val))
        return card.id
    
    def random_action(self, belote_game):
        legal_moves = self.get_legal_moves(belote_game)
        card = np.random.choice(legal_moves, 1, replace=False)[0]
        return card.id
    
    def get_all_possible_states(self, belote_game):
        remaining_cards = []
        other_players = [p for p in belote_game.players if p.id != self.id ]
        other_card_counts = [len(p.hand) for p in other_players]
        for player in other_players:
            remaining_cards.extend([c.id for c in player.hand])
        possible_states = []
        for group1 in itertools.combinations(remaining_cards, other_card_counts[0]):
            remaining_after_group1 = [c for c in remaining_cards if c not in group1]
            for group2 in itertools.combinations(remaining_after_group1, other_card_counts[1]):
                group3 = [c for c in remaining_after_group1 if c not in group2]
                possible_states.append([list(group1), list(group2), group3])
        return possible_states
    
    def sample_possible_states(self, n_states, belote_game):
        remaining_cards = []
        other_players = [p for p in belote_game.players if p.id != self.id ]
        other_card_counts = [len(p.hand) for p in other_players]
        for player in other_players:
            remaining_cards.extend([c.id for c in player.hand])
        possible_states = []
        for _ in range(n_states):
            shuffled = random.sample(remaining_cards, len(remaining_cards))
            group1 = shuffled[:other_card_counts[0]]
            group2 = shuffled[other_card_counts[0]:other_card_counts[0]+other_card_counts[1]]
            group3 = shuffled[other_card_counts[0]+other_card_counts[1]:]
            possible_states.append([list(group1), list(group2), list(group3)])

        return possible_states