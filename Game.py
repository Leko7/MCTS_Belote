# Game.py
import numpy as np

from Card import BeloteCard32, BeloteCard12

class BeloteGame(object):
    def __init__(self, players, n_cards, attack_tricks=None, defense_tricks=None, cards_on_table=None):
        self.players = players
        self.attack_tricks = attack_tricks if attack_tricks else []
        self.defense_tricks = defense_tricks if defense_tricks else []
        self.cards_on_table = cards_on_table if cards_on_table else []
        self.n_cards = n_cards
        self.n_players = len(self.players)
        self.n_tricks = self.n_cards // self.n_players
        self.n_cards_per_trick = self.n_cards // self.n_tricks
        self.table_color = None
        if cards_on_table:
            self.table_color = cards_on_table[0].color

    def get_next_to_play_idx(self):
        i_start = next(k for k, player in enumerate(self.players) if player.first_to_play)
        player_idx = (i_start + len(self.cards_on_table))%self.n_players
        return player_idx

    def reset_game(self):
        self.attack_tricks = []
        self.defense_tricks = []

    def distribute_cards(self):
        cards_ids = [id for id in range(self.n_cards)]
        np.random.shuffle(cards_ids)
        if self.n_cards == 12:
            cards = [BeloteCard12(id) for id in cards_ids]
        elif self.n_cards == 32:
            cards = [BeloteCard32(id) for id in cards_ids]
        for i, player in enumerate(self.players):
            player.hand = cards[i*self.n_tricks : i*self.n_tricks + self.n_tricks]
        candidates = [player for player in self.players]
        taker = candidates[0]
        taker.team = 'attack'
        taker.first_to_play = True
        if self.n_players < 4:
            for player in [p for p in self.players if p != taker]:
                player.team = 'defense'
        elif self.n_players ==4:
            candidates[1].team = 'defense'
            candidates[2].team = 'attack'
            candidates[3].team = 'defense'

    def give_trick_to_win_team(self):
        for i, player in enumerate(self.players):
            if player.first_to_play == True:
                k = i
        # print(f"table color : {self.table_color}")
        if 'pique' in [card.color for card in self.cards_on_table]:
            win_card_idx = np.argmax([card.value if card.color == 'pique' else 0 for card in self.cards_on_table])
        else:
            win_card_idx = np.argmax([card.value if card.color == self.table_color else 0 for card in self.cards_on_table])
        win_player = self.players[(k + win_card_idx)%self.n_players]
        win_team = win_player.team
        win_team_tricks_att = f'{win_team}_tricks'

        self.__getattribute__(win_team_tricks_att).extend(self.cards_on_table)

        for player in self.players:
            if player == win_player:
                player.first_to_play = True
            else:
                player.first_to_play = False
        # print(f"player {win_player.id} wins the trick.")
        # trick_reward = sum(card.value for card in self.cards_on_table)
        # if win_team == 'attack':
        #     print(f"reward for attack : {trick_reward}")
        #     print(f"reward for defense : {-trick_reward}")
        # else:
        #     print(f"reward for attack : {-trick_reward}")
        #     print(f"reward for defense : {trick_reward}")
            
    def reset_table(self):
        self.cards_on_table = []
        self.table_color = None


    def step(self, action):
        player = self.players[self.get_next_to_play_idx()]
        card = next((c for c in player.get_legal_moves(self) if c.id == action))
        player.play_card(card, self)

        if len(self.cards_on_table) == self.n_cards_per_trick:
            self.give_trick_to_win_team() # (updates self.reward)
            self.reset_table() # (updates cards on table)
        # if len(self.attack_tricks) + len(self.defense_tricks) == self.n_cards:
            # print("Game Finished.")


    def random_step(self):
        player = self.players[self.get_next_to_play_idx()]
        player.naive_move(self)

        if len(self.cards_on_table) == self.n_cards_per_trick:
            self.give_trick_to_win_team() # (updates self.reward)
            self.reset_table() # (updates cards on table)

        # if len(self.attack_tricks) + len(self.defense_tricks) == self.n_cards:
            # print("Game Finished.")

    def playout(self):
        while len(self.attack_tricks) + len(self.defense_tricks) != self.n_cards:
            self.random_step()

        return {
            "attack reward": sum([c.value for c in self.attack_tricks]),
            "defense reward" : sum([c.value for c in self.defense_tricks])
            }