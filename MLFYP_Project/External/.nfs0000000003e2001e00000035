from __future__ import division
from random import random
import numpy as np
import pandas as pd
import os
import uuid
from abc import abstractmethod, ABCMeta



'''
    Use regret-matching algorithm to play Poker
'''


class Game:
    def __init__(self, max_game=5):

        Player1 = Player(uuid.uuid1() ,'Adam', CardHolding('-','-','-','-'), 'BTN')
        Player2 = Player(uuid.uuid1() ,'Bill', CardHolding('-','-','-','-'), 'SB')
        Player3 = Player(uuid.uuid1() ,'Chris', CardHolding('-','-','-','-'), 'BB')
        Player4 = Player(uuid.uuid1() ,'Dennis', CardHolding('-','-','-','-'), 'CO')
        player_list = [Player1, Player2, Player3, Player4]
        positions_at_table = {0: Player1.position, 1: Player2.position, 2: Player3.position, 3: Player4.position} # mutable

        # Create more players for Poker game
        self.table = Table(player_list, positions_at_table)
        self.max_game = max_game


class Table(Game):

    num_of_players = 0

    def __init__(self, player_list, positions_at_table):
        self.player_list= player_list.copy()
        for i in player_list:
            Table.num_of_players += 1
        self.positions_at_table = positions_at_table.copy()

    def get_players_at_table(self):
        return self.player_list

    def get_player_at_position(self, position):
        for i in self.player_list:
            if i.position == position:
                return i

    def get_player_by_ID(self, ID):
        for player in self.player_list:
            if player.ID == ID:
                return player

    def get_position_of_player(self, player):
        for i in self.player_list:
            if i == player:
                return i.position

    def get_number_of_players(self):
        return self.num_of_players

    def rotate(self):
        keys = self.positions_at_table.keys()
        values = self.positions_at_table.values()
        shifted_values = values.insert(0, values.pop())
        new_positions_at_table = dict(zip(keys, shifted_values))
        self.positions_at_table = new_positions_at_table

    def remove_player(self, player):
        for i in self.player_list:
            if i == player:
                self.player_list.remove(player)
        for index, pos in self.positions_at_table.items():
            if pos == player.position:
                del self.positions_at_table[pos]
                self.reinstantiate_positions_at_table(pos)

    def reinstantiate_positions_at_table(self, player_to_remove):
        keys = []
        for index in range(len(self.positions_at_table)):
            keys.append(index)
        values = self.positions_at_table.values()
        new_dict = dict(zip(keys, values))
        self.positions_at_table = new_dict


class Player(Game):


    def __init__(self, ID, name, card_holding, position, stack_size = 50):
        self.ID = ID
        self.name = name
        self.card_holding = card_holding
        self.position = position
        self.strategy, self.avg_strategy,\
        self.strategy_sum, self.regret_sum = np.zeros((4, 3))
        self.list_of_actions_game = np.array([])
        self.stack_size = stack_size
        self.action = ''

    def __str__(self):
        return self.name

    def take_action(self):
        pass


#INTERFACE
class Action(Player):

    __metaclass_ = ABCMeta


    @abstractmethod
    def determine_action(self): pass

    @abstractmethod
    def determine_table_stats(self): pass

    @abstractmethod
    def send_file(self): pass

    @abstractmethod
    def get_action_of_preceding_player(self): pass

    @abstractmethod
    def populate_regret_table(self): pass


class Bet(Action):

    def __init__(self, amount):
        self.amount = amount

    def determine_action(self):
        pass

    def determine_table_stats(self):
        pass

    def send_file(self):
        pass

    def get_action_of_preceding_player(self):
        pass

    def populate_regret_table(self):
        pass




class Call(Action):
    def __init__(self, amount):
        self.amount = amount

    def determine_if_this_action_works(self):
        pass

    def determine_table_stats(self):
        pass

    def send_file(self):
        pass

    def get_action_of_preceding_player(self):
        pass

    def populate_regret_table(self):
        pass

class Fold(Action):

    def __init__(self, amount):
        self.amount = amount

    def determine_action(self):
        pass

    def determine_table_stats(self):
        pass

    def send_file(self):
        pass

    def get_action_of_preceding_player(self):
        pass

    def populate_regret_table(self):
        pass



class CardHolding(Player):

    def __init__(self, first_card_suit, first_card_rank, second_card_suit, second_card_rank):
        self.first_card_suit = first_card_suit
        self.first_card_rank = first_card_rank
        self.second_card_suit = second_card_suit
        self.second_card_rank = second_card_rank

    def __str__(self):
        return self.first_card_suit, self.first_card_rank, self.second_card_suit, self.second_card_rank



class PokerRound(Table):

    poker_round_count = 0

    def __init__(self, sb, bb, pot):
        PokerRound.poker_round_count += 1
        self.sb = sb
        self.bb = bb
        self.pot = pot

    def deal_holecards(self):
        pass



if __name__ == '__main__':


    # Establish connection with interface:
    

    game = Game()

    #os.remove("strategy_stats.txt")
    #print('==== Use simple regret-matching strategy === ')
    #game.play()
    #print('==== Use averaged regret-matching strategy === ')
    #game.conclude()
    #game.play(avg_regret_matching=True)
