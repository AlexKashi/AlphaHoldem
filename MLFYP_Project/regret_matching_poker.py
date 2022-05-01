from __future__ import division
import re
from random import random
import numpy as np
import pandas as pd
import os
import uuid
from abc import abstractmethod, ABCMeta
import pyinotify
from treys import *
import regret_matching_poker

most_recent_file_changed = ''

class HandEvaluation():

    def __init__(self, card_holding, playerID, evaluation = None, event = "Preflop"):
        self.card_holding = card_holding
        self.card_a, self.card_b = self.parse_cards()
        self.evaluation = self.evaluate(event)
        self.rc = ''
        self.event = event
        self.playerID = playerID

    def __str__(self):
        st = "{}\t\t Player ID: {}\n".format(self.event, self.playerID) if self.event=="Preflop" else "{}\t\t\t Player ID: {}\n".format(self.event, self.playerID) 
        st += "Cards: {}{}\n".format(Card.int_to_pretty_str(self.hand[0]), Card.int_to_pretty_str(self.hand[1]))
        st += "Board {}{}{}\n".format(Card.int_to_pretty_str(self.board[0]), Card.int_to_pretty_str(self.board[1]), Card.int_to_pretty_str(self.board[2]))
        st += "Evaluation: {} ({}), Rank_Class: {}, \n".format(self.evaluation[0], self.evaluation[2], self.evaluation[1]) if self.event=="Preflop" else "Evaluation: {} ({}), Rank_Class: {}, \n-----------------".format(self.evaluation[0], self.evaluation[2], self.evaluation[1])
        return st

    def parse_cards(self):
        a, b = self.card_holding.get_card(0) , self.card_holding.get_card(1)
        a_rank, a_suit = a
        b_rank, b_suit = b
        
        a_card = Card.new(str(a_rank) + str(a_suit))
        b_card = Card.new(str(b_rank) + str(b_suit))
    
        return [a_card, b_card]

    def setup_board(self, board, random, hand = None):
        #Example board -- DEBUG
        b = []
        if board == None and random == 'False': #FLOP
            #import from file giving hand status
            b = [
                Card.new('Ah'),
                Card.new('Kd'),
                Card.new('Jc')
            ]
        if board == None and random == 'True': #PREFLOP
            deck = Deck()
            b = deck.draw(3)
        return b

    def evaluate(self, event):
        evaluator = Evaluator()
        if event == "Preflop":
            self.hand = self.parse_cards()
            self.board = self.setup_board(None, 'True', self.hand)
            evaluation = evaluator.evaluate(self.hand, self.board)
            rc = self.rank_class(evaluator, evaluation)
            score_desc = evaluator.class_to_string(rc)
            return evaluation, rc, score_desc, event 
        
        elif event == "Flop":
            self.hand = self.parse_cards()
            self.board = self.setup_board(None, 'False')     # Only pass in none for now
            evaluation = evaluator.evaluate(self.hand, self.board)
            rc = self.rank_class(evaluator, evaluation)
            score_desc = evaluator.class_to_string(rc)
            return evaluation, rc, score_desc, event

    def rank_class(self, evaluator, evaluation):
        rc = evaluator.get_rank_class(evaluation)
        return rc

    def get_evaluation(self):
        return self.evaluation




def get_status_from_file(file_name):
    data = ''
    with open('/usr/local/home/u180455/Desktop/Project/MLFYP_Project/MLFYP_Project/pokercasino/botfiles/' + file_name, 'rt') as f:
        data = f.read()
    return data


'''
    Use regret-matching algorithm to play Poker
'''


class Game:

    cards = []

    def __init__(self, max_game=5):
        global cards
        cards = self.create_cards_for_game() 
        Player1 = Player(uuid.uuid1() ,'Adam', CardHolding('-','-','-','-','-'), 'BTN', '/give_hand_bot0', cards, None)
        Player2 = Player(uuid.uuid1() ,'Bill', CardHolding('-','-','-','-','-'), 'SB', '/give_hand_bot1', cards, None)
        #Player3 = Player(uuid.uuid1() ,'Chris', CardHolding('-','-','-','-','-'), 'BB', '/give_hand_bot2', cards, None)
        #Player4 = Player(uuid.uuid1() ,'Dennis', CardHolding('-','-','-','-','-'), 'CO', '/give_hand_bot3', cards, None)
        self.player_list = [Player1, Player2] #, Player3, Player4]
        positions_at_table = {0: Player1.position, 1: Player2.position} #, 2: Player3.position, 3: Player4.position} # mutable
        self.table = Table(self.player_list, positions_at_table)
        self.max_game = max_game
        self.parse_data_from_GHB()
        #self.preflop()
        #print(most_recent_file_changed)

    # def preflop(self):
    #     for player in self.player_list:             
    #         he, evaluation, rc, score_desc, _ =  player.hand_evaluate_preflop()
    #         #print(evaluation, rc, score_desc)
    #         evaluator = Evaluator()
    #         card_str = '{}{}'.format(Card.int_to_pretty_str(he.parse_cards()[0]), Card.int_to_pretty_str(he.parse_cards()[1]))
    #         print(card_str, evaluation, rc, score_desc)

            #make_action()

    # def flop(self):
    #     pass
    #     player.hand_evaluate_flop(card_holding)

    def parse_data_from_GHB(self):

        #player.take_action(card_holding)
        # self.compute_starting_random_actions(player)

        self.main_watch_manager = main_watch_manager(self.player_list)
        #print(self.player_list[0].card_holding)
        
    def create_cards_for_game(self):
        suits = ['h','c','s','d']
        li = []
        
        for rank in range(13):
            for suit in suits:
                if(rank == 8):
                    card_r = 'T'
                elif(rank == 9):
                    card_r = 'J'
                elif(rank == 10):
                    card_r = 'Q'
                elif(rank == 11):
                    card_r = 'K'
                elif(rank == 12):
                    card_r = 'A'
                else:
                    card_r = str(rank+2)
                card_str = card_r+suit
                li.append(card_str)
              
        return li

    def make_action(self):
        pass

    

class MyEventHandler(pyinotify.ProcessEvent):
    casino_to_bot_list_b1 = []
    times = 0
    def my_init(self, **kargs):
        """
        This is your constructor it is automatically called from
        ProcessEvent.__init__(), And extra arguments passed to __init__() would
        be delegated automatically to my_init().
        """
        self.player_list = kargs["player_list"]
        

    def process_IN_ACCESS(self, event):
        print "ACCESS event:", event.pathname

    def process_IN_ATTRIB(self, event):
        print "ATTRIB event:", event.pathname

    def process_IN_CLOSE_NOWRITE(self, event):
        print "CLOSE_NOWRITE event:", event.pathname

     def process_IN_MODIFY(self, event):
        print "MODIFY event:", event.pathname

    def process_IN_OPEN(self, event):
        print "OPEN event:", event.pathname

    def process_IN_CLOSE_WRITE(self, event):
        ### declaring a bot_number and event_type 
        #print(event.pathname)
        global file_changed
        arr = re.split(r'[/]',event.pathname)
        most_recent_file_changed = (arr[len(arr)-1])
        last_letter = most_recent_file_changed[len(most_recent_file_changed)-1]
        bot_number = last_letter if (last_letter =='0' or last_letter == '1') else ''
        event_type = most_recent_file_changed if bot_number == '' else most_recent_file_changed[0:len(most_recent_file_changed)-1]
        filename = str(event_type+bot_number)
        file_data = get_status_from_file(str(filename))

        if event_type == "give_hand_bot":
        
            # for i in range(len(self.player_list)):
            if bot_number == '0':
                self.player_list[0].card_holding = self.player_list[0].GHB_Parsing(file_data) #check cards
                #print(self.player_list[0].card_holding)
                he, evaluation, rc, score_desc, _ = self.player_list[0].hand_evaluate_preflop(self.player_list[0].card_holding, self.player_list[0].name)
                print(he, evaluation, rc, score_desc)

            elif bot_number == '1':
                self.player_list[1].card_holding = self.player_list[1].GHB_Parsing(file_data) #check cards
                #print(self.player_list[0].card_holding)
                he, evaluation, rc, score_desc, _ = self.player_list[1].hand_evaluate_preflop(self.player_list[1].card_holding, self.player_list[1].name)
                print(he, evaluation, rc, score_desc)

        if event_type == "casinoToBot":   
            
            if bot_number == '0':
                #self.player_list[0].game_state = self.casinoToBot_Parsing(file_data) #check cards
                pass

            elif bot_number == '1':
                self.player_list[1].game_state.append(self.casinoToBot_Parsing(file_data)) #check cards
                #print(self.player_list[1].game_state)

 

    def casinoToBot_Parsing(self, file_data):
        # <hand number> D <dealer button position> P <action by all players in order from first to 
        # act, e.g. fccrf...> F <flop card 1> F <flop 2> F <flop 3> F <flop action starting with first player to act>
        # T <turn card> T <turn action> R <river card> R <river action>

        arr = re.split(r'[DPFFFFTTRR]',file_data)
        # dictionary = {"hand_num" : arr[0],
        #                 "button" : arr[1] ,
        #                 "preflop_action" : arr[2],
        #                 "flop_card_1" : arr[3] ,
        #                 "flop_card_2" : arr[4],
        #                 "flop_card_3" : arr[5] ,
        #                 "flop_action" : arr[6] ,
        #                 "turn_card" : arr[7],
        #                 "turn_action" : arr[8],
        #                 "river_card" : arr[9],
        #                 "river_action" : arr[10]}
       
        for i in range(len(arr)):
            print(i, arr[i], arr)
            # if arr[i] not in self.casino_to_bot_list_b1:
            if(arr[i] != self.casino_to_bot_list_b1[i])
            
            #     #return block
            self.casino_to_bot_list_b1.append(arr[i])
            #     return block

        #return self.casino_to_bot_list_b1
        

class main_watch_manager():
    #build one for all players
    def __init__(self, player_list ,communication_files_directory='/usr/local/home/u180455/Desktop/Project/MLFYP_Project/MLFYP_Project/pokercasino/botfiles'):
        self.communication_files_directory = communication_files_directory
        self.player_list = player_list
        # watch manager
        wm = pyinotify.WatchManager()
        wm.add_watch(self.communication_files_directory, pyinotify.ALL_EVENTS, rec=True)

        # event handler
        kwargs = {"player_list": self.player_list}
        eh = MyEventHandler(**kwargs)  #**kwargs

        # notifier
        notifier = pyinotify.Notifier(wm, eh)
        notifier.loop()



class Player(Game):


    def __init__(self, ID, name, card_holding, position, GHB_file, cards,mwm,stack_size = 50):
        self.cards = cards
        self.ID = ID
        self.name = name
        self.card_holding = card_holding
        self.position = position
        self.strategy, self.avg_strategy,\
        self.strategy_sum, self.regret_sum = np.zeros((4, 3))
        self.list_of_actions_game = np.array([])
        self.GHB_file = GHB_file
        self.mwm = mwm
        self.stack_size = stack_size
        self.game_state = []
        
    def __str__(self):
        st = self.ID, self.name, self.position, self.stack_size
        return 'ID: {}, Name: {}, Position: {}, Stack Size: {}'.format(str(self.ID), str(self.name), str(self.position), str(self.stack_size))
        
    def hand_evaluate_preflop(self, card_holding, name):
        he = HandEvaluation(card_holding, name, event = 'Preflop') #Unique to player instance
        evaluation, rc, score_desc, event = he.get_evaluation()
        
        self.take_action_flop(he, evaluation, rc, score_desc)
        return he, evaluation, rc, score_desc, event 

    def take_action_flop(self, he, evaluation, rc, score_desc):
        # Hand strength is valued on a scale of 1 to 7462, where 1 is a Royal Flush and 7462 is unsuited 7-5-4-3-2, as there are only 7642 
        # distinctly ranked hands in poker. Once again, refer to my blog post for a more mathematically complete explanation of why this is so.
        limit = 5
        act = Action()
        position = self.position
        game_state = self.game_state
        #print(position, game_state)

        #self.CFR_table[]
        cut_lower = 3000
        cut_upper = 7000

        # How tight do I want to play? This will determine the cut values. 
        # It depends on my current position and values from the CFR table. 
        # If a hand has a low evaluation but negative regret, then it may 
        # be preferable to fold.

        if evaluation < cut_lower:  ## Account for position
            act = Bet(limit, self)
        elif evaluation >= cut_lower and evaluation < cut_upper:
            act = Call(limit, self)
        else:
            #First check if free to fold but do this in subclass
            act = Fold()

    def hand_evaluate_flop(self, card_holding, name):
        limit = 5
        he = HandEvaluation(self.card_holding, name, event = 'Flop') #Unique to player instance
        #print(he)   

        #  for starting training, if hand is "sufficiently good" (If evaluation score is good), then c/r. 
        # Otherwise fold. 
        my_eval_score = he.get_evaluation()[0]
        if my_eval_score < 3000:  ## Account for position
            act = Bet(limit, self)
        elif my_eval_score >= 3000 and my_eval_score < 7000:
            act = Call(limit, self)
        else:
            #First check if free to fold but do this in subclass
            act = Fold()

    def GHB_Parsing(self, GHB_Status):
        
        # GHB_STATUS = <hand number>D<button position>A<holecard1>B<holecard2>
        # cards are 4 * rank + suit where rank is 0 .. 12 for deuce to ace, and suits is 0 .. 3

        #restrict to just give_hand_bot files
        deck_size = 52
        arr = re.split(r'[DAB]',GHB_Status)
        suits = ['h','c','s','d']
        card_a = arr[2] #card from file / REPRESENTS INDEX OF SELF.CARDS
        card_a_suit = ''
        card_a_rank = ''
        card_b = arr[3] #card from file / REPRESENTS INDEX OF SELF.CARDS
        card_b_suit = ''
        card_b_rank = ''
        a,b,c,x,y,z = ('', '', '', '', '', '')
        for card in self.cards:
            if(str(self.cards.index(card)) == card_a):
                if(len(card) == 2):
                    a,b = card
                # elif(len(card) == 3):
                #     a,b,c = card
                #     a = a+b
                #     b = c
            elif(str(self.cards.index(card))== card_b):
                if(len(card) == 2):
                    x,y = card
                elif(len(card) == 3):
                    x,y,z = card
                    x = x+y
                    y = z
        card_a_rank = a
        card_a_suit = b
        card_b_rank = x
        card_b_suit = y
        self.card_holding = CardHolding(self.name,card_a_suit,card_a_rank,card_b_suit, card_b_rank)
       
        return self.card_holding

    
    
class CardHolding(Player):

    def __init__(self, name, first_card_suit, first_card_rank, second_card_suit, second_card_rank):
        self.name = name
        self.first_card_suit = first_card_suit
        self.first_card_rank = first_card_rank
        self.second_card_suit = second_card_suit
        self.second_card_rank = second_card_rank

    def __str__(self):
        first_card = self.first_card_suit, self.first_card_rank
        second_card = self.second_card_suit, self.second_card_rank
        st = 'Name: {}'.format(self.name) + '\tFirst Card: {}'.format(str(first_card)) + '\tSecond Card: {}\n'.format(str(second_card))
        return (str(st))
    
    def get_card(self, card_no):
        if card_no == 0:
            return self.first_card_rank + self.first_card_suit
            
        else:
            return self.second_card_rank,self.second_card_suit
            
    

class Table(Game):

    num_of_players = 0

    def __init__(self, player_list, positions_at_table):
        self.player_list= list(player_list)
        #print(type(player_list))
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


#INTERFACE
class Action(object):

    __metaclass_ = ABCMeta

    communication_files_directory='/usr/local/home/u180455/Desktop/Project/MLFYP_Project/MLFYP_Project/pokercasino/botfiles'
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

    def __init__(self, amount, player):
        self.amount = amount
        self.player = player

    def determine_table_stats(self):
        pass

    def send_file(self):
        btc_file = "botToCasino0" if self.player.name == "Adam" else "botToCasino1"
        file_name = self.communication_files_directory + btc_file 
        with open(file_name, 'wt') as f:
            f.write('r')

    def populate_regret_table(self):
        pass


class Call(Action):
    def __init__(self, amount, player):
        self.amount = amount
        self.player = player

    def determine_if_this_action_works(self):
        pass

    def determine_table_stats(self):
        pass

    def send_file(self):
        btc_file = "botToCasino0" if self.player.name == "Adam" else "botToCasino1"
        file_name = self.communication_files_directory + btc_file 
        with open(file_name, 'wt') as f:
            f.write('c')

    def get_action_of_preceding_player(self):
        pass

    def populate_regret_table(self):
        pass


class Fold(Action):

    def __init__(self):
        pass

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

    game = Game()

    #os.remove("strategy_stats.txt")
    #print('==== Use simple regret-matching strategy === ')
    #game.play()
    #print('==== Use averaged regret-matching strategy === ')
    #game.conclude()
    #game.play(avg_regret_matching=True)
