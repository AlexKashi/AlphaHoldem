from abc import abstractmethod, ABC, ABCMeta
import re
import Hand
import low_level_functions as llf
from includes import *
import heapq
import main as main
from collections import deque
from treys import Card
import statistics

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1
    def pop(self):
        return heapq.heappop(self._queue)[-1] # pop method returns the smallest item, not the largest

class Item:
    def __init__(self, action, position):
        self.action = action
        self.position = position
    def __repr__(self):
        return 'Item({!r})'.format(self.name)

# class EvaluationData():

#     def __init__(self, table_event):
#         self.table_event = table_event
#         self.he = ''
#         self.evaluation = ''
#         self.rc = '' 
#         self.score_desc = '' 
#         self.player_action = ''
    
    # def setAll(self, he, evaluation, rc, score_desc, player_action):
    #     self.he = he
    #     self.evaluation = evaluation
    #     self.rc = rc
    #     self.score_desc = score_desc
    #     self.player_action = player_action


class Player():

    game_state = {'hand_no': '',
                'dealer_position': '',
                'action_preflop': '',
                'flop1': '',
                'flop2': '',
                'flop3': '',
                'action_flop': '',
                'turn': '',
                'action_turn': '',
                'river': '',
                'action_river': '',
                'p1': {'position_showdown': '', 'cards': ['', ''] } ,
                'p2': {'position_showdown': '', 'cards': ['', ''] }, 
                'p3': {'position_showdown': '', 'cards': ['', ''] }, 
                'winners': []
                }
    player_list = []
    level_raises = {0:0, 1:0, 2:0}
    is_new_game_u = False
    def __init__(self, ID, name, card_holding, position, GHB_file, cards,mwm,stack_size = 50):
        
        self.hand_num = 0
        self.cards = cards # cards passed in are all cards used in game
        self.ID = ID  ## acts as position tracker using 0 and 1
        self.name = name
        self.card_holding = card_holding # blank
        self.position = '' # blank
        self.GHB_file = GHB_file # give_hand_bot file
        self.mwm = mwm  ## UNKNOWN PURPOSE
        self.stack_size = stack_size
        #self.game_state = []
        self.dealer_status = False
        self.perspective_opposing_player = ['', ''] ## one for each of the opposing players. First is to left of this player. 
        #self.available_options = []
        self.evaluation_preflop = {'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
        self.evaluation_flop = {'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
        self.evaluation_turn = {'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
        self.evaluation_river = {'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
        self.round = {'moves_i_made_in_this_round_sofar': '', 'possible_moves': set([]), 'raises_owed_to_me': 0, "raises_i_owe": 0}
        self.action = None
        self.is_new_game_pp = False
        self.he = None
        

    def hand_evaluate(self, card_holding, name, event, first_meeting, first_player):
        if first_player:
            Player.level_raises = {0:0, 1:0, 2:0}
        if self.is_new_game_pp:
            self.make_new_game_per_player()
            self.make_new_round()
            self.he = Hand.HandEvaluation(card_holding, name, event) #Unique to player instance
        if self.is_new_game_u:
            print("\n\n**************New Game****************\n\n") 
            self.make_new_game()
        if self.is_new_round(first_meeting):
            self.make_new_round()
        
        evaluation, rc, score_desc, hand, board = self.he.get_evaluation(event)
        self.set_attributes(evaluation, self.he, rc, score_desc, event)
        self.populatePlayerPossibleMoves(event)
        self.calc_raises_i_face()
        #self.getPerceivedRange(player1, player2) # Only use this if coming from CTB0 (learner)  
        player_action = self.take_action(self.he, rc, score_desc, event) 
        self.set_player_action(event, player_action)
        self.debug_print(player_action, hand, board)     
        return self.he, rc, score_desc, player_action

    def __str__(self):
        st = self.ID, self.name, self.position, self.stack_size
        # return 'ID: {}, Position: {}, \n\tEvaluation-Preflop (score): {}, \n\tRound: {}'.format(str(self.ID), str(self.position), str(self.evaluation_preflop['evaluation']), str(self.round))
        return 'ID: {}, \tEval-Preflop: {}, \tEval-Flop: {}, \tEval-Turn: {}, \tEval-River: {}, \tAction: {}'.format(str(self.ID), str(self.evaluation_preflop['evaluation']), str(self.evaluation_flop['evaluation']), str(self.evaluation_turn['evaluation']), str(self.evaluation_river['evaluation']),str(self.action) )    

    def debug_print(self, player_action, hand, board):
        if (str(player_action)) != 'None':
            #print(self, "\tplayer_action: ", player_action, "\taction_where: ", player_action.round_game)
            print(self, "\taction_where: ", player_action.round_game)
            print()
        else:
            print(self) #, "\tplayer_action", "f")
            print(len(self.he.official_board))
            if len(self.he.official_board) > 0:
                
                print("\tMy_Cards: {} \n\tBoard: {}".format(Card.print_pretty_cards([self.he.card_a, self.he.card_b]), Card.print_pretty_cards([card for card in self.he.official_board])))
            else:
                print("\tMy_Cards: {}".format(Card.print_pretty_cards([self.he.card_a, self.he.card_b])))
            print()

    def is_new_round(self, first_meeting):
        x = False
        for key, value in first_meeting.items():
            if value == True:
                x = True
        return x 
        
    def make_new_round(self):
        self.round = {'moves_i_made_in_this_round_sofar': '', 'possible_moves': set([]), 'raises_owed_to_me': 0, "raises_i_owe": 0}

    def make_new_game_per_player(self):
        self.action = None
        self.evaluation_preflop = {'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
        self.evaluation_flop = {'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
        self.evaluation_turn = {'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
        self.evaluation_river = {'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}

    def make_new_game(self):
        
        for i in range(1, 4):
            self.game_state['p'+str(i)]['position_showdown'] = ''
            self.game_state['p'+str(i)]['cards'][0] = ''
            self.game_state['p'+str(i)]['cards'][1] = ''
        
        for att in self.game_state:
            if att != 'p1' and att != 'p2' and att != 'p3':
                self.game_state[att] = ''

    def set_attributes(self, evaluation, he, rc, score_desc, event):
        if event == 'Preflop':
            if self.evaluation_preflop["he"] == '':
                self.evaluation_preflop["he"] = he
                self.evaluation_preflop["rc"] = rc
                self.evaluation_preflop["score_desc"] = score_desc
                self.evaluation_preflop["evaluation"] = evaluation
        elif event == 'Flop':
            if self.evaluation_flop["he"] == '':
                self.evaluation_flop["he"] = he
                self.evaluation_flop["rc"] = rc
                self.evaluation_flop["score_desc"] = score_desc
                self.evaluation_flop["evaluation"] = evaluation
        elif event == 'Turn':
            if self.evaluation_turn["he"] == '':
                self.evaluation_turn["he"] = he
                self.evaluation_turn["rc"] = rc
                self.evaluation_turn["score_desc"] = score_desc
                self.evaluation_turn["evaluation"] = evaluation
        elif event == 'River':
            if self.evaluation_river["he"] == '':
                self.evaluation_river["he"] = he
                self.evaluation_river["rc"] = rc
                self.evaluation_river["score_desc"] = score_desc
                self.evaluation_river["evaluation"] = evaluation

    # Previewing game_state for raises made previously and counting possible moves can be made as a result. 
    # Also populates player object with possible moves they may make after having seen the moves made so far.
    # Returns: count of raises made so far in current round  ~
    def populatePlayerPossibleMoves(self, event):
        bot_position = self.position  
        bot_position_num = self.stposition_to_numposition(bot_position)

        last_seq_move = None
        if event == 'Preflop':    
            last_seq_move = self.game_state['action_preflop'] 
        elif event == 'Flop':
            last_seq_move = self.game_state['action_flop']
        elif event == 'Turn':
            last_seq_move = self.game_state['action_turn']
        elif event == 'River':
            last_seq_move = self.game_state['action_river']

        if(llf.count_r(last_seq_move) > 3):
            print("error: num or raises cannot be =", llf.count_r(last_seq_move), "\t", "bot_position",bot_position_num)
        elif(llf.count_r(last_seq_move) == 3):
            #print("llf.count_r("+last_seq_move+") == 3")
            self.round['possible_moves'].clear()
            self.round['possible_moves'].add('c')
            self.round['possible_moves'].add('f')
            
        else:
            #print("llf.count_r("+last_seq_move+") < 3")
            self.round['possible_moves'].clear()
            self.round['possible_moves'].add('r')
            self.round['possible_moves'].add('c')
            self.round['possible_moves'].add('f')    
            

    def calc_raises_i_face(self):
        bot_position_num = self.stposition_to_numposition(self.position)
        my_lr_value = self.level_raises[bot_position_num]
        highest_lr_value, highest_lr_bot = highest_in_LR()
        add_me = highest_lr_value - my_lr_value
        self.round['raises_i_owe'] = self.round['raises_i_owe'] + add_me

    def set_player_action(self, event, player_action):
        if event == 'Preflop':
            if self.evaluation_preflop["player_action"] == '':
                self.evaluation_preflop["player_action"] = player_action
        elif event == 'Flop':
            if self.evaluation_flop["player_action"] == '':
                self.evaluation_flop["player_action"] = player_action
        elif event == 'Turn':
            if self.evaluation_turn["player_action"] == '':
                self.evaluation_turn["player_action"] = player_action
        elif event == 'River':
            if self.evaluation_river["player_action"] == '':
                self.evaluation_river["player_action"] = player_action
    

    
        

    def stposition_to_numposition(self, st_bot_position):
        if st_bot_position == 'BTN':
            return 0
        elif st_bot_position == 'SB': 
            return 1
        elif st_bot_position == 'BB':
            return 2

    

    def is_possible(self, move):
        move_possible = False
        for item in self.round['possible_moves']:
            if item == move:
                return True
                break
        return move_possible    

    def make_decision(self, round_game, bot_position_num):
        range_structure = None
        if round_game == 'Preflop': 
            range_structure = preflop_range
            which_eval = self.evaluation_preflop
        elif round_game == 'Flop':
            range_structure = flop_range
            which_eval = self.evaluation_flop
        elif round_game == 'Turn' or round_game == 'River':
            range_structure = turn_river
            if self.evaluation_river == '' and self.evaluation_river != None:
                which_eval = self.evaluation_turn

            else:
                which_eval = self.evaluation_river

        # print("\n\nround_game", round_game)
        # print("\traises_i_owe:", self.round['raises_i_owe'])
        # print("\n\tcall: ", range_structure['calling'][self.round['raises_i_owe']][bot_position_num], "\traises_i_owe:", self.round['raises_i_owe'])
        # print(type(which_eval["evaluation"]))
        if (which_eval["evaluation"] < range_structure['betting'][self.round['raises_i_owe']][bot_position_num]) and (self.is_possible('r')): 
            
            #print("case 1")
            act = Bet(limit, round_game ,self)
            self.round['moves_i_made_in_this_round_sofar'] += 'r'
            return act
        elif which_eval["evaluation"] < range_structure['calling'][self.round['raises_i_owe']][bot_position_num] and (self.is_possible('c')): 
            #print("case 2")
            act = Call(limit, round_game,self)
            self.round['moves_i_made_in_this_round_sofar'] += 'c'
            return act
        else: 
            #print("case 3")
            act = Fold(round_game, self)
            self.round['moves_i_made_in_this_round_sofar'] += 'f'
            return act
           
            
    def take_action(self, he, rc, score_desc, round_game):
        last_seq_move = ''
        dealer_position = self.game_state['dealer_position']
        if round_game == 'Preflop':    
            last_seq_move = self.game_state['action_preflop'] 
        elif round_game == 'Flop':
            last_seq_move = self.game_state['action_flop']
        elif round_game == 'Turn':
            last_seq_move = self.game_state['action_turn']
        elif round_game == 'River':
            last_seq_move = self.game_state['action_river']

        bot_position = self.position  # DEBUG: self.position has not been assigned
        bot_position_num = self.stposition_to_numposition(bot_position)
        q = PriorityQueue()

        #print('\npossible_moves' , self.round['possible_moves'])
        self.make_decision(round_game, bot_position_num)


def save_player_list(my_list):
    Player.player_list = my_list               
            
    

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

def highest_in_LR():
    highest_lr_bot = 0
    highest_lr_value = 0
    
    for key, value in Player.level_raises.items():
        if value > highest_lr_value:
            highest_lr_value = value
            highest_lr_bot = key
    return highest_lr_value, highest_lr_bot

def get_active_players(player):
    player_list = Player.player_list # Note: (each player gets copy of player list)
    active_players_nums = []
    for p in player_list:
        if 'f' not in p.round['moves_i_made_in_this_round_sofar']:
            active_players_nums.append(player.stposition_to_numposition(p.position))
    
    return active_players_nums

def get_next_player_left_in_game(player):
    active_players_nums = get_active_players(player)
    for p_num in active_players_nums:
        if player.stposition_to_numposition(player.position) == p_num:
            return getNext(active_players_nums, player.stposition_to_numposition(player.position))
            break

def getNext(my_list, current):
    for i in range(len(my_list)):
        if my_list[i] == current:
            if i == (len(my_list)-1):
                return my_list[0]
            else:
                return my_list[i+1]


def send_file(action_string, player, position, directory):
    btc_file = 'botToCasino'
    btc_num = Player.stposition_to_numposition(player, position)
    next_player_num = get_next_player_left_in_game(player) # accounts for players that have folded and loop-around
    # we want to write to next_player_num to make sure he sees the action of player acting before him
    file_name_ownPlayer = directory + btc_file + str(player.ID)
    #file_name_nextPlayer = directory + btc_file + str(next_player_num)  

    # Using player.ID as above instead of player.position at table solved problem which allowed us to now use
    # 3 or more rounds without any issues: 
    #   commit 8a70ebab6587f0e0aa2287d26833e5c2326c4d16 (HEAD -> master)
    #   Author: Gary Harney <garyjh126@gmail.com>
    #   Date:   Tue Feb 12 13:41:26 2019 +0000

    #       Bug fixed: nRounds > 3


    try:
        with open(file_name_ownPlayer, 'wt') as f:
            f.write(action_string)
            f.close()
    except:
        print("Could not write", action_string, "to ", btc_file + str(btc_num), "from", position)

#INTERFACE
class Action(ABC):

    __metaclass_ = ABCMeta

    #communication_files_directory='/usr/local/home/u180455/Desktop/Project/MLFYP_Project/MLFYP_Project/pokercasino/botfiles'
    communication_files_directory = ''

    try: 
        communication_files_directory = main.path_to_file_changed2
    except: 
        communication_files_directory = '/home/gary/Desktop/MLFYP_Project/MLFYP_Project/pokercasino/botfiles/'

    @abstractmethod
    def determine_table_stats(self): pass

    @abstractmethod
    def get_action_of_preceding_player(self): pass

    @abstractmethod
    def populate_regret_table(self): pass

    @abstractmethod
    def __str__(self): 
        return "A"    


class Bet(Action):
    count_bets = {"Preflop":0, "Flop":0, "Turn":0, "River":0}
    def __init__(self, amount, round_game, player):
        self.amount = amount
        self.player = player
        self.count_bets[round_game] = self.count_bets[round_game] + 1
        self.round_game =  round_game
        self.populate_levelRaises()
        player.round['raises_owed_to_me'] = player.round['raises_owed_to_me'] + 1
        player.action = self
        player.round['raises_i_owe'] = 0
        send_file('r', self.player, self.player.position, self.communication_files_directory)
       # print("Player: {} bets {}".format(player.ID, amount))

    def populate_levelRaises(self):
        bot_no = Player.stposition_to_numposition(self.player, self.player.position)
        highest_value_LR, _ = highest_in_LR()
        Player.level_raises[bot_no] = highest_value_LR + 1

    def __str__(self): 
        return "r"

    def determine_table_stats(self):
        pass

    
    def populate_regret_table(self):
        pass

    def determine_action(self): pass

    def get_action_of_preceding_player(self): pass

class Call(Action):
    count_calls = {"Preflop":0, "Flop":0, "Turn":0, "River":0}
    def __init__(self, amount, round_game, player):
        self.amount = amount
        self.player = player
        self.count_calls[round_game] = self.count_calls[round_game] + 1
        self.round_game =  round_game
        self.populate_levelRaises()
        player.action = self
        player.round['raises_i_owe'] = 0
        send_file('c', self.player, self.player.position, self.communication_files_directory)

       # print("Player: {} calls".format(player.ID))

    def populate_levelRaises(self):
        last_seq_move = ''
        if self.round_game == 'Preflop':    
            last_seq_move = Player.game_state['action_preflop']
        elif self.round_game == 'Flop':
            last_seq_move = Player.game_state['action_flop']
        elif self.round_game == 'Turn':
            last_seq_move = Player.game_state['action_turn']
        elif self.round_game == 'River':
            last_seq_move = Player.game_state['action_river']

        if self.c_call(last_seq_move):
            bot_no = Player.stposition_to_numposition(self.player, self.player.position)
            highest_value_LR, _ = highest_in_LR()
            Player.level_raises[bot_no] = highest_value_LR
            
    
    def c_call(self, last_seq_move):
        c_is_call = False
        if len(last_seq_move) > 1:
            if last_seq_move[-1] == 'r' or last_seq_move[-2] == 'r':
                c_is_call = True
        elif len(last_seq_move) == 1:
            if last_seq_move[-1] == 'r':
                c_is_call = True
        return c_is_call
        
        

    def __str__(self): 
        return "c"

    def determine_action(self): 
        pass

    def determine_if_this_action_works(self):
        pass

    def determine_table_stats(self):
        pass

    def get_action_of_preceding_player(self):
        pass

    def populate_regret_table(self):
        pass


class Fold(Action):
    count_folds = {"Preflop":0, "Flop":0, "Turn":0, "River":0}
    def __init__(self, round_game, player):
        self.player = player
        self.count_folds[round_game] = self.count_folds[round_game] + 1
        self.round_game =  round_game
        player.action = self
        send_file('f', self.player, self.player.position, self.communication_files_directory)


    def determine_table_stats(self):
        pass

    def get_action_of_preceding_player(self):
        pass

    def __str__(self): 
        return "f"

    def populate_regret_table(self):
        pass

