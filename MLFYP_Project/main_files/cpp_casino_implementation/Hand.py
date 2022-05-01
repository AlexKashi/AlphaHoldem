import regret_matching_poker as rm_poker
from treys import *
import Player as p
from low_level_functions import from_num_to_cardstring
class HandEvaluation():

    def __init__(self, card_holding, playerID, event, evaluation = None):
        self.official_board = []
        self.card_holding = card_holding
        self.card_a, self.card_b = self.parse_cards()
        self.summary = self.evaluate(event)
        self.evaluation = self.summary[0]
        self.rc = self.summary[1]
        self.event = self.summary[3]
        self.playerID = playerID  # player name
        self.flop_cards, self.turn_card, self.river_card = None, None, None

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

    def parse_flop_cards(self):
        #work on this parsing Saturday Feb 02
        a, b, c = p.Player.game_state['flop1'], p.Player.game_state['flop2'], p.Player.game_state['flop3']
        card1 = from_num_to_cardstring(a)
        card2 = from_num_to_cardstring(b)
        card3 = from_num_to_cardstring(c)
        if card1 == '' or card2 == '' or card3 == '':
            deck = Deck()
            b = deck.draw(3)
            return Card.int_to_str(b[0]), Card.int_to_str(b[1]), Card.int_to_str(b[2])
        else:
            return card1, card2, card3

    def parse_turn_river_cards(self, event):
        a = None
        if event == 'Turn':
            a = p.Player.game_state['turn']
        elif event == 'River':
            a = p.Player.game_state['river']
        card1 = from_num_to_cardstring(a)
        return (card1)

    def setup_board(self, board, random, hand = None):
        #Example board -- DEBUG
        b = []
        if random == 'False': #FLOP
            #import from file giving hand status
            for card in board:
                c = Card.new(card)
                b.append(c)
            self.official_board = b

        if board == None and random == 'True': #PREFLOP
            deck = Deck()
            b = deck.draw(3)
            while(self.is_duplicates(b, hand)):
                b = deck.draw(3)
        
        return b

    def is_duplicates(self, board, hand):
        duplicate = False
        for card_b in board:
            for card_h in hand:
                if card_b == card_h:
                    duplicate = True

        return duplicate

    def evaluate(self, event):
        evaluation = None
        rc = None
        score_desc = None
        evaluator = Evaluator()
        self.hand = self.parse_cards()
        self.board = ''
        
        if event == 'Flop':
            self.flop_cards = self.parse_flop_cards()
            if(self.flop_cards != 'Blank'):
                self.board = self.setup_board(self.flop_cards, 'False', self.hand)  
        elif event == 'Turn':
            self.turn_card = self.parse_turn_river_cards(event)
            board = self.board_join(self.flop_cards, self.turn_card)
            if(self.turn_card != 'Blank'):
                self.board = self.setup_board(board, 'False', self.hand)
        elif event == 'River':
            self.river_card = self.parse_turn_river_cards(event)
            board = self.board_join(self.board_join(self.flop_cards, self.turn_card), self.river_card)
            if(self.river_card != 'Blank'):
                self.board = self.setup_board(board, 'False', self.hand)

        
        try: 
            evaluation = None
            if event == 'Preflop':
                evaluation = self.do_mean_evaluation(self.hand, evaluator)
            else:
                evaluation = evaluator.evaluate(self.hand, self.board)
            rc = self.rank_class(evaluator, evaluation)
            score_desc = evaluator.class_to_string(rc)
            
        except KeyError:
            print("KeyError:", self.hand, self.board)

        
        return evaluation, rc, score_desc, self.hand, self.board

    def do_mean_evaluation(self, hand, evaluator):
        
        total_sum_evals = 0
        list_evaluations = []
        n = 100
        for i in range(n):
            self.board = self.setup_board(None, 'True', self.hand)
            evaluation = evaluator.evaluate(hand, self.board)
            list_evaluations.append(evaluation)
            total_sum_evals = total_sum_evals + evaluation
        mean = total_sum_evals/n
        which_eval = self.closest_to_mean(mean, list_evaluations)
        return which_eval
         
    def closest_to_mean(self, mean, list_evaluations):
        sdfm = {'eval': None, 'smallest_distance_from_mean':None}
        sdfm['smallest_distance_from_mean'] = 7462
        for evaluation in list_evaluations:
            this_distance = abs(evaluation - mean)
            if(this_distance < sdfm['smallest_distance_from_mean']):
                sdfm['smallest_distance_from_mean'] = this_distance
                sdfm['eval'] = evaluation
        return sdfm['eval']

    def board_join(self, a, b):

        l1 = []
        l2 = []
        for elem_a in a:
            l1.append(elem_a)
        l2.append(b)
        l3 = l1 + l2
        return tuple(l3)

    def rank_class(self, evaluator, evaluation):
        rc = evaluator.get_rank_class(evaluation)
        return rc

    def get_evaluation(self, event):
        self.summary = self.evaluate(event)
        return self.summary
