from treys import Card, Evaluator, Deck
from itertools import combinations 
from holdem.player import Player

class HandEvaluation(Player):
    preflop_opprank_control = 5
    preflop_evaluation_mean_control = 100

    def __init__(self, cards, playerID, event, evaluation = None):
        self.evaluator = Evaluator()
        self.hand = cards
        self.create_cards_for_game() # Remaining cards after dealt two hole cards to this player. 15/02: This is updated after he in instantiated
        self.make_combinations() # all possible card permuations (1326) used to describe opponents range
        self.official_board = []
        self.summary = None
        self.evaluation = None
        self.rc = None
        self.score_desc = None
        self.hand_strength = None
        self.event = event
        self.playerID = playerID  # player name
        self.flop_cards, self.turn_card, self.river_card = None, None, None
        self.board = None # Fictional board
        # self.ew_score = None

    def make_combinations(self):
        self._combinations = list(combinations(self.deck_of_cards, 2))
        # for combo in _combinations:
        #     combo = self.parse_cards(combo[0], combo[1])

    def parse_cards(self, a, b):
        a_rank, a_suit = a
        b_rank, b_suit = b
        a_card = Card.new(str(a_rank) + str(a_suit))
        b_card = Card.new(str(b_rank) + str(b_suit))
        return [a_card, b_card]

    

    def from_num_to_cardstring(self, my_card):
        deck_size = 52
        suits = ['h','c','s','d']
        card_a_suit = ''
        card_a_rank = ''
        a,b = ('', '')
        for card in self.deck_of_cards: ## all cards in game
            if(str(self.deck_of_cards.index(card)) == my_card):
                if(len(card) == 2):
                    a,b = card
                    break
        card_a_rank = a
        card_a_suit = b
        return str(a+b)

    def set_community_cards(self, board, _round):

        i = 0
        while i < (len(board)):
            
            if(board[i] == -1):
                del board[i]
            else:
                i = i+1

        if not(all([card is -1 for card in board])):
            self.board = board 


    def take(self, num_take):
        import random
        cards_return_user = []
        for num in range(num_take):
            c = random.choice(self.deck_of_cards)
            while c in cards_return_user:
                c = random.choice(self.deck_of_cards)
            cards_return_user.append(c)
        return cards_return_user

    def random_board(self, hand, with_full_deck):
        deck = self.deck_of_cards
        b = self.take(3)
        while(self.is_duplicates(b, hand)):
            b = self.take(3)
        b = [Card.new(b[0]), Card.new(b[1]), Card.new(b[2])]
        return b

    def setup_random_board(self, hand = None):
        b = []
        if self.board is None: #PREFLOP
            b = self.random_board(hand, with_full_deck = False)
        
        return b 

    def shares_duplicate(self, cardA, cardB, check_this):
        if cardA in check_this or cardB in check_this:
            return True
        else:
            return False 

    def is_duplicates(self, board, hand):
        duplicate = False
        for card_b in board:
            for card_h in hand:
                if card_b == card_h:
                    duplicate = True

        return duplicate
    

    ## TODO: May need to modify handstrength to use 1036 * 2 in the case of having 2 opponents
    def handStrength(self, event):
        ahead, tied, behind = 0, 0, 0
        a, b, random_board, ourRank, oppRank = None, None, None, None, None
        count_none_debug = 0
        # Consider all two card combinations of remaining cards
        for potential_opp_cards in (self._combinations*(Player.total_plrs-1)):
            a, b = Card.new(potential_opp_cards[0]), Card.new(potential_opp_cards[1])
            if self.shares_duplicate(a, b, self.hand):
                continue
            if event is "Preflop":
                oppRank = self.do_mean_evaluation([a,b], event, n=self.preflop_opprank_control)
            else:
                need_skip = False
                while need_skip is False:
                    if (self.shares_duplicate(a, b, self.board)):
                        need_skip = True
                    break
                if need_skip:
                    continue
                oppRank = self.evaluator.evaluate(self.board, [a,b])
                    
            if(oppRank is None):
                continue
                count_none_debug+=1
            elif(self.evaluation < oppRank): # Note: With treys evaluation, lower number means better hand
                ahead = ahead + 1 
            elif self.evaluation == oppRank:
                tied = tied + 1
            else:
                behind = behind + 1
        hand_strength = (ahead+tied/2) / (ahead+tied+behind)
        return hand_strength

    
        


    def set_evaluation(self, value):
        self.evaluation = value

    def evaluate(self, event):
        if event == 'Preflop':
            self.set_evaluation(self.do_mean_evaluation(self.hand, event, n=self.preflop_evaluation_mean_control))
            self.hand_strength = ((1 - self.evaluation/7462)*2) if ((1 - self.evaluation/7462)*2) < 1.0 else 1.0
            # self.hand_strength = self.handStrength(event)
            # self.detect_draws()

        else:
            # self.detect_draws()
            self.set_evaluation(self.evaluator.evaluate(self.board, self.hand))
            self.hand_strength = self.handStrength(event) # UPDATE 12/03: Only using handStrength for post-flop for the moment
        self.rc = self.rank_class(self.evaluation)
        self.score_desc = self.evaluator.class_to_string(self.rc)
        self.summary = self.hand_strength, self.evaluation, self.rc, self.score_desc, self.hand, self.board
        return self.summary
    
    def ew_parse(self, card_list, is_num=True):
        list_trey_to_st = []
        if(is_num):
            for card in card_list:
                list_trey_to_st.append(Card.int_to_str(card))
        list_st_to_ppe = []
        for card_st in list_trey_to_st:
            list_st_to_ppe.append(card_st[1].upper()+card_st[0])

        return list_st_to_ppe

    def do_mean_evaluation(self, hand, event, n):
        fictional_board = None
        evaluation = None
        total_sum_evals = 0
        list_evaluations = []
        for i in range(n):
            if event is "Preflop":
                fictional_board = self.setup_random_board(hand) # fictional board used to evaluate 5-card set in treys evaluation function. hand is passed in to avoid duplicates in creating board
                while self.shares_duplicate(hand[0],hand[1], fictional_board):
                    fictional_board = self.setup_random_board(hand)
                evaluation = self.evaluator.evaluate(fictional_board, hand)
                del fictional_board
            else:
                evaluation = self.evaluator.evaluate(self.board, hand)
            list_evaluations.append(evaluation)
            total_sum_evals = total_sum_evals + evaluation
            del evaluation
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

    def rank_class(self, evaluation):
        rc = self.evaluator.get_rank_class(evaluation)
        return rc

    def set_hand(self, hand):
        self.hand = hand

    def get_evaluation(self, event):
        return self.summary

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
                if card_str != self.hand[0] and card_str != self.hand[1]:
                    li.append(card_str)
        
        self.deck_of_cards = li