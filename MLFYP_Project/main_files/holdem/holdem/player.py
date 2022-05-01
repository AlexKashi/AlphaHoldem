from random import randint
from gym import error
from treys import Card
from include import *

class Player(object):

  CHECK = 0
  CALL = 1
  RAISE = 2
  FOLD = 3

  total_plrs = 0
  def __init__(self, player_id, stack=2000, emptyplayer=False):
    
    self.player_id = player_id
    self.hand = []
    self.stack = stack
    self.currentbet = 0
    self.lastsidepot = 0
    self._seat = -1
    self.handrank = -1
    # flags for table management
    self.emptyplayer = emptyplayer
    self.betting = False
    self.isallin = False
    self.playing_hand = False
    self.playedthisround = False
    self.sitting_out = True
    self.evaluation_preflop = {'hand_strength': '', 'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
    self.evaluation_flop = {'hand_strength': '', 'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
    self.evaluation_turn = {'hand_strength': '', 'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
    self.evaluation_river = {'hand_strength': '', 'he': '', 'evaluation': 0, 'rc': '', 'score_desc': '', 'player_action': ''}
    self.he = None
    self.round = {'moves_i_made_in_this_round_sofar': '', 'possible_moves': set([]), 'raises_owed_to_me': 0, "raises_i_owe": 0}
    self.possible_moves = []
    self.position = player_id 
    self.debug_raises = {}
    self.reward = None
    self.action_type = None
    self.regret = {}
    self.raise_possible_tba = False
    self.certainty_to_call = 0
    self.round_track_stack = stack
    self.stack_start_game = stack
   

  def get_seat(self):
    return self._seat

  def is_possible(self, move):
    move_possible = False
    for item in self.round['possible_moves']:
      if item == move:
        return True
    return move_possible 

  def count_r(self, my_string, spec=None):
    count_r = 0

    if spec is None:

      for letter in my_string:
        if letter == 'R' or letter == 'r':
          count_r = count_r + 1

    
    else:

      for letter in my_string:
        if letter == 'c':
          count_r = count_r + 1

    return count_r


  def set_handrank(self, value):
    self.handrank = value

  def populatePlayerPossibleMoves(self, env):
    possible_moves = []
    if(self.count_r(env.last_seq_move) == 3):
      self.round['possible_moves'].clear()
      self.round['possible_moves'].add('c')
      self.round['possible_moves'].add('f')
      
    else:
      self.round['possible_moves'].clear()
      self.round['possible_moves'].add('r')
      self.round['possible_moves'].add('c')
      self.round['possible_moves'].add('f')

  def choose_action(self,  _round, range_structure, env):
    self.debug_raises.update({_round:env.level_raises})
    if self.round['raises_i_owe'] == 3:
      raise("error")
    betting_threshold = range_structure['betting'][self.round['raises_i_owe']][self.position]
    calling_threshold = range_structure['calling'][self.round['raises_i_owe']][self.position]
    action = None
    using_handstrength = False

    if range_structure == preflop_range:
      eval_cards = self.evaluation_preflop["evaluation"]
      
    else:
      eval_cards = self.he.hand_strength
      using_handstrength = True

    decide_boundaries = self.compare_eval_threshold(eval_cards, [betting_threshold, calling_threshold])

    self.raise_possible_tba = self.is_possible('r')
    # Now, for distributing rewards later we need to know how the probability that our villain will stay in the hand given that he faces another raise:
    # The reason for checking that here is that, with the artifical agent, we get certainties about his actions.
    check_next = self.round['raises_i_owe'] + 1
    if check_next < 3:
        
      potential_calling_threshold = range_structure['calling'][check_next][self.position] # Tells you how strong villains next hand must be
    
      if using_handstrength:
        self.certainty_to_call = 1 if eval_cards > potential_calling_threshold else 0            
      else:
        
        self.certainty_to_call = 1 if eval_cards < potential_calling_threshold else 0       

    if (decide_boundaries == betting_threshold) and self.is_possible('r'):
      # total_bet = env._tocall + env._bigblind - self.currentbet if _round == 'Preflop' else 25
      total_bet = None
      if _round == 'Preflop' and self.position == 0:
      
        total_bet = 40
      else:
        total_bet = 25

      action = (2, total_bet)
      assert action[1] == 40 or action[1] == 25
    elif (decide_boundaries == calling_threshold or decide_boundaries == betting_threshold) and self.is_possible('c'):
      action = [(1, 0), (0, 0)] # or 0
    else:
      action = (3, 0)

    return action

  def set_seat(self, value):
    self._seat = value

  
  def compare_eval_threshold(self, a, list_ev):
    ans = -1
    for b in list_ev:
      st = (a>b)-(a<b)
      if(type(a) is float and a <= 1.0): # HandStrength (Post-Flop)
        if st >= 1:
          return b
        else:
          continue
      else:                               # Standard Evaluation (Pre-Flop)
        if st >= 1:
          continue
        else:
          return b

    return -1

  def reset_hand(self):
    self._hand = []
    self.playedthisround = False
    self.betting = False
    self.isallin = False
    self.currentbet = 0
    self.lastsidepot = 0
    self.playing_hand = (self.stack != 0)

  def bet(self, bet_size):
    self.playedthisround = True
    if not bet_size:
      return
    self.stack -= (bet_size - self.currentbet) # Total - 10 in case of SB calling to see flop
    self.currentbet = bet_size
    if self.stack == 0:
      self.isallin = True

  def refund(self, ammount):
    self.stack += ammount

  def player_state(self):
    return (self.get_seat(), self.stack, self.playing_hand, self.betting, self.player_id)

  def reset_stack(self):
    self.stack = 2000

  def update_localstate(self, table_state):
    self.stack = table_state.get('stack')
    self.hand = table_state.get('pocket_cards')

  # cleanup
  def player_move(self, table_state, action, last_seq_move=None, _round=None):
    self.update_localstate(table_state)
    bigblind = table_state.get('bigblind')
    tocall = min(table_state.get('tocall', 0), self.stack)
    minraise = 0 #table_state.get('minraise', 0) - 10
    [action_idx, raise_amount] = action
    raise_amount = int(raise_amount) 
    action_idx = int(action_idx)
    if action[0] == 2:
      if _round == 0:
        if self.position == 0 and self.count_r(last_seq_move) == 0:
          action[1] = 40
        elif self.position == 2:
          action[1] = 50 if self.count_r(last_seq_move) > 0 else 25
        if self.get_seat() == 2 and self.count_r(last_seq_move) > 0:
          action[1] = 50 if self.count_r(last_seq_move) > 1 else 25
      else:
      
        action[1] = 50 if self.count_r(last_seq_move) > 0 else 25
    
    if (self.count_r(last_seq_move)) > 1 or _round != 0:
      
        to_call = 25 * (self.count_r(last_seq_move)) 

    [action_idx, raise_amount] = action
    raise_amount = int(raise_amount) 
    action_idx = int(action_idx)

    if tocall == 0:
      # if not(action_idx in [Player.CHECK, Player.RAISE]):
      #   print("watch")
      assert action_idx in [Player.CHECK, Player.RAISE]
      if action_idx == Player.RAISE:
        if raise_amount < minraise:
          raise error.Error('raise must be greater than minraise {}'.format(minraise))
        if raise_amount > self.stack:
          raise_amount = self.stack
        move_tuple = ('raise', raise_amount)
      elif action_idx == Player.CHECK:
        move_tuple = ('check', 0)
      else:
        raise error.Error('invalid action ({}) must be check (0) or raise (2)'.format(action_idx))
    else:
      if action_idx not in [Player.RAISE, Player.CALL, Player.FOLD]:
        raise error.Error('invalid action ({}) must be raise (2), call (1), or fold (3)'.format(action_idx))
      if action_idx == Player.RAISE:
        if raise_amount < minraise:
          raise error.Error('raise must be greater than minraise {}'.format(minraise))
        if raise_amount > self.stack:
          raise_amount = self.stack
        move_tuple = ('raise', raise_amount)
      elif action_idx == Player.CALL:
        move_tuple = ('call', tocall)
      elif action_idx == Player.FOLD:
        move_tuple = ('fold', -1)
      else:
        raise error.Error('invalid action ({}) must be raise (2), call (1), or fold (3)'.format(action_idx))
    return move_tuple
