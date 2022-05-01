
from gym import Env, error, spaces, utils
from gym.utils import seeding
import HandHoldem
from treys import Card, Deck, Evaluator

from .player import Player
from .utils import hand_to_str, format_action
from collections import OrderedDict
from statistics import mean
import time

class TexasHoldemEnv(Env, utils.EzPickle):
	BLIND_INCREMENTS = [[10,25], [25,50], [50,100], [75,150], [100,200],
						[150,300], [200,400], [300,600], [400,800], [500,10000],
						[600,1200], [800,1600], [1000,2000]]
	
	current_player_notifier = ""
	weighting_coefficient_regret_fold = 10
	weighting_coefficient_regret_check = 10
	weighting_coefficient_regret_call = 10
	weighting_coefficient_regret_raise = 10
	weighting_coefficient_round_resolve = 100

	

	def __init__(self, n_seats, max_limit=100000, debug=False):
		n_suits = 4                     # s,h,d,c
		n_ranks = 13                    # 2,3,4,5,6,7,8,9,T,J,Q,K,A
		n_community_cards = 5           # flop, turn, river
		n_pocket_cards = 2
		n_stud = 5

		self.level_raises = {0:0, 1:0, 2:0} # Assuming 3 players
		
		self.n_seats = n_seats
		self._blind_index = 0
		[self._smallblind, self._bigblind] = TexasHoldemEnv.BLIND_INCREMENTS[0]
		self._deck = Deck()
		self._evaluator = Evaluator()
		self.last_seq_move = [] 
		self.filled_seats = 0
		self.signal_end_round = False
		self.winning_players = None
		self.starting_stack_size = None
		self.community = []
		self._round = 0
		self._button = 0
		self._discard = []
		self.game_resolved = False
		self.is_new_r = True
		self._side_pots = [0] * n_seats
		self._current_sidepot = 0 # index of _side_pots
		self._totalpot = 0
		self._tocall = 0
		self._lastraise = 0
		self._number_of_hands = 0
		self._record_players = []

		# fill seats with dummy players
		self._seats = [Player(i, stack=0, emptyplayer=True) for i in range(n_seats)]
		self.learner_bot = None
		self.villain = None
		self.emptyseats = n_seats
		self._player_dict = {}
		self._current_player = None
		self._debug = debug
		self._last_player = None
		self._last_actions = None



		# (PSEUDOCODE)
        # MODEL HYPERPARAMETERS: 
        # state_size = [(position, learner.stack, learner.handrank, played_this_round ...[card1, card2]), (pot_total, learner.to_call, opponent.stack, community_cards)]
        # action_size = env.action_space.n
        # learning_rate = 0.00025

		self.observation_space = spaces.Tuple([

			spaces.Tuple([                # players
				spaces.MultiDiscrete([
				max_limit,           # stack
				max_limit,           # handrank
				1,                   # playedthisround
				1,                   # is_betting
				max_limit,           # last side pot
				]),
				spaces.Tuple([
					spaces.MultiDiscrete([    # card
						n_suits,          # suit, can be negative one if it's not avaiable.
						n_ranks,          # rank, can be negative one if it's not avaiable.
					])
				] * n_pocket_cards)
			] * 4),

			spaces.Tuple([
				spaces.Discrete(max_limit),   # learner position
				spaces.Discrete(max_limit),   # pot amount
				spaces.Discrete(max_limit),   # last raise
				spaces.Discrete(n_seats - 1), # current player seat location.
				spaces.Discrete(max_limit),   # minimum amount to raise
				spaces.Discrete(max_limit), # how much needed to call by current player.
				spaces.Tuple([
					spaces.MultiDiscrete([    # card
						n_suits - 1,          # suit
						n_ranks - 1,          # rank
						1,                     # is_flopped
					])
				] * n_community_cards)
			])
		])

		### MAY NEED TO ALTER FOR HEADS-UP
		# self.action_space = spaces.Tuple([
		# spaces.MultiDiscrete([
		# 	3,                     # action_id
		# 	max_limit,             # raise_amount
		# ]),
		# ] * n_seats) 
		self.action_space = spaces.Discrete(3)
		

	def seed(self, seed=None):
		_, seed = seeding.np_random(seed)
		return [seed]


	# Important Note: Positions are only assigned at end of game. Be aware in 
	# case of reporting stats on position type
	def assign_positions(self):
		if (self.filled_seats > 2):
			count_players = len(self._player_dict)
			for player in self._player_dict.values():
				if not(player.emptyplayer):
					player.position = 0 if (player.position == count_players - 1) else player.position + 1

		elif (self.filled_seats == 2):
			new_positions = []
			# We want to only use positions 0 and 2, which are encodings of BTN and BB respectively

			# Sort for positions 0 and 2 first
			for player in self._player_dict.values():
				if not (player.emptyplayer):
					if player.position == 2:
						player.position = 0
						new_positions.append(player.position)
					elif player.position == 0:
						player.position = 2
						new_positions.append(player.position)

			# Special case of former position 1 depends on new positions allocated above
			if len(new_positions) == 1:
				for player in self._player_dict.values():
					if player.position == 1:
						if new_positions[0] == 0:
							player.position = 2
						elif new_positions[0] == 2:
							player.position = 0

	def add_player(self, seat_id, stack=2000):
		"""Add a player to the environment seat with the given stack (chipcount)"""
		player_id = seat_id
		if player_id not in self._player_dict:
			new_player = Player(player_id, stack=stack, emptyplayer=False)
			Player.total_plrs+=1
			self.starting_stack_size = stack
			if self._seats[player_id].emptyplayer:
				self._seats[player_id] = new_player
				new_player.set_seat(player_id)
			else:
				raise error.Error('Seat already taken.')
			self._player_dict[player_id] = new_player
			self.emptyseats -= 1
			self.filled_seats +=1
		if new_player.get_seat() == 0:
			self.learner_bot = new_player
		else:
			self.villain = new_player
		self._record_players.append(new_player)
			
			
			

	def move_player_to_empty_seat(self, player):
		# priority queue placing active players at front of table
		for seat_no in range(len(self._seats)):
			if self._seats[seat_no].emptyplayer and (seat_no < player._seat):
				unused_player = self._seats[seat_no]
				self._seats[seat_no] = player
				self._seats[player.get_seat()] = unused_player

	def reassign_players_seats(self):
		for player in self._player_dict.values():
			self.move_player_to_empty_seat(player)

	def remove_player(self, seat_id):
		"""Remove a player from the environment seat."""
		player_id = seat_id
		
		try:
			idx = self._seats.index(self._player_dict[player_id])
			self._seats[idx] = Player(0, stack=0, emptyplayer=True)
			
			self._seats[idx].position = None # Very important for when transitioning from 3 to 2 players.
			del self._player_dict[player_id]
			self.emptyseats += 1
			self.filled_seats-=1
			Player.total_plrs-=1

			#self.reassign_players_seats()
		except ValueError:
			pass

	def reset(self):
		self._reset_game()
		self._ready_players()
		self._number_of_hands = 1
		[self._smallblind, self._bigblind] = TexasHoldemEnv.BLIND_INCREMENTS[0]
		if (self.emptyseats < len(self._seats) - 1):
			players = [p for p in self._seats if p.playing_hand]
			self._new_round()
			self._round = 0
			self._current_player = self._first_to_act(players, "post_blinds")
			self._post_smallblind(self._current_player)
			self._current_player = self._next(players, self._current_player)
			self._post_bigblind(self._current_player)
			self._current_player = self._next(players, self._current_player)
			self._tocall = self._bigblind
			self._round = 0
			self._deal_next_round()
			self.organise_evaluations()
			
			self._folded_players = []
		return self._get_current_reset_returns()


	def organise_evaluations(self):
		for idx, player in self._player_dict.items():
			if player is not None:
				player.he = HandHoldem.HandEvaluation(player.hand, idx, "Preflop") #Unique to player instance
				player.he.evaluate(event='Preflop')
				player.set_handrank(player.he.evaluation)
		

	def assume_unique_cards(self, players):
		cards_count = {}
		this_board = None
		for player in players:
			player_cards = player.hand
			for card in player_cards:
				cards_count.update({card: 1}) if card not in cards_count else cards_count.update({card: cards_count[card] + 1})
			if this_board is None and player.he is not None:
				if player.he.board is not None:
					this_board = player.he.board 
		if this_board is not None:
			for card in this_board:
				cards_count.update({card: 1}) if card not in cards_count else cards_count.update({card: cards_count[card] + 1})
		
		for card, no_occurence in cards_count.items():
			if no_occurence > 1:
				return False
			else:
				return True

	def step(self, actions):
		"""
		CHECK = 0
		CALL = 1
		RAISE = 2
		FO

		RAISE_AMT = [0, minraise]
		"""
		
		players = [p for p in self._seats if p.playing_hand]
		assert self.assume_unique_cards(players) is True

		self._last_player = self._current_player
		# self._last_actions = actions
		

		# if self._last_player.count_r(self.last_seq_move) > 1:
		# 	if [3,0] in actions:
		# 		print("r")	

		# if current player did not play this round 
		if not self._current_player.playedthisround and len([p for p in players if not p.isallin]) >= 1:
			if self._current_player.isallin:
				self._current_player = self._next(players, self._current_player)
				return self._get_current_step_returns(False)

			move = self._current_player.player_move(self._output_state(self._current_player), actions[self._current_player.player_id], last_seq_move = self.last_seq_move, _round = self._round)
			if self.am_i_only_player_wmoney() and self.level_raises[self._current_player.get_seat()] >= self.highest_in_LR()[0]:
				move = ("check", 0) # Protects against player making bets without any other stacked/active players
			self._last_actions = move
			if move[0] == 'call':
				assert self.action_space.contains(0)
				self._player_bet(self._current_player, self._tocall, is_posting_blind=False, bet_type=move[0])
				if self._debug:
					print('Player', self._current_player.player_id, move)
				self._current_player = self._next(players, self._current_player)
				self.last_seq_move.append('C')
				self.playedthisround = True
				self._current_player.round['raises_i_owe'] = 0

			elif move[0] == 'check':
				# assert self.action_space.contains(0)
				self._player_bet(self._current_player, self._current_player.currentbet, is_posting_blind=False, bet_type=move[0])
				if self._debug:
					print('Player', self._current_player.player_id, move)
				self._current_player = self._next(players, self._current_player)
				self.last_seq_move.append('c')
				self.playedthisround = True

			elif move[0] == 'raise':
				# if self._current_player is self.learner_bot and self.level_raises == {0: 1, 1: 0, 2: 2} or self.level_raises == {0: 2, 1: 0, 2: 3} or self.level_raises == {0: 3, 1: 0, 2: 4} or self.level_raises == {0: 4, 1: 0, 2: 5} or self.level_raises == {0: 5, 1: 0, 2: 6} or self.level_raises == {0: 5, 1: 0, 2: 6} and 'R' in self.last_seq_move:
				# 	print("watch")
				assert self.action_space.contains(1)
				
				self._player_bet(self._current_player, move[1]+self._current_player.currentbet, is_posting_blind=False, bet_type="bet/raise")
				if self._debug:
					print('Player', self._current_player.player_id, move)
				for p in players:
					if p != self._current_player:
						p.playedthisround = False
				self._current_player = self._next(players, self._current_player)
				
				self.last_seq_move.append('R')
				self._current_player.round['raises_i_owe'] = 0
				
			elif move[0] == 'fold':
				# if self.highest_in_LR()[0] > 4:
				# 	print("watch")
				assert self.action_space.contains(2)
				self._current_player.playing_hand = False
				self._current_player.playedthisround = True
				if self._debug:
					print('Player', self._current_player.player_id, move)
				self._current_player = self._next(players, self._current_player)
				
				self._folded_players.append(self._current_player)
				self.last_seq_move.append('F')
				# break if a single player left
				# players = [p for p in self._seats if p.playing_hand]
				# if len(players) == 1:
				# 	self._resolve(players)

		players = [p for p in self._seats if p.playing_hand]

		# else:	## This will help eliminate infinite loop
		# 	self._current_player = self._next(players, self._current_player)
			
		# This will effectively dictate who will become dealer after flop	
		players_with_money = []
		for player in players:
			if(player.stack > 0):
				players_with_money.append(player)
		if all([player.playedthisround for player in players_with_money]):
			self._resolve(players)
			for player in self._player_dict.values():
				player.round == {'moves_i_made_in_this_round_sofar': '', 'possible_moves': set([]), 'raises_owed_to_me': 0, "raises_i_owe": 0}
		

		terminal = False
		if all([player.isallin for player in players]):
			while self._round < 4:
				self._deal_next_round()
				self._round += 1

		elif self.count_active_wmoney() == 1 and all([player.playedthisround for player in players]):
			# do something else here
			while self._round < 3:
				self._round += 1
				self._deal_next_round()
			

		if self._round == 4 or len(players) == 1:
			terminal = True
			self._resolve(players)
			self._resolve_round(players)


		return self._get_current_step_returns(terminal, action=move)
		

	def am_i_only_player_wmoney(self):
		count_other_broke = 0
		for player in self._player_dict.values():
			if player is not self._current_player and player.stack <= 0:
				count_other_broke += 1
		if count_other_broke == (len(self._player_dict) - 1):
			return True
		else:
			return False

	def count_active_wmoney(self):
		count = 0
		account_active_money = {0:{"is_active":False, "has_money":False},1:{"is_active":False, "has_money":False},2:{"is_active":False, "has_money":False}}
		for player in self._player_dict.values():
			if player.playing_hand:
				account_active_money[player.get_seat()].update({"is_active": True})
			if player.stack > 0:
				account_active_money[player.get_seat()].update({"has_money": True})
			
		for player, account in account_active_money.items():
			if account["is_active"] is True and account["has_money"] is True:
				count+=1

		return count



	def render(self, mode='human', close=False, initial=False, delay=None):
		if delay:
			time.sleep(delay)

		if(initial is True):
			print("\n")
				
		if self._last_actions is not None and initial is False:
			pid = self._last_player.player_id
			#print('last action by player {}:'.format(pid))
			print(format_action(self._last_player, self._last_actions))

		print("\n\n")
		print('Total Pot: {}'.format(self._totalpot))
		(player_states, community_states) = self._get_current_state()
		(player_infos, player_hands) = zip(*player_states)
		(community_infos, community_cards) = community_states

		print('Board:')
		print('-' + hand_to_str(community_cards))
		print('Players:')
		# for player in self._player_dict:
		# 	assert player.round['raises_i_owe']
		for idx, hand in enumerate(player_hands):
			if self._current_player.get_seat() == idx:
				self.current_player_notifier = "<" + str(self._current_player.position)
				
			print('{}{}stack: {} {}'.format(idx, hand_to_str(hand), self._seats[idx].stack, self.current_player_notifier))
			self.current_player_notifier = ""

	def _resolve(self, players):
		
		self.signal_end_round = True
		self._current_player = self._first_to_act(players)
		self._resolve_sidepots(players + self._folded_players)
		self._new_round()
		self._deal_next_round()
		if self._debug:
			print('totalpot', self._totalpot)

	def _resolve_postflop(self, players):
		self._current_player = self._first_to_act(players)
		# print(self._current_player)

	def _deal_next_round(self):
		if self._round == 0:
			self._deal()
		elif self._round == 1:
			self._flop()
		elif self._round == 2:
			self._turn()
		elif self._round == 3:
			self._river()

	def _increment_blinds(self):
		self._blind_index = min(self._blind_index + 1, len(TexasHoldemEnv.BLIND_INCREMENTS) - 1)
		[self._smallblind, self._bigblind] = TexasHoldemEnv.BLIND_INCREMENTS[self._blind_index]

	def _post_smallblind(self, player):
		if self._debug:
			print('player ', player.player_id, 'small blind', self._smallblind)
		self._player_bet(player, self._smallblind, is_posting_blind=True)
		player.playedthisround = False

	def _post_bigblind(self, player):
		if self._debug:
			print('player ', player.player_id, 'big blind', self._bigblind)
		self._player_bet(player, self._bigblind, is_posting_blind=True)
		player.playedthisround = False
		self._lastraise = self._bigblind

	def highest_in_LR(self, specific=None, request_is_seq=None):
		highest_lr_bot = 0
		highest_lr_value = 0
		if specific is None:
			spec = self.level_raises
		else:
			spec = specific
		for key, value in spec.items():
			if value > highest_lr_value:
				highest_lr_value = value
				highest_lr_bot = key
		rep = [(highest_lr_value, highest_lr_bot)]
		if request_is_seq:
			for key, value in spec.items():
				if value == highest_lr_value and key != highest_lr_bot:
					rep.append((value, key))
			return rep
		else:
			return highest_lr_value, highest_lr_bot

	def is_level_raises_allzero(self):
		count_zero = 0
		for value in self.level_raises.values():
			if value == 0:
				count_zero+=1
		if(count_zero == len(self.level_raises)):
			return True
		else: 
			return False

	def _player_bet(self, player, total_bet, **special_betting_type):
		# Case 1: New round, players have incosistent raises
		# Case 2: End of round, difference of raises is 2
		import operator
		sorted_lr = sorted(self.level_raises.items(), key=operator.itemgetter(1))
		
		# if (self.is_off_balance_LR() and self.is_new_r) or ( ((int(self.highest_in_LR()[0]) - int(sorted_lr[1][1])) == 2) and (self.is_new_r is False)):
		# 	print("raise")

		if "is_posting_blind" in special_betting_type and "bet_type" not in special_betting_type: # posting blind (not remainder to match preceding calls/raises)
			if special_betting_type["is_posting_blind"] is True:
				self.level_raises[player.get_seat()] = 0 

		elif "is_posting_blind" in special_betting_type and "bet_type" in special_betting_type: # Bet/Raise or call. Also accounts for checks preflop.
			highest_lr_value, highest_lr_bot = self.highest_in_LR()
			if special_betting_type["is_posting_blind"] is False:
				if special_betting_type["bet_type"] == "bet/raise":
					if self.level_raises[player.get_seat()] < highest_lr_value:
						player.action_type = "raise"
						self.level_raises[player.get_seat()] = highest_lr_value + 1
					elif self.level_raises[player.get_seat()] == highest_lr_value:
						player.action_type = "bet"
						self.level_raises[player.get_seat()] += 1

				elif special_betting_type["bet_type"] == "call":
					if self.level_raises[player.get_seat()] < highest_lr_value:
						player.action_type = "call"
						self.level_raises[player.get_seat()] = highest_lr_value

					elif self.is_level_raises_allzero():
						if player.position == 0:
							player.action_type = "call"
							self.level_raises[player.get_seat()] = 1


					elif player.position == 2:
						player.action_type = "call"
						self.level_raises[player.get_seat()] = highest_lr_value

				elif special_betting_type["bet_type"] == "check" and self._round is 0:	# BB checking preflop
					if player.position == 2:
						self.level_raises[player.get_seat()] = 1
					
		
		# relative_bet is how much _additional_ money is the player betting this turn,
		# on top of what they have already contributed
		# total_bet is the total contribution by player to pot in this round
		relative_bet = min(player.stack, total_bet - player.currentbet)
		player.bet(relative_bet + player.currentbet)

		self._totalpot += relative_bet
		self._tocall = max(self._tocall, total_bet)
		if self._tocall > 0:
			self._tocall = max(self._tocall, self._bigblind)
		self._lastraise = max(self._lastraise, relative_bet  - self._lastraise)
		self.is_new_r = False

	def _first_to_act(self, players, my_event="Postflop"):
		# if self._round == 0 and len(players) == 2:
		# 	return self._next(sorted(
		# 		players + [self._seats[self._button]], key=lambda x:x.get_seat()),
		# 		self._seats[self._button])
		
		first_to_act = None

		if self.filled_seats == 2:
			if my_event is "Preflop" or my_event is "post_blinds":
				first_to_act = self.assign_next_to_act(players, [0,2])

			elif my_event is "Postflop" or my_event is "sidepot":
				first_to_act = self.assign_next_to_act(players, [2,0])

		elif self.filled_seats == 3:
			if my_event is "Preflop":
				first_to_act = self.assign_next_to_act(players, [0,1,2])

			elif my_event is "Postflop" or my_event is "post_blinds" or my_event is "sidepot":
				first_to_act = self.assign_next_to_act(players, [1,2,0])

		# else: 
		# 	my_return = [player for player in players if player.get_seat() > self._button][0]
			
		#assert first_to_act is not None and not(first_to_act.emptyplayer) and not(first_to_act.stack <= 0)

		if len(players) == 1:
			first_to_act = self._record_players[0]

		return first_to_act

	def assign_next_to_act(self, players, precedence_positions):
		for pos in precedence_positions:
			for player in players:
				if player.position == pos and not(player.emptyplayer) and player.playing_hand and player.stack > 0:
					assert player is not None
					return player

	def _next(self, players, current_player):
		i = 1
		current_player_seat = players.index(current_player)
		
		while(players[(current_player_seat+i) % len(players)].stack <= 0):
			i+=1
			if i > 10: 
				break
				# In this case of inifinte loop, self._current_player is assigned to _next but will be irrelevant anyway so okay.
		assert players[(current_player_seat+i) % len(players)] is not None
		return players[(current_player_seat+i) % len(players)]

	def _deal(self):
		for player in self._seats:
			if player.playing_hand and player.stack > 0:
				player.hand = self._deck.draw(2)
				
				

	def _flop(self):
		self._discard.append(self._deck.draw(1)) #burn
		this_flop = self._deck.draw(3)
		self.flop_cards = this_flop
		self.community = this_flop

	def _turn(self):
		self._discard.append(self._deck.draw(1)) #burn
		self.turn_card = self._deck.draw(1)
		self.community.append(self.turn_card)
		# .append(self.community)

	def _river(self):
		self._discard.append(self._deck.draw(1)) #burn
		self.river_card = self._deck.draw(1)
		self.community.append(self.river_card)

	def _ready_players(self):
		for p in self._seats:
			if not p.emptyplayer and p.sitting_out:
				p.sitting_out = False
				p.playing_hand = True
		
		

	def _resolve_sidepots(self, players_playing):
		players = [p for p in players_playing if p.currentbet]
		if self._debug:
			print('current bets: ', [p.currentbet for p in players])
			print('playing hand: ', [p.playing_hand for p in players])
		if not players:
			return
		try:
			smallest_bet = min([p.currentbet for p in players if p.playing_hand])
		except ValueError:
			for p in players:
				self._side_pots[self._current_sidepot] += p.currentbet
				p.currentbet = 0
			return

		smallest_players_allin = [p for p, bet in zip(players, [p.currentbet for p in players]) if bet == smallest_bet and p.isallin]

		for p in players:
			self._side_pots[self._current_sidepot] += min(smallest_bet, p.currentbet)
			p.currentbet -= min(smallest_bet, p.currentbet)
			p.lastsidepot = self._current_sidepot

		if smallest_players_allin:
			self._current_sidepot += 1
			self._resolve_sidepots(players)
		if self._debug:
			print('sidepots: ', self._side_pots)

	def _new_round(self):
		for player in self._player_dict.values():
			player.currentbet = 0
			player.playedthisround = False
			player.round = {'moves_i_made_in_this_round_sofar': '', 'possible_moves': set([]), 'raises_owed_to_me': 0, "raises_i_owe": 0}
			player.round_track_stack =  player.stack

		self.is_new_r = True
		self._round += 1
		self._tocall = 0
		self._lastraise = 0
		self.last_seq_move = []
		# if self.is_off_balance_LR():
		# 	if self._last_actions[0] != 'fold':
		# 		raise error.Error()
		
	def is_off_balance_LR(self):
		
		lr = self.level_raises
		highest_value, highest_bot  = self.highest_in_LR()
		lr_without_highest = dict(lr)
		del lr_without_highest[highest_bot]
		next_highest_value, next_highest_bot = self.highest_in_LR(specific=lr_without_highest)
		
		if highest_value != next_highest_value:
			return True
		elif highest_value == next_highest_value:
			return False

		
	def _resolve_round(self, players):
		# if len(players) == 1:
		# 	if (self._round == 1 or self._round == 2) and self._last_player.get_seat() == 0 and self._last_actions[0] == 'fold':
		# 		if self._last_player.count_r(self.last_seq_move) < 1:
		# 			if self.learner_bot.position == 0:
		# 				players[0].refund(self._bigblind + self._smallblind)
		# 				self._totalpot = 0
		# 				self.winning_players = players[0]
		# 			else:
		# 				players[0].refund(self._bigblind + self._smallblind + 40)
		# 				self._totalpot = 0
		# 				self.winning_players = players[0]
		# 	else:
		# 		players[0].refund(sum(self._side_pots))
		# 		self._totalpot = 0
		# 		self.winning_players = players[0]
		if len(players) == 1:
			winner = None # Heads-Up
			losers = []
			for p in self._record_players:
				if p == players[0]:
					winner = p
				else:
					losers.append(p)

			winner_investment = winner.stack_start_game - winner.stack
			sum_losers_lost = 0
			for loser in losers:
				sum_losers_lost += loser.stack_start_game - loser.stack

			players[0].refund(winner_investment + sum_losers_lost)

			self._totalpot = 0
			self.winning_players = players[0]

		else:
			# compute hand ranks
			for player in players:
				# assert (len(self.community) <= 5) is True
				player.handrank = self._evaluator.evaluate(player.hand, self.community)

			# trim side_pots to only include the non-empty side pots
			temp_pots = [pot for pot in self._side_pots if pot > 0]

			# compute who wins each side pot and pay winners
			for pot_idx,_ in enumerate(temp_pots):
				# find players involved in given side_pot, compute the winner(s)
				pot_contributors = [p for p in players if p.lastsidepot >= pot_idx]
				winning_rank = min([p.handrank for p in pot_contributors])
				winning_players = [p for p in pot_contributors if p.handrank == winning_rank]
				self.winning_players = winning_players[0]
				for player in winning_players:
					split_amount = int(self._side_pots[pot_idx]/len(winning_players))
					if self._debug:
						print('Player', player.player_id, 'wins side pot (', int(self._side_pots[pot_idx]/len(winning_players)), ')')
					player.refund(split_amount)
					self._side_pots[pot_idx] -= split_amount

				# any remaining chips after splitting go to the winner in the earliest position
				if self._side_pots[pot_idx]:
					earliest = self._first_to_act([player for player in winning_players], "sidepot")
					earliest.refund(self._side_pots[pot_idx])

			# for player in players: ## THIS IS AT THE END OF THE GAME. NOT DURING. (safe)
			# 	if(player.stack == 0):
			# 		self.remove_player(player.get_seat())
		self.game_resolved = True

		# assert(self._player_dict[0].stack + self._player_dict[2].stack + self._totalpot == 2*self.starting_stack_size)
		
	def report_game(self, requested_attributes, specific_player=None):
		if "stack" in requested_attributes:
			player_stacks = {}
			for key, player in self._player_dict.items():
				
				player_stacks.update({key: player.stack})
				
			# if len(player_stacks) < 3:
			# 	for i in range(3):
			# 		if i not in player_stacks:
			# 			player_stacks.update({i:0})
			if specific_player == None:
				return (player_stacks)
				assert (player_stacks.values()) != None
			else:
				return (player_dict[specific_player].values())
				 
			
		
		

		


		

	def _reset_game(self):
		
		playing = 0

		# if self._player_dict[0].stack is not None and self._player_dict[2].stack is not None:
		# 	assert(self._player_dict[0].stack + self._player_dict[2].stack == 2*self.starting_stack_size)

		
		for player in self._seats:
			if not player.emptyplayer and not player.sitting_out:
				player.stack_start_game = player.stack
				player.reset_hand()
				playing += 1
		self.community = []
		self._current_sidepot = 0
		self._totalpot = 0
		self._side_pots = [0] * len(self._seats)
		self._deck.shuffle()
		self.level_raises = {0:0, 1:0, 2:0}
		self.winning_players = None
		self.game_resolved = False


		if playing:
			self._button = (self._button + 1) % len(self._seats)
			while not self._seats[self._button].playing_hand:
				self._button = (self._button + 1) % len(self._seats)

	def _output_state(self, current_player):
		return {
		'players': [player.player_state() for player in self._seats],
		'community': self.community,
		'my_seat': current_player.get_seat(),
		'pocket_cards': current_player.hand,
		'pot': self._totalpot,
		'button': self._button,
		'tocall': (self._tocall - current_player.currentbet),
		'stack': current_player.stack,
		'bigblind': self._bigblind,
		'player_id': current_player.player_id,
		'lastraise': self._lastraise,
		'minraise': max(self._bigblind, self._lastraise + self._tocall),
		}

	def _pad(self, l, n, v):
		if (not l) or (l == None):
			l = []
		return l + [v] * (n - len(l))

	def _get_current_state(self):
		player_states = []
		for player in self._seats:
			player_features = [
				int(player.stack),
				int(player.handrank),
				int(player.playedthisround),
				int(player.betting),
				int(player.lastsidepot),
			]
			player_states.append((player_features, self._pad(player.hand, 2, -1)))
		community_states = ([
			int(self.learner_bot.position),
			int(self._totalpot),
			int(self._lastraise),
			int(self._current_player.get_seat()),
			int(max(self._bigblind, self._lastraise + self._tocall)),
			int(self._tocall - self._current_player.currentbet),
		], self._pad(self.community, 5, -1))
		# if sum(self.level_raises.values()) > 6:
		# 	print("")
		return (tuple(player_states), community_states)

	def _get_current_reset_returns(self):
		return self._get_current_state()

	def distribute_rewards_given_endgame(self):
	
		if self.learner_bot == self.winning_players:
			self.learner_bot.reward = self.compute_reward() + self._totalpot
		else:
			self.learner_bot.reward = self.learner_bot.round_track_stack


	def _get_current_step_returns(self, terminal, action=None):

		observations = self._get_current_state()
		stacks = [player.stack for player in self._seats]
		reward = None
			
		if(action == None):
			return observations, reward, terminal, [] # TODO, return some info?

		else: 	 # Focus on this. At end of step, when player has already decided his action. 
			respective_evaluations = [player.he.evaluation if player.he != None else None for player in self._seats]
			evaluations_opposing_players = [x for i,x in enumerate(respective_evaluations) if i!= self._last_player.get_seat() and x!=None]
			
			if (self._last_player == self.learner_bot): 					# Learner bot step return

				if(self.signal_end_round == True):
					self.signal_end_round = False
			
				self.learner_bot.reward = self.compute_reward()		# Most common entry point (Learner Checks or raises)

			else:  		
																	# Artifical agent step return
				self.learner_bot.reward = 0

				if(self.signal_end_round == True):
					if(action == ('fold', 0)): # Opponent folded
						self.learner_bot.reward = self._totalpot
					
			# if action is ('fold', 0) or action is ('check', 0) or action[0] is 'call' or action[0] is 'raise':
			# 	regret = self.compute_regret_given_action(action, respective_evaluations, evaluations_opposing_players)
			
			

			return observations, action, reward, terminal, [] # TODO, return some info?


	def compute_reward(self): #only gets called when last player is learner

		# Expected value is a mathematical concept used to judge whether calling a raise in a game of poker will be profitable.  
		# When an opponent raises a pot in poker, such as on the flop or river, your decision whether to call or fold is more or less 
		# completely dependant on expected value.  This is the calculation of whether the probability of winning a pot will make a call 
		# profitable in the long-term.
		# Expected Value is a monetary value (e.g. +$10.50). It can be positive or
		# negative. EV tells you how profitable or unprofitable a certain play (e.g.
		# calling or betting) will be. We work out EV when we are faced with a decision.

		# EV = (Size of Pot x Probability of Winning) – Cost of Entering it.

		equity = self.equity()
		ev = None
		if self._round == 0 and self._last_player.position == 0: # Only works for heads up: Due to bug with tocall
			to_call = 15
			total_pot = self._totalpot - to_call
		else:
			to_call = self._last_actions[1]
			total_pot = self._totalpot if self._last_player != self.learner_bot else (self._totalpot - self._last_actions[1])
			
				

		# Here we compute expected values for actions that were possible during their execution, and we reflect on them here by comparing the expected values
		# of alternatives.
		expected_values_order = [0, 0, 0] # In order of call/check, raise/bet, fold
		
		if self._last_actions[0] == 'call' or self._last_actions[0] == 'check':
			action_taken = 0
		elif self._last_actions[0] == 'raise' or self._last_actions[0] == 'bet':
			action_taken = 1
		else:
			action_taken = 2

		# Call/Check Regret
		learner_equity, opp_equity = equity[0], equity[1]
		stand_to_win = (total_pot * learner_equity) 
		stand_to_lose = to_call * opp_equity
		expected_value = stand_to_win - stand_to_lose
		expected_values_order[0] = expected_value

		# Fold Regret
		stand_to_win = to_call * opp_equity
		stand_to_lose = (total_pot) * learner_equity
		expected_value = stand_to_win - stand_to_lose
		expected_values_order[2] = expected_value

		# Raise/Bet Regret
		if (self.learner_bot.raise_possible_tba):
			# implied raise (How much more we stand to win given that villain shows confidence in his hand)
			stand_to_win = ( ((total_pot + 25) * learner_equity) * self.villain.certainty_to_call ) + (total_pot * learner_equity) * (1 - self.villain.certainty_to_call)
			stand_to_lose = (to_call + 25) * opp_equity
			expected_value = stand_to_win - stand_to_lose
			expected_values_order[1] = expected_value

	
		max_ev = max(expected_values_order)
		highest_paying_action = [i for i, j in enumerate(expected_values_order) if j == max_ev]
		
		# reward = expected_values_order[action_taken]/max_ev
		# how much does reward deviate from mean - this determines quality of action in the context of all possible actions
		reward = expected_values_order[action_taken] - mean(expected_values_order)
		return reward 

	def compute_reward_end_round_fold(self, respective_evaluations, evaluations_opposing_players):
		return (respective_evaluations[self._last_player.get_seat()] - mean([other_player_eval for other_player_eval in evaluations_opposing_players])) / self.weighting_coefficient_round_resolve

	def compute_regret_given_action(self, my_action, respective_evaluations, evaluations_opposing_players):
		
		self.compare_evaluations_players(my_action, respective_evaluations, evaluations_opposing_players)
		# Now player has his regret filled in to his own player instance
		pass



	


	def equity(self):

		# Equity is a percentage (e.g. 70%). Equity tells you how much of the pot 
		# “belongs” to you, or to put it another way, the percentage of the time
		#  you expect to win the hand on average from that point onwards.
		_round = self._round if self.signal_end_round != True else self._round - 1
		if (_round == 1 or _round == 2 or _round ==3): # Implies last rounds were either 1 or 2
			learner_utility, opp_utility = self.compute_winner_simulation(_round)
			equity = learner_utility, opp_utility
			
		else:
			learner_hs = self.learner_bot.he.hand_strength, 1 - self.villain.he.hand_strength
			bot_hs = self.villain.he.hand_strength, 1 - self.learner_bot.he.hand_strength
			equity = (learner_hs[0] + learner_hs[1])/2, (bot_hs[0] + bot_hs[1])/2
		return equity


	def compute_winner_simulation(self, _round):
		_evaluator = self._evaluator
		deck = self._deck
		if _round == 1:
			community = [self.community[i] for i in range(3)]
		elif _round == 2:
			community = [self.community[i] for i in range(4)]
		else:
			community = [self.community[i] for i in range(5)]
		opp1_cards = self.learner_bot.hand
		opp2_cards = self.villain.hand
		unrevealed_cards = sorted([card for card in deck.cards if card not in community and card not in opp1_cards and card not in opp2_cards])
		# print(Card.print_pretty_cards(opp1_cards))
		# print(Card.print_pretty_cards(opp2_cards))
		winning_players_list = []
		learner_wins = 0
		opp_wins = 0
		if _round == 1:
			for turn_card_idx in range(len(unrevealed_cards)):
				# print(turn_card_idx)
				for river_card_idx in range(turn_card_idx, len(unrevealed_cards)):
					if [unrevealed_cards[turn_card_idx]] == [unrevealed_cards[river_card_idx]]:
						continue
					# print(Card.print_pretty_cards(community + [unrevealed_cards[turn_card_idx]] + [unrevealed_cards[river_card_idx]]))
					learner_eval = (_evaluator.evaluate(opp1_cards, community + [unrevealed_cards[turn_card_idx]] + [unrevealed_cards[river_card_idx]]))
					opp_eval = (_evaluator.evaluate(opp2_cards, community + [unrevealed_cards[turn_card_idx]] + [unrevealed_cards[river_card_idx]]))

					winning_rank = min([learner_eval, opp_eval])
					winning_players = [player for player, rank in enumerate([learner_eval, opp_eval]) if rank == winning_rank]
					if len(winning_players) is 2:
						learner_wins+=1
						opp_wins+=1
					else:
						if winning_players[0] == 0:
							learner_wins+=1
						else:
							opp_wins+=1
		

		elif _round == 2:

			for river_card in unrevealed_cards:
				player_handranks = []
				# print(Card.print_pretty_cards(community+[river_card]))
				learner_eval = (_evaluator.evaluate(opp1_cards, community+[river_card]))
				opp_eval = (_evaluator.evaluate(opp2_cards, community+[river_card]))

				winning_rank = min([learner_eval, opp_eval])
				winning_players = [player for player, rank in enumerate([learner_eval, opp_eval]) if rank == winning_rank]
				if len(winning_players) is 2:
					learner_wins+=1
					opp_wins+=1
				else:
					if winning_players[0] == 0:
						learner_wins+=1
					else:
						opp_wins+=1

		elif _round == 3:
			if self.learner_bot is self.winning_players:
				return 1.0, 0.0
			else:
				return 0.0, 1.0
		
		if opp_wins == 0 and learner_wins == 0:
			raise("error: division by zero")
		return (learner_wins/(learner_wins + opp_wins), opp_wins/(learner_wins + opp_wins))





	#Using evlaluation here. Might be better to use player.handstrength
	def compare_evaluations_players(self, my_action, respective_evaluations, evaluations_opposing_players):
		
		pass

		# expected_value = self.expected_value()
		
		# if my_action is ('fold', 0):
		# 	# calculate how good my cards are compared to raisers cards
		# 	_, raiser_bot = self.highest_in_LR()
		# 	raiser_strength = raiser_bot.he.evaluation
		# 	regret = (raiser_strength - respective_evaluations[self._current_player.get_seat()]) / self.weighting_coefficient_regret_fold
		# 	# Remember: Higher evaluation means worse cards, lower means better cards.
		# 	# e.g. If my evaluation was 5400, and my opponents evaluation was 7500, I would have positive regret ( I would regret having folded)
		# 	self._current_player.regret.update({'fold': regret})
		# elif my_action is ('check', 0):
		# 	# calculate how good my cards are compared to other players, and thus compute how much I regret not having raised
		# 	# If my evaluation is lower (better cards) than my opponents relatively high evaluation (worse cards), I would have positive regret
		# 	_, opposing_bot = self.current_player() # We can assign opposing as current_player (2-players heads-up) because we already rotated the table position
		# 	opposing_bot_strength = opposing_bot.he.evaluation
		# 	regret = (opposing_bot_strength - respective_evaluations[self._current_player.get_seat()]) / self.weighting_coefficient_regret_check
		# 	self._current_player.regret.update({'check': regret})
		# elif my_action[0] is 'call':
		# 	# Now we must compute the regret based on how much we would have been better of taking another action: Here, unlike other times, we have
		# 	# 2 possible alternatives : Raise or fold. If we take a call action, we must compute the expected value for the other alternatives. 
		# 	pass

		# elif my_action[0] is 'raise':
		# 	_, raiser_bot = self.highest_in_LR()
		# 	raiser_strength = raiser_bot.he.evaluation
		# 	regret = (raiser_evaluation - respective_evaluations[self._current_player.get_seat()]) / self.weighting_coefficient_regret_check
		# 	self._current_player.regret.update({'check': regret})