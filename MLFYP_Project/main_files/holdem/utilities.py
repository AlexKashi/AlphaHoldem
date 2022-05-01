from include import *

def which_round(community_cards):
	count_cards = 0
	_round = ''
	for i in community_cards:
		if not i == -1:
			count_cards = count_cards + 1
	if count_cards == 0:
		_round = 'Preflop'
	elif count_cards == 3:
		_round = 'Flop'
	elif count_cards == 4:
		_round = 'Turn'
	elif count_cards == 5:
		_round = 'River'
	return _round 

def fill_range_structure(_round, player):
	range_structure = None
	if _round == 'Preflop': 
		range_structure = preflop_range
	elif _round == 'Flop':
		range_structure = hand_strength_flop
	elif _round == 'Turn':
		range_structure = hand_strength_turn
	elif _round == 'River':
		range_structure = hand_strength_river
	return range_structure

def set_attributes(hand_strength, evaluation, player, rc, score_desc, event):
	if event == 'Preflop':
		if player.evaluation_preflop["he"] == '':
			player.evaluation_preflop["hand_strength"] = hand_strength
			player.evaluation_preflop["he"] = player.he
			player.evaluation_preflop["rc"] = rc
			player.evaluation_preflop["score_desc"] = score_desc
			player.evaluation_preflop["evaluation"] = evaluation
	elif event == 'Flop':
		if player.evaluation_flop["he"] == '':
			player.evaluation_flop["hand_strength"] = hand_strength
			player.evaluation_flop["he"] = player.he
			player.evaluation_flop["rc"] = rc
			player.evaluation_flop["score_desc"] = score_desc
			player.evaluation_flop["evaluation"] = evaluation
	elif event == 'Turn':
		if player.evaluation_turn["he"] == '':
			player.evaluation_turn["hand_strength"] = hand_strength
			player.evaluation_turn["he"] = player.he
			player.evaluation_turn["rc"] = rc
			player.evaluation_turn["score_desc"] = score_desc
			player.evaluation_turn["evaluation"] = evaluation
	elif event == 'River':
		if player.evaluation_river["he"] == '':
			player.evaluation_river["hand_strength"] = hand_strength
			player.evaluation_river["he"] = player.he
			player.evaluation_river["rc"] = rc
			player.evaluation_river["score_desc"] = score_desc
			player.evaluation_river["evaluation"] = evaluation

def highest_in_LR(player_o, env):
    highest_lr_bot = 0
    highest_lr_value = 0
    
    for key, value in env.level_raises.items():
        if value > highest_lr_value:
            highest_lr_value = value
            highest_lr_bot = key
    return highest_lr_value, highest_lr_bot

def calc_raises_i_face(player_o, env):
	bot_position_num = player_o.get_seat()
	my_lr_value = env.level_raises[bot_position_num]
	highest_lr_value, highest_lr_bot = highest_in_LR(player_o, env)
	add_me = highest_lr_value - my_lr_value
	return player_o.round['raises_i_owe'] + add_me
	

def assign_evals_player(player_o, _round, env):
	hand_strength, evaluation, rc, score_desc, hand, board = player_o.he.get_evaluation(_round)
	set_attributes(hand_strength, evaluation, player_o, rc, score_desc, _round)
	player_o.populatePlayerPossibleMoves(env)
	player_o.round['raises_i_owe'] = calc_raises_i_face(player_o, env)
	# print("")

def do_necessary_env_cleanup(env):
    is_end_game = False
    list_players = env._player_dict.copy()
    is_end_game = None
    for player in list_players.values():
        if player.stack <= 0:
            env.remove_player(player.get_seat())
            is_end_game = True # End Game
    env.assign_positions()
    return is_end_game

def convert_list_to_tupleA(learner_bot_state, community_state):
    
    info = [tuple(p) for p in learner_bot_state]
    info = tuple(info[0]+info[1])
    community = [tuple(p) for p in community_state]
    community = tuple(community[0]+community[1])
    states_in_episode = info + community
    return states_in_episode

class action_table:
	CHECK = 0
	CALL = 1
	RAISE = 2
	FOLD = 3
	NA = 0

def convert_step_return_to_action(action_from_step):
	if action_from_step[0] == 'call' or action_from_step[0] == 'check':
		return 0
	elif action_from_step[0] == 'raise' or action_from_step[0] == 'bet':
		return 1
	else:
		return 2

def safe_actions_call_bot(community_infos, villain_choice, n_seats):
	current_player = community_infos[-3]
	to_call = community_infos[-1]
	actions = [[action_table.CHECK, action_table.NA]] * n_seats
	if to_call > 0:
		if villain_choice is None:
			actions[current_player] = [action_table.CALL, action_table.NA]
		else:
			actions[current_player] = [villain_choice[0], villain_choice[1]]
	return actions

def compress_bucket(state, env, pre=False):
	if pre:
		state[0][0][0][0] = round((state[0][0][0][0]/env.starting_stack_size) * (state[0][0][0][0]/100))
		if state[0][0][0][1] != -1:
			state[0][0][0][1] = round((state[0][0][0][1]/7462) * (16900/100))
	else:
		for i in range(0,3,2):
			state[i][0][0] = round((state[i][0][0]/env.starting_stack_size) * (state[i][0][0]/100))
			if state[i][0][1] != -1:
				state[i][0][1] = round((state[i][0][1]/7462) * (16900/100))

	return state


def convert_step_return_to_set(sar):
    
    player_features_tuples = []
    player_cards_tuples = []
    community_state_tuples = []
    pf = sar[0][0][0][0]
    player_features = tuple(pf)
    player_features_tuples.append(player_features)

    pf = sar[0][0][0][1]
    player_cards = tuple(pf)
    player_cards_tuples.append(player_cards)

    pf = sar[0][1][0]
    community_state = tuple(pf)
    community_state_tuples.append(community_state)

    # states_in_episode = list(set([sar[0] for sar in episode])) # sar--> state,action,reward
    states = []
    for i in range(len(player_features_tuples)):
        my_tup = (player_features_tuples[i] + player_cards_tuples[i] + community_state_tuples[i])
        states.append(my_tup)

    return states
