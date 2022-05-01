from treys import Card

class action_table:
  CHECK = 0
  CALL = 1
  RAISE = 2
  FOLD = 3
  NA = 0


def format_action(player, action):
  color = False
  try:
    from termcolor import colored
    # for mac, linux: http://pypi.python.org/pypi/termcolor
    # can use for windows: http://pypi.python.org/pypi/colorama
    color = True
  except ImportError:
    pass
  [aid, raise_amt] = action
  if aid == 'check':
    text = '_ CHECK'
    if color:
      text = colored(text, 'white')
    return text
  if aid == 'call':
    text = '- CALL, call amount: {}'.format(player.currentbet)
    if color:
      text = colored(text, 'yellow')
    return text
  if aid == 'raise':
    text = '^ RAISE, bet amount: {}'.format(raise_amt)
    if color:
      text = colored(text, 'green')
    return text
  if aid == 'fold':
    text = 'fold'
    if color:
      text = colored(text, 'red')
    return text


def card_to_str(card):
  if card == -1:
    return ''
  return Card.int_to_pretty_str(card)


def hand_to_str(hand):
  output = " "
  for i in range(len(hand)):
    c = hand[i]
    if c == -1:
      if i != len(hand) - 1:
        output += '[  ],'
      else:
        output += '[  ] '
      continue
    if i != len(hand) - 1:
      output += str(Card.int_to_pretty_str(c)) + ','
    else:
      output += str(Card.int_to_pretty_str(c)) + ' '
  return output


def safe_actions(to_call, community_infos, villain_choice, n_seats, choice=None, player_o=None, best_nonlearning_action=None):
  current_player = community_infos[-3]
  # to_call = community_infos[-1]
  actions = [[action_table.CHECK, action_table.NA]] * n_seats
  
  if to_call > 0:
    # CALL/RAISE (Rule excludes opening up with paying of the blinds)
    if player_o.stack <= 25: 
      actions[current_player] = [action_table.CALL, action_table.NA]
      return actions
    if villain_choice is None: # Learner bot
      if choice == 0:
        actions[current_player] = [action_table.CALL, action_table.NA]
      elif type(choice) is tuple:
        if player_o.is_possible('r') and player_o.stack > 25:
          actions[current_player] = [choice[0], choice[1]] 
        else:
          actions[current_player] = [action_table.CALL, action_table.NA]
      elif choice == 1:
        if player_o.is_possible('r') and player_o.stack > 25:
          actions[current_player] = [2, 50]
        else:
          if player_o.round['raises_i_owe'] > 0:
            actions[current_player] = [action_table.CALL, action_table.NA]
          else:
            actions[current_player] = [action_table.CHECK, action_table.NA]
      else:
        actions[current_player] = [3, 0]
    else:
      if type(villain_choice) is list: # Call
        
        if villain_choice == [3, 0]:
          actions[current_player] = [3, 0] # ~
        else: 
          actions[current_player] = [villain_choice[0][0], villain_choice[0][1]]
      else:
        actions[current_player] = [villain_choice[0], villain_choice[1]]
  else:
    ## This is where a player may take initiative and BET (Rule excludes opening up with paying of the blinds)
    ## They may also CHECK
    if player_o.stack <= 25: 
      actions[current_player] = [action_table.CHECK, action_table.NA]
      return actions

    if villain_choice is None: # Learner bot
      if choice == 0:
        actions[current_player] = [action_table.CHECK, action_table.NA]
      elif type(choice) is tuple:
        actions[current_player] = [choice[0], choice[1]]
      elif choice == 1:
        if player_o.is_possible('r') and player_o.stack > 25:
            actions[current_player] = [2, 25] 
        else:
          if player_o.round['raises_i_owe'] > 0:
            actions[current_player] = [action_table.CALL, action_table.NA]
          else:
            actions[current_player] = [action_table.CHECK, action_table.NA]
    else:
      if type(villain_choice) is list: # Check
        
        if villain_choice == [3, 0]:
          actions[current_player] = [3, 0] # ~
        else:
          actions[current_player] = [villain_choice[1][0], villain_choice[1][1]]
      else:
        if [villain_choice[0], villain_choice[1]] == [3, 0]: # Prevent against folding when to_call = 0
          actions[current_player] = [action_table.CHECK, action_table.NA]
        else:
          actions[current_player] = [villain_choice[0], villain_choice[1]]
  # if actions[current_player][0] is 2:
  #   print("e")
  return actions		


def safe_actions_call_bot(community_infos, which_action, n_seats):
  current_player = community_infos[-3]
  to_call = community_infos[-1]
  actions = [[action_table.CHECK, action_table.NA]] * n_seats
  if to_call > 0:
    if which_action is None:
      actions[current_player] = [action_table.CALL, action_table.NA]
    else:
      actions[current_player] = [which_action[0], which_action[1]]
  return actions