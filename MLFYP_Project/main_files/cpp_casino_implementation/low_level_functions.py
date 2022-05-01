import re 
import Player as p
import numpy as np


def create_cards_for_game():
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

def casinoToBot_ParsingUpdateUniversal(self, file_data_original_change, plr, player_list, player_action):
    
    # arr =  re.split(r'[DPFFFFTTRR]',file_data_original_change)
    # pre_flop_last_move = arr[2]
    # btc_file = "/casinoToBotUniversal"
    # file_name = communication_files_directory='/usr/local/home/u180455/Desktop/Project/MLFYP_Project/MLFYP_Project/pokercasino/botfiles' + btc_file 
    # with open(file_name, 'wt') as f:
    #     f.append(pre_flop_last_move)
    #     f.close()
    # print("last_move:", pre_flop_last_move)
    pass

def count_showd(file_data):
    count_showdown = 0
    for letter in file_data:
        if letter == 'S':
            count_showdown = count_showdown + 1

    return count_showdown

def count_winr(file_data):
    count_winners = 0
    for letter in file_data:
        if letter == 'W':
            count_winners = count_winners + 1

    return count_winners

def getWinnersShowdown(ctb_file_content, file_data):

    count_showdown = count_showd(file_data)
    count_winners = count_winr(file_data)
    showdown = []
    winners = []

    
    for win in range(count_winners):
        winners.append(ctb_file_content[len(ctb_file_content) - (win + 2) ])


    for show in range(0, count_showdown*3, 3):
        a = []
        for block in range(3):
            b = (ctb_file_content[len(ctb_file_content) - (len(winners) + show + block + 2)])
            a.append(b)
        a = list(reversed(a))
        showdown.append(a)

    winners = list(reversed(winners))
    showdown = list(reversed(showdown))

    return winners, showdown


def hand_summary(balance, file_data):

    file_data_change_CTB =  re.split(r'[DPFFFFTTRRSABWWE]',file_data)
    p.Player.game_state['winners'] = []

    with open("/home/gary/Desktop/MLFYP_Project/MLFYP_Project/main_files/" + "files_change", 'a+') as f:
        
        st = file_data + "\n" + "\thand.no: " + file_data_change_CTB[0] + "\tcount_showd(file_data)" + str(count_showd(file_data)) + "\t" + '(count_winr(file_data)' + str(count_winr(file_data)) 
        f.write(st)
        f.close()

    if count_showd(file_data) > 0 and count_winr(file_data) > 0:
        
        winners_arr, showdown_arr = getWinnersShowdown(file_data_change_CTB, file_data)
                
        for i in range(1, len(showdown_arr) + 1):
            p.Player.game_state['p'+str(i)]['position_showdown'] = showdown_arr[i-1][0]
            p.Player.game_state['p'+str(i)]['cards'][0] = showdown_arr[i-1][1] 
            p.Player.game_state['p'+str(i)]['cards'][1] = showdown_arr[i-1][2] 

        for winner in winners_arr:
           
            p.Player.game_state['winners'].append(winner)
    
        with open("/home/gary/Desktop/MLFYP_Project/MLFYP_Project/main_files/" + "test_file_data_change", 'a+') as f:
            st = str(p.Player.game_state) + "\n"
            f.write(st)
            f.close()

    

def casinoToBot_ParsingRead(self, file_data_change_CTB, player_list, bot_no, file_data, is_get_last_action):
    # <hand number> D <dealer button position> P <action by all players in order from first to 
    # act, e.g. fccrf...> F <flop card 1> F <flop 2> F <flop 3> F <flop action starting with first player to act>
    # T <turn card> T <turn action> R <river card> R <river action>
    # print("len(fdcctb):", len(file_data_change_CTB))
    is_preflop_action_filled = False
    is_flop_action_filled = False
    is_turn_action_filled = False
    is_river_action_filled = False

    button = file_data_change_CTB[1]
    iDealer = False

    if not is_get_last_action:
        self.player_list[int(bot_no)].position = ''
        number_of_players = 3 
        if button == bot_no:
            iDealer = True
            self.player_list[int(bot_no)].position = 'BTN'
            self.player_list[(int(bot_no)+1) % number_of_players].position = 'SB'
            self.player_list[(int(bot_no)+2) % number_of_players].position = 'BB'

        if button == str((int(bot_no) + 1) % number_of_players):
            iDealer = False
            self.player_list[int(bot_no)].position = 'BB'
            self.player_list[(int(bot_no)+1) % number_of_players].position = 'BTN'
            self.player_list[(int(bot_no)+2) % number_of_players].position = 'SB'

        if button == str((int(bot_no) + 2) % number_of_players):
            iDealer = False
            self.player_list[int(bot_no)].position = 'SB'
            self.player_list[(int(bot_no)+1) % number_of_players].position = 'BB'
            self.player_list[(int(bot_no)+2) % number_of_players].position = 'BTN'

    # need to do same for other rotations of table other than the standard fixation
    #print("{}.. Hand Number: {}, Game_status: {}".format(plr, file_data_change_CTB[0], file_data_change_CTB))
    # Here we must update the local static game_state variable to the status read from CasinoToBot
    blank = True
    if file_data_change_CTB[0] != None and file_data_change_CTB[0] != '':
        blank = False
    
    if blank == False:
        #HAND NO
        
        if file_data_change_CTB[0] != None:
            p.Player.game_state['hand_no'] = file_data_change_CTB[0]

        
        #DEALER POSITION
        if file_data_change_CTB[1] != None:
            for i in range(len(player_list)):
                if str(file_data_change_CTB[1]) == str(i):
                    player_list[i].dealer_status = True
                    p.Player.game_state['dealer_position'] = i
                else: 
                    player_list[i].dealer_status = False


        if file_data_change_CTB[2] != None:
            if file_data_change_CTB[2] == '':
                is_preflop_action_filled = False
            else:
                is_preflop_action_filled = True
        else: 
            is_preflop_action_filled = False



        if len(file_data_change_CTB) >= 3:
            #PREFLOP ACTION
            ## FIX: GET RID OF FIRST IF STATEMENT
            
                
            if file_data_change_CTB[2] != None:
                p.Player.game_state['action_preflop'] = file_data_change_CTB[2]
       

        if len(file_data_change_CTB) >= 7:

            if file_data_change_CTB[3] != None and file_data_change_CTB[4] != None and file_data_change_CTB[5] != None:
                p.Player.game_state['flop1'] = file_data_change_CTB[3]
                p.Player.game_state['flop2'] = file_data_change_CTB[4]
                p.Player.game_state['flop3'] = file_data_change_CTB[5]

            #FLOP ACTION
            if file_data_change_CTB[6] != None:
                p.Player.game_state['action_flop'] = file_data_change_CTB[6]


            if file_data_change_CTB[6] != None:
                if file_data_change_CTB[6] == '':
                    is_flop_action_filled = False
                else:
                    is_flop_action_filled = True
            else: 
                is_flop_action_filled = False


        if len(file_data_change_CTB) >= 9:
            #TURN 
            if file_data_change_CTB[7] != None:
                p.Player.game_state['turn'] = file_data_change_CTB[7]

            #TURN ACTION 
            if file_data_change_CTB[8] != None:
                p.Player.game_state['action_turn'] = file_data_change_CTB[8]

            if file_data_change_CTB[8] != None:
                if file_data_change_CTB[8] == '':
                    is_turn_action_filled = False
                else:
                    is_turn_action_filled = True
            else: 
                is_turn_action_filled = False

            
        if len(file_data_change_CTB) >= 11:

            #RIVER 
            if file_data_change_CTB[9] != None:
                p.Player.game_state['river'] = file_data_change_CTB[9]


            #RIVER ACTION 
            if file_data_change_CTB[10] != None:
                p.Player.game_state['action_river'] = file_data_change_CTB[10]
                
           

            # including preflop, flop and turn for DEBUGGING
            if file_data_change_CTB[2] != None:
                if file_data_change_CTB[2] == '':
                    is_preflop_action_filled = False
                else:
                    is_preflop_action_filled = True
            else: 
                is_preflop_action_filled = False

            if file_data_change_CTB[6] != None:
                if file_data_change_CTB[6] == '':
                    is_flop_action_filled = False
                else:
                    is_flop_action_filled = True
            else: 
                is_flop_action_filled = False

            if file_data_change_CTB[8] != None:
                if file_data_change_CTB[8] == '':
                    is_turn_action_filled = False
                else:
                    is_turn_action_filled = True
            else: 
                is_turn_action_filled = False

            if file_data_change_CTB[10] != None:
                if file_data_change_CTB[10] == '':
                    is_river_action_filled = False
                else:
                    is_river_action_filled = True
            else: 
                is_river_action_filled = False
        

    return is_preflop_action_filled, is_flop_action_filled, is_turn_action_filled, is_river_action_filled

def get_last_action(file_data, a, b, c, d):
    file_data_change_CTB =  re.split(r'[DPFFFFTTRRSABWWE]',file_data)
    if(a and b and c and d):
        p.Player.game_state['action_river'] = file_data_change_CTB[10]
    elif(a and b and c and not d):
        p.Player.game_state['action_turn'] = file_data_change_CTB[8]
    elif(a and b and not c and not d):
        p.Player.game_state['action_flop'] = file_data_change_CTB[6]
    elif(a and not b and not c and not d):
        p.Player.game_state['action_preflop'] = file_data_change_CTB[2]

def check_cards_shown(file_data_change_CTB):
    flop_cards = []
    turn_card = ''
    river_card = ''
    file_data_change_CTB =  re.split(r'[DPFFFFTTRR]',file_data_change_CTB)
    flop_cards_present = False
    turn_cards_present = False
    river_cards_present = False
    if len(file_data_change_CTB) > 3:
        flop_cards.extend([file_data_change_CTB[3], file_data_change_CTB[4], file_data_change_CTB[5]])
    if len(file_data_change_CTB) > 7:
        turn_card = (file_data_change_CTB[7])
    if len(file_data_change_CTB) > 9:
        river_card = (file_data_change_CTB[9])
        
        
    count_flop_cards = 0
    for card in flop_cards:
        
        if card != None and card != '':
            count_flop_cards = count_flop_cards + 1
    
    if(count_flop_cards == 3):
        flop_cards_present = True
    
    if (turn_card != None and turn_card != ''):
        turn_cards_present = True

    if (river_card != None and river_card != ''):
        river_cards_present = True

    return flop_cards_present, turn_cards_present, river_cards_present
    
        

def count_r(my_string):
        count_r = 0
        for letter in my_string:
            if letter == 'r':
                count_r = count_r + 1

        return count_r

def GHB_Parsing(player, GHB_Status):
    
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
    for card in player.cards:
        if(str(player.cards.index(card)) == card_a):
            if(len(card) == 2):
                a,b = card
            # elif(len(card) == 3):
            #     a,b,c = card
            #     a = a+b
            #     b = c
        elif(str(player.cards.index(card))== card_b):
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
    player.card_holding =   p.CardHolding(player.name,card_a_suit,card_a_rank,card_b_suit, card_b_rank)
    
    return player.card_holding

def from_num_to_cardstring(my_card):
  
    deck_size = 52
    suits = ['h','c','s','d']
    
    card_a_suit = ''
    card_a_rank = ''
    a,b = ('', '')
    cards_for_game = create_cards_for_game()
    for card in cards_for_game: ## all cards in game
        if(str(cards_for_game.index(card)) == my_card):
            if(len(card) == 2):
                a,b = card
                break
    card_a_rank = a
    card_a_suit = b
    
       
    return str(a+b)