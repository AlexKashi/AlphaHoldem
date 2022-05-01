import numpy as np
import sys
import os
from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../") 
sys.path.append(os.path.abspath(os.path.join('..'))+"/MLFYP_Project/main_files/holdem")
import utilities
from include import *
import gym
import holdem
import tkinter as tk
from tkinter import ttk
from treys import Card
import time
from PIL import Image, ImageTk

from monte_carlo import get_action_policy, make_epsilon_greedy_policy
# from DQN import create_np_array, agent

with_render = False

n_episodes = 100 # n games we want agent to play (default 1001)

villain = "CallChump"

starting_stack_size = 2000

epsilon = 0.9


env = gym.make('TexasHoldem-v1') # holdem.TexasHoldemEnv(2)
env.add_player(0, stack=starting_stack_size) # add a player to seat 0 with 2000 "chips"
# env.add_player(1, stack=2000) # tight
env.add_player(2, stack=starting_stack_size) # aggressive

LARGE_FONT= ("Verdana", 17)



class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        self.state = [None, False]
        tk.Tk.__init__(self, *args, **kwargs)

        # tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "Texas Hold'em Casino")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        
        for F in (StartPage, PageOne, PagePokerGameMC, PagePokerGameDQN, StartGame):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

        # PagePokerGameMC.simulation(PagePokerGameMC)

    def show_frame(self, cont, info=None, ag=None):

        frame = self.frames[cont]
        frame.info = info
        frame.ag = ag
        if frame.__str__() == 'StartGame':
            frame.start()
        frame.tkraise()

    def receive_info(self, state):
        self.state[0] = state
        self.state[1] = False

    def get_state(self):
        self.state[1] = True
        return self.state[0]


        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent, bg='#218c16')
        label = tk.Label(self, text="Main Menu", font=LARGE_FONT, bg='#218c16')
        label.pack(pady=10,padx=10)

        s = ttk.Style()
        s.configure('Kim.TButton', foreground='maroon', padding = 6, relief='raised', background="#ccc")

        button = ttk.Button(self, text="Analyze Agents", style = 'Kim.TButton',
                            command=lambda: controller.show_frame(PageOne, info="Analyze"))
        button.place(relx=0.5, rely=0.45, anchor='center')

        button2 = ttk.Button(self, text="Compete Against Agents", style = 'Kim.TButton',
                            command=lambda: controller.show_frame(PageOne, info="Compete"))
        button2.place(relx=0.5, rely=0.55, anchor='center')

        

class PageOne(tk.Frame):

    def __init__(self, parent, controller, info=None):
        tk.Frame.__init__(self, parent, bg='#218c16')
        label = tk.Label(self, text="Choose Agent", font=LARGE_FONT, bg='#218c16')
        label.pack(pady=10,padx=10)
        self.info = info
        s = ttk.Style()
        s.configure('Kim.TButton', foreground='maroon', padding = 6, relief='raised', background="#ccc")

        button1 = ttk.Button(self, text="Monte-Carlo Agent", style='Kim.TButton',
                            command=lambda: controller.show_frame(PagePokerGameMC, info=self.info, ag='MC'))
        button1.place(relx=0.5, rely=0.45, anchor='center')

        button2 = ttk.Button(self, text="Deep Q-Learning Agent", style='Kim.TButton',
                            command=lambda: controller.show_frame(PagePokerGameDQN, info=self.info, ag='DQN'))
        button2.place(relx=0.5, rely=0.55, anchor='center')




class PagePokerGameMC(tk.Frame):

    def __init__(self, parent, controller, info=None, ag=None):

        tk.Frame.__init__(self, parent, bg='#218c16')
        label = tk.Label(self, text="Monte Carlo Agent", font=("Arial Bold", 30), bg='#218c16')
        label.pack(pady=10,padx=10)
        self.info = info 
        self.ag = ag
        start_button = ttk.Button(self, text="Start Game",
                            command=lambda: controller.show_frame(StartGame, info=self.info, ag=self.ag))
        start_button.pack() 

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage, info=self.info))
        button1.pack()


class PagePokerGameDQN(tk.Frame):

    def __init__(self, parent, controller, info=None, ag=None):

        tk.Frame.__init__(self, parent, bg='#218c16')
        label = tk.Label(self, text="Deep Q-Learning Agent", font=("Arial Bold", 30), bg='#218c16')
        label.pack(pady=10,padx=10)
        self.info = info 
        self.ag = ag
        start_button = ttk.Button(self, text="Start Game",
                            command=lambda: controller.show_frame(StartGame, info=self.info, ag=self.ag))
        start_button.pack() 

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage, info=self.info))
        button1.pack()

		

class StartGame(tk.Frame):

	def __init__(self, parent, controller, info=None, ag=None):
		tk.Frame.__init__(self, parent, bg='#218c16')
		self.info = info
		self.ag = ag

		dir_path = os.path.dirname(os.path.realpath(__file__))
		card_image = Image.open(dir_path[0:-4]+"/JPEG/coollogo_com-16865976.png")
		photo = ImageTk.PhotoImage(card_image)
		label = tk.Label(self, image=photo, bg='#218c16')
		label.image = photo # keep a reference!

		label.pack()
		button1 = ttk.Button(self, text="Back to Home",
							command=lambda: controller.show_frame(StartPage))
		button1.pack()

		self.separator = tk.LabelFrame(self, width=50, height=150, text="Board", bd=10, bg='OrangeRed4')
		self.separator.pack(fill='x', padx=5, pady=40)
		self.s = ttk.Style()
		self.s.configure('call.TButton', foreground='green', bg='#c1bb78')
		self.s.configure('raise.TButton', foreground='red', bg='#c1bb78')
		self.s.configure('fold.TButton', foreground='blue', bg='#c1bb78')

		self.returns_sum = defaultdict(float)
		self.returns_count = defaultdict(float)
		self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
		self.policy = make_epsilon_greedy_policy(self.Q, env.action_space.n, epsilon)

		self.guest_cards = []
		self.learner_cards = []
		self.state = None
		self.player_states, self.community_infos, self.community_cards = None, None, None
		self.player_infos, self.player_hands = None, None
		self.current_state = None
		self.state_set = None
		self.p1 = env._player_dict[0] # Learner
		self.p2 = env._player_dict[2] # Guest
		self.episode_list = []
		self.total_pot_label = None
		self._round = None
		self.current_player = None
		self.guest_action = None
		self.call_button = None
		self.raise_button = None
		self.fold_button = None
		self.guest_buttons = [self.call_button, self.raise_button, self.fold_button]
		self.terminal = False
		self.guest_label = None
		self.learner_label = None
		self.p1_pos = None
		self.ps_pos = None
		self.guest_cards_st, self.learner_cards_st = None, None
		self.episodes = []
		self.last_bet_label = None
		self.community_display = []
		self.is_new_game = False
		self.is_end_game = False
		self.mrp = None
		self.sd_tr = False
		self.last_learner_cards = None
		self.initial_action = True
		# self.part_init = None

	def __str__(self):
		return 'StartGame'

	def start(self):
		self.delegate_state_info(reset=True)
		self.simulate(initial=True)
		
	def restart_game(self):
		self.is_new_game = True
		self.guest_action = None
		if self.terminal and self.info == 'Compete':
			if self.is_showdown():
				self.sd_tr = True
				self.last_learner_cards = tk.Label(self, text="Last Cards:\n" + str(self.learner_cards_st[0]+"\n"+self.learner_cards_st[1]), font=LARGE_FONT, bg='#218c16')	
				self.last_learner_cards.place(relx=0.85, rely=0.55, anchor='center')
		self.is_end_game = utilities.do_necessary_env_cleanup(env)
		self.delegate_state_info(reset=True)

		if (self.community_infos[-3] == 0):
			self.simulate(initial=self.is_initial())
		
	def is_initial(self):
		if self.is_new_game:
			initial = True
		else:
			initial = False
		return initial

	def set_guest_action(self, action):
		if self.sd_tr:
			if self.last_learner_cards is not None:
				self.last_learner_cards.place_forget()
			self.sd_tr = False
		self.update_local_state(reset=False)
		self.guest_action = action
		initial = self.is_initial()
		if (self.community_infos[-3] == 2):
			self.simulate(initial=self.is_initial())
		
		
		
		
	def simulate(self, initial = False):
		# for i_episode in range(1, n_episodes + 1):
		episode = None
		self.populate_info_pre_action()
		if self.current_player == 0:
			episode = self.generate_episode_learner_move()
		elif self.current_player == 2:
			episode = self.generate_episode_guest()
		self.episodes.append(episode)

		if self.terminal:
			self.sd_tr = True
			
			self.restart_game()
		else:
			self.update_local_state(reset=False)
			# if self.community_infos[-3] == 0:
				# time.sleep(0.5)
			self.update_display()

		if initial:
			self.is_new_game = False
		        
		if self.community_infos[-3] == 0: # When learner is BB, has first go flop (last go preflop)
			self.simulate(initial=self.is_initial())
		
	def update_display(self):
		self.assign_player_objects_to_display(reset=True)
		
		self.assign_cards_to_display(self.guest_cards_st, self.learner_cards_st, reset=True)
		
		self.assign_guest_buttons()

		self.update_pot_size()

		self.print_last_action()

	def is_showdown(self):
		# If both players are still left in the game
		if env.level_raises[self.p1.get_seat()] == env.level_raises[self.p2.get_seat()]:
			if env.winning_players == self.p1:
				return True
		else:
			return False




	def print_last_action(self, spec=None):
		# if self.last_bet_label is not None:
		# 	self.last_bet_label.place_forget()
	
		# if self.mrp is 0:
		# 	self.last_bet_label = tk.Label(self, text="Learner action:\n" + str(env._last_actions[0]), font=LARGE_FONT, bg='#218c16')	
			
		# 	self.last_bet_label.place(relx=0.85, rely=0.90, anchor='center')

		return "Learner action:\n" + str(env._last_actions[0])


	def update_local_state(self, reset=True):
		self.p1_pos = 'SB' if self.p1.position == 0 else 'BB' # Learner
		self.p2_pos = 'SB' if self.p2.position == 0 else 'BB' # Guest
		self.mrp = self.current_player
		if reset:
			self.state = env.reset()
			self.set_info_before_loop()
			
		else:
			self.state = env._get_current_state()
			self.set_info_before_loop()
			# self.update_display()

	def delegate_state_info(self, reset):
		self.update_local_state(reset=reset)
		
		self.assign_player_objects_to_display(reset=reset)

		self.guest_cards_st = [Card.int_to_str(self.p2.hand[0]).upper(), Card.int_to_str(self.p2.hand[1]).upper()]
		self.learner_cards_st = [Card.int_to_str(self.p1.hand[0]).upper(), Card.int_to_str(self.p1.hand[1]).upper()]

		self.assign_cards_to_display(self.guest_cards_st, self.learner_cards_st, reset=reset)

		self.update_pot_size()

	def set_info_before_loop(self):
		# (player_states, (community_infos, community_cards)) = env.reset()
		(self.player_states, (self.community_infos, self.community_cards)) = self.state
		(self.player_infos, self.player_hands) = zip(*self.player_states)
		self.current_state = ((self.player_infos, self.player_hands), (self.community_infos, self.community_cards))
		utilities.compress_bucket(self.current_state, env, pre=True)
		if self.ag == 'DQN':
			self.state = create_np_array(self.player_infos, self.player_hands, self.community_cards, self.community_infos)
		self.state_set = utilities.convert_list_to_tupleA(self.player_states[env.learner_bot.get_seat()], self.current_state[1])
		

	def update_pot_size(self):
		if self.total_pot_label is not None:
			self.total_pot_label.pack_forget()
		self.total_pot_label = tk.Label(self, text="Pot:\n{}\n".format(env._totalpot), font=("Arial Bold", 20), bg='#218c16')
		self.total_pot_label.pack(side='top', pady=40,padx=40)

	def assign_guest_buttons(self):
		for button in self.guest_buttons:
			if button is not None:
				button.pack_forget()
		
		self.fold_button = ttk.Button(self, text="Fold", style="fold.TButton",
						command=lambda: self.set_guest_action('f'))
		self.fold_button.pack(side='bottom', padx=5, pady=5)

		self.raise_button = ttk.Button(self, text="Raise", style="raise.TButton",
						command=lambda: self.set_guest_action('r'))
		self.raise_button.pack(side='bottom', padx=5, pady=5)
		
		self.call_button = ttk.Button(self, text="Call", style="call.TButton",
						command=lambda: self.set_guest_action('c'))
		self.call_button.pack(side='bottom', padx=5, pady=5)
		self.guest_buttons = [self.call_button, self.raise_button, self.fold_button]

		

	def assign_cards_to_display(self, guest_cards_st, learner_cards_st, reset = False):
		if reset:
			for card in self.guest_cards+self.learner_cards:
				card.pack_forget()
		position_cards = [0, 0]
		
		for card in self.guest_cards_st:
			guest_card = self.form_image(card)
			guest_card.pack(side='left', expand = False, padx=position_cards[0], pady=position_cards[1])
			self.guest_cards.append(guest_card)
        
		for card in self.learner_cards_st:
			learner_card = self.form_image(card, learner=True) #if not self.sd_tr else self.form_image(card, learner=False)
			learner_card.pack(side='right', expand = False, padx=position_cards[0], pady=position_cards[1])
			self.learner_cards.append(learner_card)
		cd = []
		if self.community_display is not None:
			for card in self.community_display:
				card.pack_forget()

		if self.community_cards is not None:
			if not(all(i < 0 for i in self.community_cards)):
				if self.community_cards[0] is not -1 and self.community_cards[1] is not -1 and self.community_cards[2] is not -1:
					cd.append(Card.int_to_str(self.community_cards[0]).upper())
					cd.append(Card.int_to_str(self.community_cards[1]).upper())
					cd.append(Card.int_to_str(self.community_cards[2]).upper())
				if self.community_cards[3] is not -1:
					cd.append(Card.int_to_str(self.community_cards[3]).upper())
				if self.community_cards[4] is not -1:
					cd.append(Card.int_to_str(self.community_cards[4]).upper())
				for card in cd:
					c = self.form_image(card, community=True)
					c.pack(side='left', expand = False, padx=20, pady=20)
					self.community_display.append(c)

	def assign_player_objects_to_display(self, reset=False):
		if reset and self.guest_label is not None and self.learner_label is not None:
			self.guest_label.pack_forget()
			self.learner_label.pack_forget()
			self.button_label.place_forget()
		position_cards = [10, 10]
		self.guest_label = tk.Label(self, text="Guest\n\nStack:{}\n{}".format(self.p2.stack, self.p2_pos), font=("Arial Bold", 12), bg='#218c16')
		self.guest_label.pack(side='left', pady=40,padx=40)
		self.learner_label = tk.Label(self, text="Learner\n\nStack:{}\n{}".format(self.p1.stack, self.p1_pos), font=("Arial Bold", 12), bg='#218c16')
		self.learner_label.pack(side='right', pady=40,padx=40)

		if self.p1_pos == 'SB':
			

			dir_path = os.path.dirname(os.path.realpath(__file__))
			card_image = Image.open(dir_path[0:-4]+"/JPEG/dealerbutton-1000.jpg")
			photo = ImageTk.PhotoImage(card_image)
			self.button_label = tk.Label(self, image=photo, bg='#218c16')
			self.button_label.image = photo # keep a reference!
			self.button_label.place(relx=0.75, rely=0.58, anchor='center')

		else:

			dir_path = os.path.dirname(os.path.realpath(__file__))
			card_image = Image.open(dir_path[0:-4]+"/JPEG/dealerbutton-1000.jpg")
			photo = ImageTk.PhotoImage(card_image)
			self.button_label = tk.Label(self, image=photo, bg='#218c16')
			self.button_label.image = photo # keep a reference!
			self.button_label.place(relx=0.25, rely=0.58, anchor='center')


		# if self.community_infos is not None:
		# 	cp = self.community_infos[-1]
		# 	if cp == 0:
		# 		self.learner_label.config(bg="red")
		# 		self.guest_label.config(bg="white")
		# 	elif cp == 2:
		# 		self.guest_label.config(bg="red")
		# 		self.learner_label.config(bg="white")
			

	def form_image(self, card, community=False, learner=False):
		
		dir_path = os.path.dirname(os.path.realpath(__file__))
		if learner and self.info == "Compete":
			card_image = Image.open(dir_path[0:-4]+"/JPEG/Red_back.jpg")
		else:
			card_image = Image.open(dir_path[0:-4]+"/JPEG/"+ card +".jpg")
		photo = ImageTk.PhotoImage(card_image)
		label = tk.Label(self, image=photo, bg='#218c16') if community is False else tk.Label(self.separator, image=photo, bg='#218c16')
		label.image = photo # keep a reference!
		return label

	def parse_action(self, action):
		if action == 'c':
			return [(1, 0), (0, 0)]
		elif action == 'r':
			total_bet = None
			if self._round == 'Preflop' and self.p2.position == 0:
			
				total_bet = 40
			else:
				total_bet = 25

			action = (2, total_bet)
			assert action[1] == 40 or action[1] == 25
			return action
		elif action == 'f':
			return [3, 0]

	def get_guest_action(self):
		action = self.guest_action
		action = self.parse_action(action)
		player_actions = holdem.safe_actions(self.community_infos[-1], self.community_infos, action, n_seats=env.n_seats, choice=None, player_o = self.p2)
		return player_actions
		

	# v = mc_prediction_poker(10)
	# # for line_no, line in enumerate(v.items()):
	# #     print(line_no, line)

	# plotting.plot_value_function(v, title="10 Steps")




	def populate_info_pre_action(self):
		self._round = utilities.which_round(self.community_cards)
		self.current_player = self.community_infos[-3]


	def get_action_for_page(self):
		if self.ag == 'MC':
			if self.current_player == 0: # Learner (RHS Screen)
				self.action = get_action_policy(self.player_infos, self.community_infos, self.community_cards, env, self._round, env.n_seats, self.state_set, self.policy)
				if self.initial_action and self.action[self.current_player] == [3, 0]:
					self.action[self.current_player] = [1, 0]
			elif self.current_player == 2: # Player on page (LHS Screen)
				self.action = self.get_guest_action()
		elif self.ag == 'DQN':
			if self.current_player == 0: # Learner (RHS Screen)
				self.action = agent.act(self.state, self.player_infos, self.community_infos, self.community_cards, env, self._round, env.n_seats, self.state_set, self.policy)
				if self.initial_action and self.action[self.current_player] == [3, 0]:
					self.action[self.current_player] = [1, 0]
			elif self.current_player == 2: # Player on page (LHS Screen)
				self.action = self.get_guest_action()
		
		self.initial_action = False



	def generate_episode_guest(self):
		part_ep = []
		
		self.get_action_for_page()
		
		(self.player_states, (self.community_infos, community_cards)), self.action, rewards, self.terminal, info = env.step(self.action)

		utilities.compress_bucket(self.player_states, env)
		parsed_return_state = utilities.convert_step_return_to_set((self.current_state, self.action, env.learner_bot.reward))
		self.action = utilities.convert_step_return_to_action(self.action)
		if self.ag == 'DQN':
			ps = list(zip(*self.player_states))
			next_state = create_np_array(ps[0], ps[1], self.community_cards, self.community_infos) # Numpy array
			agent.remember(self.state, self.action, env.learner_bot.reward, next_state, self.terminal)
			self.state = next_state
		part_ep.append((parsed_return_state, self.action, env.learner_bot.reward))
		current_state = (self.player_states, (self.community_infos, self.community_cards)) # state = next_state
		
		return part_ep

	def generate_episode_learner_move(self):
		episode = []
		
		self.get_action_for_page()
		
		(self.player_states, (self.community_infos, community_cards)), self.action, rewards, self.terminal, info = env.step(self.action)

		parsed_return_state = utilities.convert_step_return_to_set((self.current_state, self.action, env.learner_bot.reward))
		self.action = utilities.convert_step_return_to_action(self.action)
		episode.append((parsed_return_state, self.action, env.learner_bot.reward))
		current_state = (self.player_states, (self.community_infos, self.community_cards)) # state = next_state
		
		return episode



if __name__ == '__main__':
        
    app = SeaofBTCapp()
    app.geometry("1280x720")
    app.mainloop()