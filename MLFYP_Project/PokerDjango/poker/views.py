from django.shortcuts import render, get_object_or_404
import datetime
from django.http.response import HttpResponse
# Create your views here.
import os
import sys
sys.path.insert(0, "/home/gary/Desktop/Dev/Python/MLFYP_Project/main_files/holdem")
from collections import defaultdict
import numpy as np
from django.views import View
import poker.models as poker_model
from .serializers import ( 
    GameSerializer, 
    PlayerSerializer,
    CardSerializer,
    Card_PlayerSerializer,
    Card_CommunitySerializer
)

from rest_framework.mixins import (
    CreateModelMixin, ListModelMixin, RetrieveModelMixin, UpdateModelMixin
)
from rest_framework import routers, serializers, viewsets, status, authentication, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Game
from django.shortcuts import render
from django.http import HttpResponseNotAllowed, JsonResponse
from .models import (
    Game,
    Player,
    Card,
    Card_Player,
    Card_Community
)

# from DQN import create_np_array, agent
from monte_carlo import get_action_policy, make_epsilon_greedy_policy
import utilities


import gym
from treys import Card
from rest_framework.decorators import action
from django.utils.decorators import method_decorator


epsilon = 0.9


class table_view(View):

    def get(self, request):
        self.env = gym.make('TexasHoldem-v1') # holdem.TexasHoldemEnv(2)
        self.env.add_player(0, stack=2000) # add a player to seat 0 with 2000 "chips"
        # env.add_player(1, stack=2000) # tight
        self.env.add_player(2, stack=2000) # aggressive

        # s.configure('call.TButton', foreground='green', bg='#c1bb78')
        # s.configure('raise.TButton', foreground='red', bg='#c1bb78')
        # s.configure('fold.TButton', foreground='blue', bg='#c1bb78')

        
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.policy = make_epsilon_greedy_policy(self.Q, self.env.action_space.n, epsilon)
        self.guest_cards = []
        self.learner_cards = []
        self.state = None
        self.player_states, self.community_infos, self.community_cards = None, None, None
        self.player_infos, player_hands = None, None
        self.current_state = None
        self.state_set = None
        self.p1 = self.env._player_dict[0] # Learner
        self.p2 = self.env._player_dict[2] # Guest
        self.episode_list = []
        self.total_pot_label = None
        self._round = None
        self.current_player = None
        self.guest_action = None
        self.call_button = None
        self.raise_button = None
        self.fold_button = None
        # self.guest_buttons = [call_button, raise_button, fold_button]
        self.terminal = False
        self.guest_label = None
        self.learner_label = None
        self.p1_pos = None
        self.p2_pos = None
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
        # part_init = None
        self.start()

        return render(request, "poker/poker.html")


    def start(self):
        self.delegate_state_info(reset=True)
        self.simulate(initial=True)

    def restart_game(self):
        self.is_new_game = True
        self.guest_action = None
        if self.terminal and self.info == 'Compete':
            if self.is_showdown():
                self.sd_tr = True
                self.last_learner_cards = "Last Cards:\n" + str(self.learner_cards_st[0]+"\n"+self.learner_cards_st[1])
        self.is_end_game = utilities.do_necessary_env_cleanup(self.env)
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
        # self.assign_player_objects_to_display(reset=True)
        
        # self.assign_cards_to_display(self.guest_cards_st, self.learner_cards_st, reset=True)
        
        # self.assign_guest_buttons()
        print("UPDATE DISPLAY")
        self.update_pot_size()

        self.print_last_action()

    def is_showdown(self):
        # If both players are still left in the game
        if self.env.level_raises[self.p1.get_seat()] == self.env.level_raises[self.p2.get_seat()]:
            if self.env.winning_players == self.p1:
                return True
        else:
            return False

    def print_last_action(self, spec=None):
        print("PRINT LAST ACTION")
        self.last_bet_label = "Learner action:\n" + str(self.env._last_actions[0])	

    def update_local_state(self, reset=True):
        print("UPDATE LOCAL STATE")
        self.p1_pos = 'SB' if self.p1.position == 0 else 'BB' # Learner
        self.p2_pos = 'SB' if self.p2.position == 0 else 'BB' # Guest
        self.mrp = self.current_player
        if reset:
            self.state = self.env.reset()
            self.set_info_before_loop()
            
        else:
            self.state = self.env._get_current_state()
            self.set_info_before_loop()
            # self.update_display()

    def delegate_state_info(self, reset):
        self.update_local_state(reset=reset)
        
        # self.assign_player_objects_to_display(reset=reset)

        self.guest_cards_st = [Card.int_to_str(self.p2.hand[0]).upper(), Card.int_to_str(self.p2.hand[1]).upper()]
        self.learner_cards_st = [Card.int_to_str(self.p1.hand[0]).upper(), Card.int_to_str(self.p1.hand[1]).upper()]

        # self.assign_cards_to_display(self.guest_cards_st, self.learner_cards_st, reset=reset)

        self.update_pot_size()

    def set_info_before_loop(self):
        # (player_states, (community_infos, community_cards)) = self.env.reset()
        (self.player_states, (self.community_infos, self.community_cards)) = self.state
        (self.player_infos, self.player_hands) = zip(*self.player_states)
        self.current_state = ((self.player_infos, self.player_hands), (self.community_infos, self.community_cards))
        utilities.compress_bucket(self.current_state, self.env, pre=True)
        
        # IF DQN 
        self.state = create_np_array(self.player_infos, self.player_hands, self.community_cards, self.community_infos)
        
        self.state_set = utilities.convert_list_to_tupleA(self.player_states[self.env.learner_bot.get_seat()], self.current_state[1])

    def update_pot_size(self):
        self.total_pot_label = str(self.env._totalpot)

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
        
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # if learner and self.info == "Compete":
        #     card_image = Image.open(dir_path[0:-4]+"/JPEG/Red_back.jpg")
        # else:
        #     card_image = Image.open(dir_path[0:-4]+"/JPEG/"+ card +".jpg")
        # photo = ImageTk.PhotoImage(card_image)
        # label = tk.Label(self, image=photo, bg='#218c16') if community is False else tk.Label(self.separator, image=photo, bg='#218c16')
        # label.image = photo # keep a reference!
        # return label

        return "EXAMPLE IMAGE"

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
        player_actions = holdem.safe_actions(self.community_infos[-1], self.community_infos, action, n_seats=self.env.n_seats, choice=None, player_o = self.p2)
        return player_actions
        

    # v = mc_prediction_poker(10)
    # # for line_no, line in enumerate(v.items()):
    # #     print(line_no, line)

    # plotting.plot_value_function(v, title="10 Steps")

    def populate_info_pre_action(self):
        self._round = utilities.which_round(self.community_cards)
        self.current_player = self.community_infos[-3]


    def get_action_for_page(self):
        # IF DQN
        if self.current_player == 0: # Learner (RHS Screen)
            self.action = agent.act(self.state, self.player_infos, self.community_infos, self.community_cards, self.env, self._round, self.env.n_seats, self.state_set, self.policy)
            if self.initial_action and self.action[self.current_player] == [3, 0]:
                self.action[self.current_player] = [1, 0]
        elif self.current_player == 2: # Player on page (LHS Screen)
            self.action = self.get_guest_action()
        
        self.initial_action = False

    def generate_episode_guest(self):
        part_ep = []
        
        self.get_action_for_page()
        
        (self.player_states, (self.community_infos, community_cards)), self.action, rewards, self.terminal, info = self.env.step(self.action)

        utilities.compress_bucket(self.player_states, self.env)
        parsed_return_state = utilities.convert_step_return_to_set((self.current_state, self.action, self.env.learner_bot.reward))
        self.action = utilities.convert_step_return_to_action(self.action)
        ps = list(zip(*self.player_states))
        next_state = create_np_array(ps[0], ps[1], self.community_cards, self.community_infos) # Numpy array
        agent.remember(self.state, self.action, self.env.learner_bot.reward, next_state, self.terminal)
        self.state = next_state
        part_ep.append((parsed_return_state, self.action, self.env.learner_bot.reward))
        current_state = (self.player_states, (self.community_infos, self.community_cards)) # state = next_state
        
        return part_ep

    def generate_episode_learner_move(self):
        episode = []
        
        self.get_action_for_page()
        
        (self.player_states, (self.community_infos, community_cards)), self.action, rewards, self.terminal, info = self.env.step(self.action)

        parsed_return_state = utilities.convert_step_return_to_set((self.current_state, self.action, self.env.learner_bot.reward))
        self.action = utilities.convert_step_return_to_action(self.action)
        episode.append((parsed_return_state, self.action, self.env.learner_bot.reward))
        current_state = (self.player_states, (self.community_infos, self.community_cards)) # state = next_state
        
        return episode

    # return render(request, template_name="poker/poker.html"):
    #     pass

# class GameViewSet(GenericViewSet,  # generic view functionality
#                      CreateModelMixin,  # handles POSTs
#                      RetrieveModelMixin,  # handles GETs for 1 Game
#                      UpdateModelMixin,  # handles PUTs and PATCHes
#                      ListModelMixin):  # handles GETs for many Companies

#     serializer_class = GameSerializer
#     queryset = Game.objects.all()



class GameViewSet(viewsets.ViewSet):  

    # authentication_classes = [authentication.TokenAuthentication]
    # permission_classes = [permissions.IsAdminUser]

    def list(self, request, format=None):
        queryset = Game.objects.all()
        serializer = GameSerializer(queryset, context={'request': request}, many=True)
        return Response(serializer.data)
    
    def create(self, request, format=None):
        serializer = GameSerializer(data=request.data, context={'request': request})
        if serializer.is_valid(): 
            serializer.save() 
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def update(self, request, pk=None):
        queryset = Game.objects.all()
        game = get_object_or_404(queryset, pk=pk)
        serializer = GameSerializer(game, data=request.data, context={'request': request})
        if serializer.is_valid(): 
            serializer.save() 
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def retrieve(self, request, pk=None):
        queryset = Game.objects.all()
        game = get_object_or_404(queryset, pk=pk)
        serializer = GameSerializer(game, context={'request': request})
        return Response(serializer.data)


    
class PlayerViewSet(viewsets.ViewSet):
    
    def list(self, request):
        queryset = Player.objects.all()
        serializer = PlayerSerializer(queryset, context={'request': request}, many=True)
        return Response(serializer.data)

    def create(self, request, format=None):
        serializer = PlayerSerializer(data=request.data, context={'request': request})
        if serializer.is_valid(): 
            serializer.save() 
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None):
        queryset = Player.objects.all()
        player = get_object_or_404(queryset, pk=pk)
        serializer = PlayerSerializer(player, data=request.data, context={'request': request})
        if serializer.is_valid(): 
            serializer.save() 
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def retrieve(self, request, pk=None):
        queryset = Player.objects.all()
        player = get_object_or_404(queryset, pk=pk)
        serializer = PlayerSerializer(player, context={'request': request})
        return Response(serializer.data)

    @action(detail=True)
    def display_player_cards(self, request, pk=None):
        """
        Returns the cards pertaining to the player 
        """

        player = self.get_object()
        cards = player.card_player_set.all()
        return Response([card.card_str for card in cards])

class CommunityCardViewSet(viewsets.ViewSet):

    def list(self, request):
        queryset = Card_Community.objects.all()
        serializer = Card_CommunitySerializer(queryset, context={'request': request}, many=True)
        return Response(serializer.data)
    
    def create(self, request, format=None):
        serializer = Card_CommunitySerializer(data=request.data, context={'request': request})
        if serializer.is_valid(): 
            serializer.save() 
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def retrieve(self, request, pk=None):
        queryset = Card_Community.objects.all()
        card = get_object_or_404(queryset, pk=pk)
        serializer = Card_CommunitySerializer(card, context={'request': request})
        return Response(serializer.data)

class PlayerCardViewSet(viewsets.ViewSet):

    def list(self, request):
        queryset = Card_Player.objects.all()
        serializer = Card_PlayerSerializer(queryset, context={'request': request}, many=True)
        return Response(serializer.data)
    
    def create(self, request, format=None):
        serializer = Card_PlayerSerializer(data=request.data, context={'request': request})
        if serializer.is_valid(): 
            serializer.save() 
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def retrieve(self, request, pk=None):
        queryset = Card_Player.objects.all()
        card = get_object_or_404(queryset, pk=pk)
        serializer = Card_PlayerSerializer(card, context={'request': request})
        return Response(serializer.data)