import gym
import holdem
import numpy as np
from collections import defaultdict, deque
from include import *
import matplotlib.pyplot as plt
from libs import plotting
import sys
import utilities
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os # for creating directories


starting_stack_size = 200000

env = gym.make('TexasHoldem-v1') # holdem.TexasHoldemEnv(2)
env.add_player(0, stack=starting_stack_size) # add a player to seat 0 with 2000 "chips"
env.add_player(2, stack=starting_stack_size) # aggressive#

state_size = 18

action_size = env.action_space.n

batch_size = 32

epsilon = 0.8

n_episodes = 10000 # n games we want agent to play (default 1001)

output_dir = 'model_output/TexasHoldemDirectory/'

with_render = True

with_graph = True

villain = "Strong"

delay = None


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # double-ended queue; acts like list, but elements can be added/removed from either end
        self.gamma = 0.95 # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon = epsilon # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995 # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.001 # minimum amount of random exploration permitted
        self.learning_rate = 0.01 # rate at which NN adjusts models parameters via SGD to reduce cost 
        self.model = self._build_model() # private method 
    
    def _build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu')) # 1st hidden layer; states as input
        model.add(Dense(32, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear')) # 2 actions, so 2 output neurons: 0 and 1 (L/R)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later

    def act(self, state, player_infos, community_infos, community_cards, env, _round, n_seats, state_set, policy):
        if np.random.rand() <= self.epsilon: 
            action = get_action_policy(player_infos, community_infos, community_cards, env, _round, n_seats, state_set, policy, villain)
            return action
        act_values = self.model.predict(state) # if not acting according to safe_strategy, predict reward value based on current state
        predicted_action = np.argmax(act_values[0])
        env.learner_bot.he.set_community_cards(community_cards, _round)
        range_structure = utilities.fill_range_structure(_round, env.learner_bot)
        utilities.assign_evals_player(env.learner_bot, _round, env)
        choice = None
        if predicted_action == 0:
            choice = 1
        elif predicted_action == 1:
            total_bet = env._tocall + env._bigblind - env.villain.currentbet
            choice = (2, total_bet)
        elif predicted_action == 2:
            choice = 3
        predicted_action = holdem.safe_actions(community_infos[-1], community_infos, villain_choice=None, n_seats=n_seats, choice=choice, player_o=env.learner_bot)
        return predicted_action # pick the action that will give the highest reward (i.e., go left or right?)

    def replay(self, batch_size): # method that trains NN with experiences sampled from memory
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
            target = reward # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            if not done: # if not done, then predict future discounted reward
                target = (reward + self.gamma * # (target) = reward + (discount rate gamma) * 
                          np.amax(self.model.predict(next_state)[0])) # (maximum target Q based on future action a')
            target_f = self.model.predict(state) # approximately map current state to future discounted reward
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


file ="./model_output/TexasHoldemDirectory/weights_1000.hdf5"
agent = DQNAgent(state_size, action_size) # initialise agent



def create_np_array(player_infos, player_hands, community_cards, community_infos):
    ps1 = (player_infos[0])
    for card in player_hands[0]:
        ps1 = np.append(ps1, card)    
    for info in community_infos:
        ps1 = np.append(ps1, info)    
    for card in community_cards:
        ps1 = np.append(ps1, card)    
    ps1 = np.reshape(ps1, [1, state_size])
    return ps1


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation): # [call/check, raise/bet, fold]
        A = np.ones(nA, dtype=float) * epsilon / nA
        b = Q[observation]
        best_action = np.argmax(b)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


# ***********************************Interacting with environment ********************************



def get_action_policy(player_infos, community_infos, community_cards, env, _round, n_seats, state, policy, villain):
	
	player_actions = None
	current_player = community_infos[-3]
	
	player_object = env._player_dict[current_player]
	to_call = community_infos[-1]
	stack, hand_rank, played_this_round, betting, lastsidepot = player_infos[current_player-1] if current_player is 2 else player_infos[current_player]
	stack, hand_rank, played_this_round, betting, lastsidepot = player_infos[current_player-1] if current_player == 2 else player_infos[current_player]
	player_object.he.set_community_cards(community_cards, _round)
	
	if _round != "Preflop": # preflop already evaluated
		player_object.he.evaluate(_round)
	range_structure = utilities.fill_range_structure(_round, player_object)
	utilities.assign_evals_player(player_object, _round, env)

	if(current_player == 0): # learner move 
		probs = policy(state)
		choice = np.random.choice(np.arange(len(probs)), p=probs)
		best_nonlearning_action = player_object.choose_action(_round, range_structure, env) # Doesn't use
		player_actions = holdem.safe_actions(to_call, community_infos, villain_choice=None, n_seats=n_seats, choice=choice, player_o = player_object, best_nonlearning_action=best_nonlearning_action)
		
	else: # bot move 
		if villain == "CallChump":
			player_actions = utilities.safe_actions_call_bot(community_infos, villain_choice=None, n_seats=n_seats)
		else:
			villain_choice = player_object.choose_action(_round, range_structure, env) 
			player_actions = holdem.safe_actions(to_call, community_infos, villain_choice, n_seats=n_seats, choice=None, player_o = player_object)
	
	return player_actions

if __name__ == "__main__":

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
        
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, agent.epsilon, env.action_space.n)
    last_episode = None
    episode_list = []
    stacks_over_time = {}
    for index, player in env._player_dict.items():
        stacks_over_time.update({player.get_seat(): [player.stack]})
    for e in range(n_episodes): # iterate over new episodes of the game    # Print out which episode we're on, useful for debugging.
        if e % 50 == 0:
            agent.load(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
        last_episode = e + 1
        if with_render:
            print("\n\n********Episode {}*********".format(e)) 
        episode = []
        (player_states, (community_infos, community_cards)) = env.reset()
        (player_infos, player_hands) = zip(*player_states)
        current_state = ((player_infos, player_hands), (community_infos, community_cards))
        utilities.compress_bucket(current_state, env, pre=True)
        state = create_np_array(player_infos, player_hands, community_cards, community_infos)

        # Only want the state set that is relevant to learner bot every step. 
        state_set = utilities.convert_list_to_tupleA(player_states[env.learner_bot.get_seat()], current_state[1])

        if with_render:
            env.render(mode='human', initial=True, delay=delay)
        terminal = False
        while not terminal:

            _round = utilities.which_round(community_cards)
            current_player = community_infos[-3]
            if current_player != 0:
                action = get_action_policy(player_infos, community_infos, community_cards, env, _round, env.n_seats, state_set, policy, villain)
            else:
                action = agent.act(state, player_infos, community_infos, community_cards, env, _round, env.n_seats, state_set, policy)
            
            #STEP - SET BREAKPOINT ON THE FOLLOWING LINE TO OBSERVE ACTIONS TAKEN ONE BY ONE
            (player_states, (community_infos, community_cards)), action, rewards, terminal, info = env.step(action)

            utilities.compress_bucket(player_states, env)
            action = utilities.convert_step_return_to_action(action)
            ps = list(zip(*player_states))
            next_state = create_np_array(ps[0], ps[1], community_cards, community_infos) # Numpy array
            agent.remember(state, action, env.learner_bot.reward, next_state, terminal)
            state = next_state
            if terminal: 
                print("episode: {}/{}, reward: {}, e: {:.2}, Profit Margin {}" # print the episode's score and agent's epsilon
                    .format(e, n_episodes, env.learner_bot.reward, agent.epsilon, env.learner_bot.stack - starting_stack_size))
            
            current_state = (player_states, (community_infos, community_cards)) # state = next_state
            if with_render:
                env.render(mode='human', delay=delay)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size) # train the agent by replaying the experiences of the episode
            if e % 50 == 0:
                agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")

        utilities.do_necessary_env_cleanup(env) # assign new positions, remove players if stack < 0 etc ..
        if len(env._player_dict) > 1:
            count_players = len(env._player_dict)
            sum_stack = 0
            for param in env._player_dict:
                sum_stack += env._player_dict[param].stack

            if sum_stack != count_players * starting_stack_size:
                raise("Stacks should add to equal"+str(count_players * starting_stack_size))
        stack_list = env.report_game(requested_attributes = ["stack"])
        count_existing_players = 0
        for stack_record_index, stack_record in env._player_dict.items():
            arr = stacks_over_time[stack_record_index] + [stack_list[stack_record_index]]
            stacks_over_time.update({stack_record_index: arr})
            if(stack_list[stack_record_index] != 0):
                count_existing_players += 1
        episode_list.append(episode)

        if(count_existing_players == 1):
            
            break

    # Episode end
    for player_idx, stack in stacks_over_time.items():
        if player_idx == 0:
            plt.plot(stack, label = "Player {} - Learner".format(player_idx))
        else:	
            plt.plot(stack, label = "Player {}".format(player_idx))
    p1_stack_t = list(stacks_over_time.values())[0]
    p2_stack_t = list(stacks_over_time.values())[1]
    # diffs = [j-i for i, j in zip(p1_stack_t[:-1], p1_stack_t[1:])]
    # import statistics
    # lost_avg = statistics.mean(diffs)
    won_avg = p1_stack_t[len(p1_stack_t)-1] - p1_stack_t[0]
    # print(p1_stack_t)
    print('mbb/g:{}'.format(1000 * won_avg/(env._bigblind*last_episode)))
    plt.ylabel('Stack Size')
    plt.xlabel('Episode')
    plt.legend()
    if with_graph:
        plt.show()
