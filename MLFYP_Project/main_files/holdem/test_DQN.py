import unittest
import utilities, DQN
import numpy as np
from collections import defaultdict

class DQNTestCase(unittest.TestCase):
    def setUp(self):
        self.env = DQN.env
        (self.player_states, (self.community_infos, self.community_cards)) = self.env.reset()
        (self.player_infos, self.player_hands) = zip(*self.player_states)
        self.current_state = ((self.player_infos, self.player_hands), (self.community_infos, self.community_cards))
        self.state = DQN.create_np_array(self.player_infos, self.player_hands, self.community_cards, self.community_infos)
        self.state_set = utilities.convert_list_to_tupleA(self.player_states[self.env.learner_bot.get_seat()], self.current_state[1])
        self._round = utilities.which_round(self.community_cards)
        self.current_player = self.community_infos[-3]
        self.learner_bot, self.villain = self.env.learner_bot, self.env.villain
        Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.agent = DQN.DQNAgent(DQN.state_size, DQN.action_size) # initialise agent

        self.policy = DQN.make_epsilon_greedy_policy(Q, self.agent.epsilon, self.env.action_space.n)
        self.villain_action = DQN.get_action_policy(self.player_infos, self.community_infos, self.community_cards, self.env, self._round, self.env.n_seats, self.state_set, self.policy, self.villain)
        self.learner_action = self.agent.act(self.state, self.player_infos, self.community_infos, self.community_cards, self.env, self._round, self.env.n_seats, self.state_set, self.policy)


    def test_actions(self):
        self.assertIsNotNone(DQN.get_action_policy(self.player_infos, self.community_infos, self.community_cards, self.env, self._round, self.env.n_seats, self.state_set, self.policy, self.villain))
        self.assertIsNotNone(self.agent.act(self.state, self.player_infos, self.community_infos, self.community_cards, self.env, self._round, self.env.n_seats, self.state_set, self.policy))

    def test_get_learner_action(self):
        self.assertEqual(len(self.villain_action), 4)
        self.assertEqual(len(self.learner_action), 4)
        self.assertIsNotNone(self.env.step(self.villain_action))
        self.assertIsNotNone(self.env.step(self.learner_action))

    # def test_level_raises(self):
    #     func = self.env.is_off_balance_LR()
        
    #     self.assertLess(func(), self.env.level_raises)

    def tearDown(self):
        del self.env
        del self.villain_action 
        del self.learner_action
    
