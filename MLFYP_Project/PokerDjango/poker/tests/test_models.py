from django.test import TestCase

from ..models import *
from .factories import (
    Game_Factory,
    Player_Factory,
)

from treys import Card, Deck

class GameTestCase(TestCase):
    def setUp(self):
        """Test for string representation."""
        self.guest = Player_Factory.create()
        self.assertEqual(Player.objects.get(id__exact=self.guest.id).name, self.guest.name)

        self.learner = Player_Factory.create()
        self.assertEqual(Player.objects.get(id__exact=self.learner.id).name, self.learner.name)

        self.game = Game_Factory.create(players=(self.guest, self.learner))
        self.assertEqual(str(self.game), str(self.game.id))

        self.game_players = [player.id for player in Player.objects.filter(games__id=self.game.id)]
        self.assertEqual(self.game_players, [self.guest.id, self.learner.id])

    def test_card(self):
        deck = Deck()
        for i in range(2):
            self.guest.card_player_set.create(card_str = Card.int_to_str(deck.draw(1))) 
            self.learner.card_player_set.create(card_str = Card.int_to_str(deck.draw(1))) 
        self.assertTrue([len(x.card_str)<3 for x in self.guest.card_player_set.all()])
        self.assertTrue([len(x.card_str)<3 for x in self.learner.card_player_set.all()])
    
        for i in range(5):
            self.game.card_community_set.create(card_str = Card.int_to_str(deck.draw(1))) 
        self.assertTrue([len(x.card_str)<3 for x in self.game.card_player_set.all()])