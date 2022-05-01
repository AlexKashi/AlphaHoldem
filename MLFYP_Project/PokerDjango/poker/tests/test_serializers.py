import json
from ..models import Player
from django.test import TestCase

from ..serializers import GameSerializer
from .factories import (
    Player_Factory,
    Game_Factory,
    CardSerializer,
    Card_PlayerSerializer,
    Card_CommunitySerializer
) 
import pdb
from treys import Card, Deck

class GameSerializerTestCase(TestCase):
    def setUp(self):
        """Serializer data matches the Company object for each field."""
        self.guest = Player_Factory.create()
        self.learner = Player_Factory.create()
        self.game = Game_Factory.create(players=(self.guest, self.learner)) # object instance being created in query set form

        self.game_serializer = GameSerializer(self.game) # object being serialized into json
        for field_name in ['id', 'total_pot']:
    
            self.assertEqual(
                game_serializer.data[field_name],
                getattr(game, field_name)
            )

        deck = Deck()
        for i in range(2):
            self.guest.card_player_set.create(card_str = Card.int_to_str(deck.draw(1))) 
            self.learner.card_player_set.create(card_str = Card.int_to_str(deck.draw(1))) 
        for i in range(5):
            self.game.card_community_set.create(card_str = Card.int_to_str(deck.draw(1))) 

    # def test_cards(self):

    #     self.card_guest_serializer = Card_PlayerSerializer(card_str)
