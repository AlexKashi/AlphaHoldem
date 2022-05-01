import pdb
import factory
from faker import Faker
from ..models import (
    Game,
    Player,
    Card,
    Card_Player,
    Card_Community
)
from djmoney.models.fields import MoneyField
from djmoney.money import Money
import random
import pdb
from decimal import Decimal
from treys import Deck, Card
faker = Faker()

# class Card_Factory(factory.DjangoModelFactory):
#     class Meta:
#         model = Card
#         abstract = True
    
#     card_str = Card.int_to_str(Deck.draw(1))
    
# class Card_Player_Factory(Card_Factory):

#     class Meta:
#         model = Card_Player

#     player = factory.SubFactory(Player)

# class Card_Community_Factory(Card_Factory):
#     class Meta:
#         model = Card_Community

#     game = factory.SubFactory(Game)

class Player_Factory(factory.DjangoModelFactory):
    class Meta:
        model = Player

    @factory.lazy_attribute
    def name(self):
        return factory.Faker('name').generate({}) 
        
    @factory.lazy_attribute
    def stack(self):
        # pdb.set_trace()

        return "{:.2f}".format(factory.Faker('random_number').generate({}))


class Game_Factory(factory.DjangoModelFactory):
    class Meta:
        model = Game
    # pdb.set_trace()
    total_pot = "{:.2f}".format(factory.Faker('random_number').generate({}))
    
    @factory.post_generation
    def players(self, create, extracted, **kwargs):
        if not create: 
            return
        
        if extracted: 
            for player in extracted:
                self.players.add(player)