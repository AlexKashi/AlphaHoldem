from rest_framework.serializers import ModelSerializer, StringRelatedField, HyperlinkedRelatedField

from .models import (
    Game,
    Player,
    Card,
    Card_Player,
    Card_Community
)

class RelatedPlayerSerializer(ModelSerializer):
    
    class Meta: 
        model = Player
        fields = ['name', 'url']

class RelatedGameSerializer(ModelSerializer):
    
    class Meta: 
        model = Game
        fields = ['url']

class RelatedCommunityCardSerializer(ModelSerializer):
    
    class Meta: 
        model = Card_Community
        fields = ['card_str', 'url']
    
class RelatedPlayerCardSerializer(ModelSerializer):
    
    class Meta: 
        model = Card_Player
        fields = ['card_str', 'url']


class PlayerSerializer(ModelSerializer):
    
    games = RelatedGameSerializer(many=True, read_only=True)
    cards = StringRelatedField(many=True, read_only=True)

    def create(self, validated_data):
        return Player.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.name = validated_data.get('name', None)
        instance.stack = validated_data.get('stack', None)
        # self.list_save(validated_data, 'games').save()
        # self.list_save(validated_data, 'cards').save()
        instance.save()
        return instance 

    # def list_save(self, validated_data, field):
    #     li = validated_data.pop(field, None)
    #     if li is not None:
    #         for el in li:
    #             self.instance.field.add(el)
        
    #     return self.instance

    class Meta:
        model = Player
        fields = ['id', 'name', 'stack', 'games', 'cards']
        extra_kwargs = {'games': {'required': False}, 'cards': {'required': False}  }

class GameSerializer(ModelSerializer):
    players = RelatedPlayerSerializer(many=True, read_only=True)
    community_cards = StringRelatedField(many=True, read_only=True)

    def create(self, validated_data):
        return Game.objects.create(**validated_data)
        
    def update(self, instance, validated_data):
        instance.total_pot = validated_data.get('total_pot', None)
        # self.list_save(validated_data, 'players').save()
        # self.list_save(validated_data, 'community_cards').save()
        instance.save()
        return instance 

    # def list_save(self, validated_data, field):
    #     li = validated_data.pop(field, None)
    #     if li is not None:
    #         for el in li:
    #             self.instance.field.add(el)
        
    #     return self.instance

    class Meta:
        model = Game
        fields = ['id', 'total_pot', 'players', 'community_cards', 'created_at']
        extra_kwargs = {'players': {'required': False}, 'community_cards': {'required': False} }

class CardSerializer(ModelSerializer):
    class Meta:
        model = Card
        abstract = True

class Card_CommunitySerializer(CardSerializer):
    game = RelatedGameSerializer(read_only=True)
    
    def create(self, validated_data):
        return Card_Community.objects.create(**validated_data)
        
    def update(self, instance, validated_data):
        instance.card_str = validated_data.get('card_str', None)
        instance.game = validated_data.get('game', None)
        instance.save()
        return instance 

    class Meta:
        model = Card_Community
        fields = ['id', 'card_str', 'game']

class Card_PlayerSerializer(CardSerializer):
    player = RelatedPlayerSerializer(read_only=True)
    game = RelatedGameSerializer(read_only=True)

    def create(self, validated_data):
        return Card_Player.objects.create(**validated_data)
        
    def update(self, instance, validated_data):
        instance.card_str = validated_data.get('card_str', None)
        instance.player = validated_data.get('player', None)
        instance.game = validated_data.get('game', None)
        instance.save()
        return instance 

    class Meta:
        model = Card_Player
        fields = ['id', 'card_str', 'player', 'game']