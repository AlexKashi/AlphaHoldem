class Casino:
    def __init__(self):
        pass
    def shuffle_deck(self):
        # Python program to shuffle a deck of card using the module random and draw 5 cards

        # import modules
        import itertools, random

        # make a deck of cards
        deck = list(itertools.product(range(1,14),['Spade','Heart','Diamond','Club']))

        # shuffle the cards
        random.shuffle(deck)

        print(deck[0][0], deck[0][1])
    
if __name__ == '__main__':
    cas = Casino()
    cas.shuffle_deck()