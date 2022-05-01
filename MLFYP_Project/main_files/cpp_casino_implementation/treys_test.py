from treys import Card 

print("BEFORE")
Card.print_pretty_cards([268446761, 134236965, 33589533] + [67115551, 16787479])
print("AFTER")

from treys import Deck
deck = Deck()
board = deck.draw(5)
player1_hand = deck.draw(2)
player2_hand = deck.draw(2)

#Card.int_to_pretty_str(268446761)

int_to_pretty_str(268446761)


def int_to_pretty_str(card_int):
        """
        Prints a single card 
        """
        
        color = False
        try:
            from termcolor import colored
            # for mac, linux: http://pypi.python.org/pypi/termcolor
            # can use for windows: http://pypi.python.org/pypi/colorama
            color = True
        except ImportError: 
            pass

        # suit and rank
        suit_int = Card.get_suit_int(card_int)
        rank_int = Card.get_rank_int(card_int)

        # if we need to color red
        s = Card.PRETTY_SUITS[suit_int]
        if color and suit_int in Card.PRETTY_REDS:
            s = colored(s, "red")

        r = Card.STR_RANKS[rank_int]

	return "[{}{}]".format(r,s)




