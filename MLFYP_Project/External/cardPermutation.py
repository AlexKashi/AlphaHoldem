import pandas as pd
import numpy as np


class HoldingCards():
	firstCardRank = '-'
	firstCardSuit = '-'
	secondCardRank = '-'
	secondCardSuit = '-'
	cardsString = ''

	def __init__(self, fr,fs,sr,ss):
		self.firstCard = fr + fs
		self.secondCard = sr + ss
		self.cardsString = fr+fs+sr+ss

	def __str__(self):
		return '{self.firstCard}{self.secondCard}'.format(self=self)


card = ['A', 'K','Q','J','10','9','8','7','6','5','4','3','2']
arr = ['h','c','s','d']

list_of_cards = []
for loopACard in range(len(card)):
	# if card[loopACard] == '2':
	# 	break

	for loopBCard in range(loopACard, len(card)):

		for i in range(len(arr)):
			noteFirst = 0

			if loopACard == loopBCard:
				noteFirst = i+1
				if arr[i]=='d':
					break


			for j in range(noteFirst, len(arr)):
				#print(card[loopACard], arr[i], card[loopBCard], arr[j])
				hc = HoldingCards(card[loopACard], arr[i], card[loopBCard], arr[j])
				list_of_cards.append(hc)

print("length:", len(list_of_cards))
#print([str(list_of_cards[i]) for i in range(len(list_of_cards))])

array = np.arange(3978).reshape((len(list_of_cards), 3))
list_of_possible_cards = pd.Series(array, index = list_of_cards)
