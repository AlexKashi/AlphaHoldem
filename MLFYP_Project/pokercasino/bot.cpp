#include "includes.h"
//include "casino.h"
#include "bot.h"

void Bot::setSeat(int pos){
	mSeat = pos;
}

void Bot::setHand(int handno, int button, tuple<int, int> hand) {
	mHand = hand;
	ofstream fout("./botfiles/give_hand_bot" + to_string(getSeat()), ios_base::trunc);
	if (!fout.good()) {
		cerr << "Error while opening output file for bot " << getSeat() << endl;
	}
	fout << handno << "D" << button << "A" << get<0>(hand) << "B" << get<1>(hand); // coded: hand number A card1 B card2
}

tuple<int, int> Bot::getHand() {
	return mHand;
}



void Bot::addStack(int ammount) {
	mStack += ammount;
	}

	
int Bot::getSeat(){return mSeat;}

