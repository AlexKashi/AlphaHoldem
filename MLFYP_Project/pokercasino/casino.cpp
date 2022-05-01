#include "includes.h"
#include "casino.h"
#include "bot.h"
#include <thread>
#include <iostream>
//#include <WS2tcpip.h>

// Include the Winsock library (lib) file
//#pragma comment (lib, "ws2_32.lib")


//
// casino methods:
//
void Casino::shuffleDeck() { // randomly shuffles input deck todo only go to 2* players + 5
	array<int, 52> orderedDeck; // contains all cards from 0 to 51
	for (int i = 0; i < 52; i++)
		orderedDeck[i] = i;
	for (int i = 0; i < 52; i++) {
		int k = rand() % (52 - i); // next card drawn
		int counter = 0;
		for (int j = 0; j < 52; j++) { // we fetch k-th undrawn card, i.e. among those not = 99
			if (orderedDeck[j] != 99) // 99 codes for card already drawn
				counter++;
			if (counter == k + 1) {
			       orderedDeck[j] = 99;
			       mDeck[i] = j;
		       		break;
			}

		}
	}

}

void Casino::populateTable(){
	for (int i = 0; i < nBots; i++) { // creates a vector of nBots smart pointers to Bot objects.
		Bot* abot = new Bot(); //("socket" + to_string(i)); // smart pointer to a bot object
		abot->setSeat(i);
		mPlayers.push_back(abot); // push it on mPlayers
	}
}


void Casino::attend() { // time given to bots for each decision
	std::this_thread::sleep_for(kDelay);
}


void Casino::dealCards() {
	for (int i = 0; i < nBots; i++) {
		mPlayers[i]->setHand(mCounter, mButton, make_tuple(mDeck[2 * i], mDeck[2 * i + 1]));
	}
}


void Casino::tellHandSummary() {
	if (mCounter % 2 == 0) {
		ofstream fout("./botfiles/handSummaryEven", ios_base::trunc);
		if (!fout.good()) {
			cerr << "Error opening handSummaryEven file " << endl;
		}
		fout << mCurrentHand;
	}
	else {
		ofstream fout("./botfiles/handSummaryOdd", ios_base::trunc);
		if (!fout.good()) {
			cerr << "Error opening handSummaryOdd file " << endl;
		}
		fout << mCurrentHand;

	}
}



void Casino::showdown(){
	for (auto winner : table) { // print the cards of all players still in the hand at showdown.
		mCurrentHand += "S" + to_string(winner->getSeat()) + "A" + to_string(mDeck[2 * winner->getSeat()]) + "B" + to_string(mDeck[2 * winner->getSeat() + 1]) ;
	}
	for (auto v : mWinners) { // print the hands of all winners
		mCurrentHand += "W" + to_string(v);
	}
	mCurrentHand += 'E'; // end of hand

}



bool Casino::tableEmpty(){
	return table.size() == 1;
}

void Casino::prepareNext() {
	mCounter++;
	mButton = (mButton + 1) % nBots;
}

void Bot::tellAction(string handStatus) {
	ofstream fout("./botfiles/casinoToBot" + to_string(getSeat()), ios_base::trunc);
	if (!fout.good()) {
		cerr << "Error while opening output file for bot " << getSeat() << endl;
	}
	fout << handStatus;
}
void Casino::getPreflopBets() {
	table.clear();
	vector<int> vpip(nBots, 0); // track record of how many bets each bot already called
	for (int i = 0;  i < nBots; i++){
		table.push_back(mPlayers[(mButton + 3 + i) % nBots]); // queue of players, first player is first to act after sb
	}
	mPlayers[(mButton + 1) % nBots]->addStack(-1); // collect small blind
	vpip[(mButton + 1) % nBots]++; // he paid 1 bet
	mPlayers[(mButton + 2) % nBots]->addStack(-1); // big blind simplification: two players post a 1$ blind (no small or big) todo change
	vpip[(mButton + 2) % nBots]++; // big blind simplification: two players post a 1$ blind (no small or big) todo change
	mPot = 2; // two blinds
	int calls = 0;
	int raises = 1; // count blinds as one raise
	mCurrentHand = to_string(mCounter) + "D" + to_string(mButton) + "P"; // current hand is coded hand number + D + button position + "P"
	while (calls < nBots && table.size() > 1){
		Bot* currentPlayer = table.front(); // get first element
		table.pop_front(); // remove him
		currentPlayer->tellAction(mCurrentHand); // give him hand status
		attend(); // let him process it
		switch(char action = currentPlayer->getAction()) {
			case 'e':
				goto foldCase;
				break;
			case 'c':
                callCase:
                    calls++;
                    table.push_back(currentPlayer); // player stays on
                    currentPlayer->addStack(vpip[currentPlayer->getSeat()] - raises);
                    mPot += raises - vpip[currentPlayer->getSeat()];
                    vpip[currentPlayer->getSeat()] = raises; // paid all so far
                    mCurrentHand += 'c'; // current player called
                    break;
			case 'f':
				foldCase:
					calls++;
					mCurrentHand += 'f';
					break;
			case 'r':
				if (raises > nMaxRaises){
					goto callCase; // the player is called automatically
				}
				else { // valid raise
					raises++;
					currentPlayer->addStack(vpip[currentPlayer->getSeat()] - raises);
					mPot += raises - vpip[currentPlayer->getSeat()];
					vpip[currentPlayer->getSeat()] = raises;
					table.push_back(currentPlayer); // player stays on
					mCurrentHand += 'r';
					calls = nBots - table.size() + 1; // this bet round stops once all remaining players minus the raiser have called this latest raise (unless there's new raise)
					break;
				}
		}
	}
}

char Bot::getAction(){
	ifstream fin("./botfiles/botToCasino" + to_string(getSeat()));
	if (!fin) {
		cerr << "error opening file botToCasino" << getSeat()  << endl;
		return 'e';
	}
	else {
		char action;
		fin >> action;
		if (action == 'c' || action == 'f' || action == 'r')
			return action;
		else return 'e';
	}
}





void Casino::getFlopBets() {
	for (int j = 0; j < nBots; j++){// get next to act front of queue
		for (int i = 0; i < table.size(); i++){
			Bot* frontPlayer = table.front();
			if (frontPlayer->getSeat() == ((mButton + j + 1) % nBots))
				goto tableOrdered;
			else {
				table.pop_front();
				table.push_back(frontPlayer);
			}
		}
	}
	tableOrdered:
	vector<int> vpip(nBots, 0); // track record of how many bets each bot already called
	int calls = 0;
	int raises = 0;
	mCurrentHand += "F"; // current hand is coded hand number + "F"
	for (int i = 0; i < 3; i++){
		mCurrentHand += to_string(mDeck[2 * nBots + i]) + "F";
	}
	while (calls < nBots && table.size() > 1){
		Bot* currentPlayer = table.front(); // get first element
		table.pop_front(); // remove him
		currentPlayer->tellAction(mCurrentHand); // give him hand status
		attend(); // let him process it
		switch(char action = currentPlayer->getAction()) {
			case 'c':
                callCase:
                    calls++;
                    table.push_back(currentPlayer); // player stays on
                    currentPlayer->addStack(vpip[currentPlayer->getSeat()] - raises);
                    mPot += raises - vpip[currentPlayer->getSeat()];
                    vpip[currentPlayer->getSeat()] = raises; // paid all so far
                    mCurrentHand += 'c'; // current player called
                    break;
			case 'f':
				foldCase:
					calls++;
					mCurrentHand += 'f';
					break;
			case 'r':
				if (raises >  nMaxRaises){
					goto callCase; // the player is called automatically
				}
				else { // valid raise
					raises++;
					currentPlayer->addStack(vpip[currentPlayer->getSeat()] - raises);
					mPot += raises - vpip[currentPlayer->getSeat()];
					vpip[currentPlayer->getSeat()] = raises;
					table.push_back(currentPlayer); // player stays on
					mCurrentHand += 'r';
					calls = nBots - table.size() + 1; // this bet round stops once all remaining players minus the raiser have called this latest raise (unless there's new raise)
					break;
				}
			case 'e':
				goto foldCase;
				break;
		}
	}
}


void Casino::getTurnBets() {
	for (int j = 0; j < nBots; j++){
		for (int i = 0; i < table.size(); i++){ // get next to act front of queue
			Bot* frontPlayer = table.front();
			if (frontPlayer->getSeat() == ((mButton + j + 1) % nBots))
				goto tableOrdered;
			else {
				table.pop_front();
				table.push_back(frontPlayer);
			}
		}
	}
	tableOrdered:
	vector<int> vpip(nBots, 0); // track record of how many bets each bot already called
	int calls = 0;
	int raises = 0;
	mCurrentHand += "T"; // current hand is coded hand number + "T"
	mCurrentHand += to_string(mDeck[2 * nBots + 3]) + "T";
	while (calls < nBots && table.size() > 1){
		Bot* currentPlayer = table.front(); // get first element
		table.pop_front(); // remove him
		currentPlayer->tellAction(mCurrentHand); // give him hand status
		attend(); // let him process it
		switch(char action = currentPlayer->getAction()) {
			case 'e':
				goto foldCase;
				break;
			case 'c':
callCase:
				calls++;
				table.push_back(currentPlayer); // player stays on
				currentPlayer->addStack(2 * (vpip[currentPlayer->getSeat()] - raises));
				mPot += 2 * (raises - vpip[currentPlayer->getSeat()]); // bets double on turn
				vpip[currentPlayer->getSeat()] = raises; // paid all so far
				mCurrentHand += 'c'; // current player called
				break;
			case 'f':
				foldCase:
					calls++;
					mCurrentHand += 'f';
					break;
			case 'r':
				if (raises > nMaxRaises){
					goto callCase; // the player is called automatically
				}
				else { // valid raise
					raises++;
					currentPlayer->addStack(2 * (vpip[currentPlayer->getSeat()] - raises));
					mPot += 2 * (raises - vpip[currentPlayer->getSeat()]);
					vpip[currentPlayer->getSeat()] = raises;
					table.push_back(currentPlayer); // player stays on
					mCurrentHand += 'r';
					calls = nBots - table.size() + 1; // this bet round stops once all remaining players minus the raiser have called this latest raise (unless there's new raise)
					break;
				}
		}
	}
}


void Casino::getRiverBets() {
	for (int j = 0; j < nBots; j++){
		for (int i = 0; i < table.size(); i++){ // get next to act front of queue
			Bot* frontPlayer = table.front();
			if (frontPlayer->getSeat() == ((mButton + j + 1) % nBots))
				goto tableOrdered;
			else {
				table.pop_front();
				table.push_back(frontPlayer);
			}
		}
	}
	tableOrdered:
	vector<int> vpip(nBots, 0); // track record of how many bets each bot already called
	int calls = 0;
	int raises = 0;
	mCurrentHand += "R"; // current hand is coded hand number + "R"
	mCurrentHand += to_string(mDeck[2 * nBots + 4]) + "R";
	while (calls < nBots && table.size() > 1){
		Bot* currentPlayer = table.front(); // get first element
		table.pop_front(); // remove him
		currentPlayer->tellAction(mCurrentHand); // give him hand status
		attend(); // let him process it
		switch(char action = currentPlayer->getAction()) {
			case 'e':
				goto foldCase;
				break;
			case 'c':
callCase:
				calls++;
				table.push_back(currentPlayer); // player stays on
				currentPlayer->addStack(2 * (vpip[currentPlayer->getSeat()] - raises));
				mPot += 2 * (raises - vpip[currentPlayer->getSeat()]); // bets double on turn
				vpip[currentPlayer->getSeat()] = raises; // paid all so far
				mCurrentHand += 'c'; // current player called
				break;
			case 'f':
				foldCase:
					calls++;
					mCurrentHand += 'f';
					break;
			case 'r':
				if (raises > nMaxRaises){
					goto callCase; // the player is called automatically
				}
				else { // valid raise
					raises++;
					currentPlayer->addStack(2 * (vpip[currentPlayer->getSeat()] - raises));
					mPot += 2* (raises - vpip[currentPlayer->getSeat()]);
					vpip[currentPlayer->getSeat()] = raises;
					table.push_back(currentPlayer); // player stays on
					mCurrentHand += 'r';
					calls = nBots - table.size() + 1; // this bet round stops once all remaining players minus the raiser have called this latest raise (unless there's new raise)
					break;
				}
		}
	}
}

void Casino::getWinners() {
	mWinners.clear();
	if (table.size() == 1){
		mWinners.push_back(table.front()->getSeat());
	}
	else {
		using namespace omp;
		HandEvaluator eval;
		int bestScore = 0;
		for (auto v : table) {
			Hand h = Hand::empty();
			h += Hand(mDeck[2 * nBots]) + Hand(mDeck[2 * nBots + 1]) + Hand(mDeck[2 * nBots + 2]) + Hand(mDeck[2 * nBots + 3]) + Hand(mDeck[2 * nBots + 4]) + Hand(get<0>(v->getHand())) + Hand(get<1>(v->getHand()));
			int score = eval.evaluate(h);
			if (score == bestScore) { // tie
				mWinners.push_back(v->getSeat());
			}
			else if (score > bestScore) { // new winner
				bestScore = score; // new best hand
				mWinners.clear(); // get rid of all false previous winners
				mWinners.push_back(v->getSeat());
			}
		}
	}
}

void Casino::payoffs() { // pay the winners
	for (auto w : mWinners){
		mPlayers[w]->addStack(mPot / mWinners.size());
	}
}


void Casino::fileHandSummary() {
	if (mCounter % nLogFrequency == 0) {
        for (auto v : mPlayers){
            ofstream fout("./botfiles/stack" + to_string(v->getSeat()), ios_base::app);
            if (!fout.good()) {
                cerr << "Error while opening stack file number " + to_string(v->getSeat())  << endl;
            }
            else {
                fout << v->mStack << " ";
            }
        }
    }
}

void Casino::printHandSummary() {
	cout << endl << "hand number: " << mCounter << " button: " << mButton << " final pot: " << mPot << " winner seats: ";
	       for (auto p : mWinners)
	       		cout << p << " ";
	cout << endl;
	for (auto v : mPlayers){
		cout << "seat: " << v->mSeat << "  Hand: " << eRank[get<0>(v->mHand)] << " " << eRank[get<1>(v->mHand)] << "  Stack: " << v->mStack  << endl;
	}
	cout << " board: ";
	for (int i = 0; i  < 5; i++) {
		cout << eRank[mDeck[2 * nBots + i]] << " ";
	}
	cout << endl << "hand history :" << endl;

	// for (auto v : mWinners) { // print the hands of all winners
	// 	string mCurrentH = mCurrentHand + to_string(v);
	// 	for (auto player : mPlayers){
	// 		if (player->getSeat() == v){
	// 			player->tellAction(mCurrentH);
	// 		}
	// 	}
	// }

	cout << mCurrentHand << endl;
}
