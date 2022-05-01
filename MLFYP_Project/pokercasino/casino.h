#pragma once
#include "includes.h"
class Bot; // declare first for recursive def
class Casino {
	public:
		// game flow:
		void populateTable(); // creates vector of bots
		void shuffleDeck(); // randomly shuffles mDeck
		void dealCards(); // tell bots their starting hands 
		void getPreflopBets(); 
		void getFlopBets(); 
		void getTurnBets(); 
		void getRiverBets(); 
		void getWinners();
		void payoffs(); // pay winners
		void showdown(); // add showdown to mCurrentHand
		void tellHandSummary(); // tells hand summary to all in even or odd file.
		void fileHandSummary(); // tells hand summary to all in even or odd file.
		void printHandSummary(); // tells hand summary to all in even or odd file.
		void prepareNext(); // initialise stuff for next hand
        void attend(); // waiting a while for bots to make a decision
		// misc
		bool tableEmpty(); // check if only one player is left, ie can go to showdown
		
	private: // 
		vector<Bot*> mPlayers; // a queue of all the bots
		deque<Bot*> table; // all players still in the hand
		array<int, 52> mDeck; // shuffled deck
		string mCurrentHand; // status of current hand
		vector<int> mWinners; // winners among players left in hand
		int mCounter = 0; // current hand number
		int mButton = 0; // current button
		int mPot = 0; // current pot

		
};

