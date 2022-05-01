#include "includes.h"
#include "casino.h"
#include "bot.h"


int main()
{
	// initialise rand, todo use mersenne for better randomness
	std::srand(std::time(nullptr)); 
	Casino lasVegas;
	lasVegas.populateTable();
			
	for (int r = 0; r < nRounds; r++){
		lasVegas.shuffleDeck();
		lasVegas.dealCards();
		lasVegas.getPreflopBets();
		if (!lasVegas.tableEmpty())
			lasVegas.getFlopBets();
		if (!lasVegas.tableEmpty())
			lasVegas.getTurnBets();
		if (!lasVegas.tableEmpty())
			lasVegas.getRiverBets();
		lasVegas.getWinners(); // get all winners
		lasVegas.payoffs(); // pay winner
		lasVegas.showdown(); // publish hands from all players
		lasVegas.tellHandSummary();
		lasVegas.printHandSummary();
		lasVegas.fileHandSummary();
		lasVegas.prepareNext(); // increase button, counter etc.
	}

}

