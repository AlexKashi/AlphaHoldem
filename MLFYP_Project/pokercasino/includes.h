#pragma once
#include "./omp/HandEvaluator.h"
#include<chrono> 
#include<thread>
#include<memory> // smart pointers
#include<fstream> // smart files
#include<tuple> 
#include<utility> // tuples
#include<iostream>
#include<vector>
#include<deque>
#include<array>
#include <cstdlib>
#include <ctime>
#include <string>
 
using namespace std;

//Magic numbers:
const int nBots = 3; // : number of bots
const int nMaxRaises = 3;
const int nRounds = 100; // number of rounds to test the bots
const int nLogFrequency = 10; // frequency of result logs
const std::chrono::milliseconds kDelay(80); // time given to each bot is set in the function attend()
const array<string, 52> eRank = {
	"2h", "2c", "2s", "2d",
	"3h", "3c", "3s", "3d",
	"4h", "4c", "4s", "4d",
	"5h", "5c", "5s", "5d",
	"6h", "6c", "6s", "6d",
	"7h", "7c", "7s", "7d",
	"8h", "8c", "8s", "8d",
	"9h", "9c", "9s", "9d",
	"Th", "Tc", "Ts", "Td",
	"Jh", "Jc", "Js", "Jd",
	"Qh", "Qc", "Qs", "Qd",
	"Kh", "Kc", "Ks", "Kd",
	"Ah", "Ac", "As", "Ad",
};

