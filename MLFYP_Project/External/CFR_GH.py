import numpy as np
import random
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

ROCK, PAPER, SCISSORS = 0, 1, 2
NUM_ACTIONS = 3
oppStrategy = np.array([0.4, 0.3, 0.3])

def value(p1, p2):
    if p1==p2:
        return 0
    if (p1 - 1)% NUM_ACTIONS == p2:
        return 1
    else:
        return -1

def normalize(strategy):
    strategy = np.copy(strategy)
    normalizingSum = np.sum(strategy)
    if normalizingSum > 0:
        strategy /= normalizingSum
    else:
        strategy= np.ones(strategy.shape[0])/strategy.shape[0]

    return strategy

def getStrategy(regretSum):
    strategy = np.maximum(regretSum, 0)
    return normalize(strategy)

def getAverageStrategy(strategySum):
    return normalize(strategySum)

def getAction(strategy):
    strategy = strategy/ np.sum(strategy)  ## Normalize
    rr = random.random()
    a = np.cumsum(strategy)
    x = np.searchsorted(a, rr)
    return x


def train(iterations):
    # <Get regret-matched mixed-strategy actions>
    # <Compute action utilities>
    # <Accumulate action regrets>

    regretSum = np.zeros(NUM_ACTIONS, dtype = np.float64)
    strategySum = np.zeros(NUM_ACTIONS, dtype = np.float64)

    actionUtility = np.zeros(NUM_ACTIONS)
    for i in range(iterations):
        strategy = getStrategy(regretSum)
        strategySum += strategy

        myAction = getAction(strategy)
        otherAction = getAction(oppStrategy)

        actionUtility[otherAction] = 0
        actionUtility[(otherAction + 1) % NUM_ACTIONS] = 1
        actionUtility[(otherAction - 1) % NUM_ACTIONS] = -1

        regretSum += actionUtility - actionUtility[myAction]

    return strategySum



strategySum = train(100000)


strategy = getAverageStrategy(strategySum)
print(strategy)
vvv = []

for i in range(100):
    vv = 0
    for j in range(100):
        myAction = getAction(strategy)
        otherAction = getAction(oppStrategy)
        vv += value(myAction, otherAction)
    vvv.append(vv)
plt.title("CFR Algorithm")
plt.xlabel('Episode')
plt.ylabel('Wins')
plt.plot(sorted(vvv))
plt.show()



