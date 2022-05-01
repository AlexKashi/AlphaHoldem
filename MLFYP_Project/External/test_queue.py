import heapq
from collections import deque
# A heap is a binary tree in which each node has a smaller key than its children; this property is called the heap property or heap invariant. 
#    0
#   / \
#  1   2
# / \ / \
# 3 4 5 6


evaluation = 3000
preflop_range_upper_notdealer = 5000


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority,item))
        self._index += 1
    def pop(self):
        return heapq.heappop(self._queue)[-1] # pop method returns the smallest item, not the largest

class Item:
    def __init__(self, action, position):
        self.action = action
        self.position = position
    def __repr__(self):
        return 'Item({!r})'.format(self.name)


def func(last_seq_move, who_dealer, bot_position):


    q = PriorityQueue()
    
    if(bot_position == 'BTN'):

        sb_move = ''
        bb_move = ''

        try: 
            sb_move = last_seq_move[-2]
        except:
            print("Cannot access sb_move with last_seq_move of length: ", len(last_seq_move))
        try: 
            bb_move = last_seq_move[-1]
        except:
            print("Cannot access bb_move with last_seq_move of length: ", len(last_seq_move))

        if(bb_move == 'r'):     
            q.push(Item('Raise'), 1)
            q.push(Item('Call'), 3) # Design Nueral network to learn these weights
            q.push(Item('Fold'), 3)
            act = 'c'
            
            
            if(sb_move == 'c'):
                q.push(Item('Raise'), 1)
                q.push(Item('Call'), 3) # Design Nueral network to learn these weights
                q.push(Item('Fold'), 3)
                act = 'c' 

            return act
            

        elif(bb_move == 'c'):
            q.push(Item('Raise'), 3)
            q.push(Item('Call'), 2)
            act = 'r'
            
            if(sb_move == 'c'):
                q.push(Item('Raise'), 1)
                q.push(Item('Call'), 3) # Design Nueral network to learn these weights
                q.push(Item('Fold'), 3)
                act = 'c'

            return act
                

        if(len(last_seq_move) == 0): # very first move of game
            # limp_success_cfr = q.pop()
            q.push(Item('Raise', bot_position), 3)
            q.push(Item('Call', bot_position), 1) # Design Nueral network to learn these weights
            q.push(Item('Fold', bot_position), 0)
            return 'r'
        


    elif(bot_position == 'SB'):

        btn_move = ''
        bb_move = ''

        try: 
            btn_move = last_seq_move[-1]
        except:
            print("Cannot access btn_move with last_seq_move of length: ", len(last_seq_move))
        try: 
            bb_move = last_seq_move[-2]
        except:
            print("Cannot access bb_move with last_seq_move of length: ", len(last_seq_move))
            
        if len(last_seq_move) == 1:
            
            if evaluation < preflop_range_upper_notdealer: 
                
                return 'r'
                
            else:
                
                return 'c'

        elif len(last_seq_move) >= 2:  
            if last_seq_move == 'r':
                return 'c'

    elif(bot_position == 'BB'):
        
        btn_move = ''
        sb_move = ''

        try: 
            sb_move = last_seq_move[-1]
        except:
            print("Cannot access sb_move with last_seq_move of length: ", len(last_seq_move))
        try: 
            btn_move = last_seq_move[-2]
        except:
            print("Cannot access btn_move with last_seq_move of length: ", len(last_seq_move))
            

        if             
            if evaluation < preflop_range_upper_notdealer: 
                return 'r'



last_seq_move = ''
roles = deque(['BTN', 'SB', 'BB'])
for i in range(3):
    who_dealer = 0
    for j in range(len(roles)):
        if(roles[j] == 'BTN'):
            who_dealer = j
    bot = roles[0]
    action_taken = func(last_seq_move, who_dealer, bot)
    roles.rotate(-1)
    last_seq_move += action_taken
    