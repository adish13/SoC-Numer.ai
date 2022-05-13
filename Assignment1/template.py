import numpy as np
import pickle
import random

corr = None
ifTest = False
Ztr = {}
Zte = {}

# States are tuples of the form (x,y) where both x,y belong to the closed interval [0,9].

# The terminology for action is :
# -1 for SELL
# 0 for HOLD
# 1 for BUY

# The returnNext function would take in the state and action, and return the next state, which you would get alongwith the price change associated with this tuple of (state,action)

# This might be different in different turns as this change is stochastic in general with only a co-relation to the (state, action) tuple.

def tuple_to_int(state):
    return 10*state[1] + state[0]

def int_to_tuple(num):
    b = num % 10
    a = int((num - b)/10)
    return (a,b)

def returnNext(state, action):
    # Returns (priceChange, nextState) as a tuple
    #nextState is actually a 2-tuple representing the state
    a,b = state
    L = []
    if ifTest:
        L = Zte[(a,b)]
    else:
        L = Ztr[(a,b)]
    r = L[random.randint(0,len(L)-1)]
    pricc = None
    if action==-1:
        if r[0]==0:
            pricc =  r[4]
        elif r[0]==1:
            pricc =  r[3]
        else:
            pricc =  -r[3]
    elif action==1:
        if r[2]==0:
            pricc =  r[4]
        elif r[2]==1:
            pricc =  r[3]
        else:
            pricc =  -r[3]
    else:
        if r[1]==0:
            pricc =  r[4]
        elif r[1]==1:
            pricc =  r[3]
        else:
            pricc =  -r[3]
    num = 3*(10*a + b) + action + 1

    F = corr[num,:]
    ch = np.random.choice(np.arange(0,100),p = list(F))
    ca = ch // 10
    cb = ch % 10
    return (pricc,(ca,cb))


## Implemented the Q - learning algorithm
Q_table = np.zeros((100,3))


# Train your Markov Decision Process, make use of the returnNext function to model and act on the priceChanges dependence on the states and actions
def Train():
    epochs = 100
    max_iter = 10000

    #initialize the exploration probability to 1
    exploration_prob = 1


    #exploartion decreasing decay for exponential decreasing
    exploration_decreasing_decay = 0.01
    
    min_exploration_prob = 0.01

    #discounted factor
    gamma = 0.99

    #learning rate
    lr = 0.1


    for e in range(epochs):
        current_state = tuple_to_int((random.randint(0,9),random.randint(0,9)))

        for _ in range(max_iter):
            if np.random.uniform(0,1) < exploration_prob:
                action = random.randint(-1,1)
            else:
                action = np.argmax(Q_table[current_state,:])
            
            current_state = int_to_tuple(current_state)
            reward,next_state = returnNext(current_state,action)
            current_state = tuple_to_int(current_state)
            next_state = tuple_to_int(next_state)

            # Q update rule
            Q_table[current_state, action] = (1-lr) * Q_table[current_state, action] +lr*(reward + gamma*max(Q_table[next_state,:]))

            current_state = next_state
    
        # update the exploration proba using exponential decay formula 
        exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay*e))



# This you need to write after training your MDP, this should just take in the state and return the Action you would perform
# Please do not make use of the returnNext function inside here, that would defeat the purpose of training the model, as it would be known to you!

def Run(state):
    st = tuple_to_int(state)
    a = Q_table[st,:]
    return a.argmax()

# This is the main function, you don't need to tamper with it!
def mainRun(iter = 1000):
    initstate = (random.randint(0,9),random.randint(0,9))
    i = 0
    initprice = 100.00
    price = initprice
    balance = 0
    bondbal = 0
    networth = 0
    st = initstate
    while i < iter:
        act = Run(st)
        pricCh, ns = returnNext(st,act)
        price += pricCh
        if act==1:
            balance -= 10*price
            bondbal += 10
        elif act==-1:
            balance += 10*price
            bondbal -= 10
        
        networth = balance + bondbal*price
        st = ns
        print('Your Networth has went from 0 to ',networth)
        i+=1

ftr = input('Test : Filename to read from?')
fte = input('Train : Filename to read?')


with open(ftr,'rb') as f:
    Ztr = pickle.load(f)

with open(fte,'rb') as f:
    Zte = pickle.load(f)

with open('correlations.npy','rb') as f:
    corr = np.load(f)

ifTest  = False
Train()
ifTest = True   
mainRun()
# print(Q_table)