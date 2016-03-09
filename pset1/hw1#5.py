#David Wang
#CS542 Spring 2016
#Professor Chin

#Machine-Learning Rock-Paper-Scissors Algorithm
#To beat: http://www.nytimes.com/interactive/science/rock-paper-scissors.html?_r=0
#Difficulty: Novice

#Rock = 0, Paper = 1, Scissors = 2
import random

def txtToArray(fileName):
    moveHistory = []
    for line in open(fileName):
        if line[0] == 'R':
            moveHistory.append(0)
        elif line[0] == 'P':
            moveHistory.append(1)
        elif line[0] == 'S':
            moveHistory.append(2)
        else:
            continue
    return moveHistory

def nextMove(RPSHistory):
    if len(RPSHistory) < 5:
        return random.randint(0,2)
    last = len(RPSHistory) - 2
    secl = last - 1
    thirl = secl - 1
    fourl = thirl - 1
    mostRecent = [RPSHistory[last + 1], RPSHistory[secl + 1], RPSHistory[thirl + 1], RPSHistory[fourl + 1]]
    pattern = [0]*4
    #for most recent 4 moves
    for i in range(len(RPSHistory) - 4):
        match = [RPSHistory[last - i], RPSHistory[secl - i], RPSHistory[thirl - i], RPSHistory[fourl - i]]
        if match[0] == mostRecent[0] and match[1] == mostRecent[1] and match[2] == mostRecent[2] and match[3] == mostRecent[3]:
            pattern[3] = 1
            if RPSHistory[last - i + 1] == 0:
                pattern[0] += 1
            elif RPSHistory[last - i + 1] == 1:
                pattern[1] += 1
            else:
                pattern[2] += 1
    if pattern[3]:
        index = pattern[:3].index(max(pattern[:3]))
        if index == 0:
            return 2
        elif index == 1:
            return 0
        else:
            return 1
    #for most recent 3 moves
    for i in range(len(RPSHistory) - 3):
        match = [RPSHistory[last - i], RPSHistory[secl - i], RPSHistory[thirl - i]]
        if match[0] == mostRecent[0] and match[1] == mostRecent[1] and match[2] == mostRecent[2]:
            pattern[3] = 1
            if RPSHistory[last - i + 1] == 0:
                pattern[0] += 1
            elif RPSHistory[last - i + 1] == 1:
                pattern[1] += 1
            else:
                pattern[2] += 1
    if pattern[3]:
        index = pattern[:3].index(max(pattern[:3]))
        if index == 0:
            return 2
        elif index == 1:
            return 0
        else:
            return 1
    #for most recent 2 moves
    for i in range(len(RPSHistory) - 2):
        match = [RPSHistory[last - i], RPSHistory[secl - i]]
        if match[0] == mostRecent[0] and match[1] == mostRecent[1]:
            pattern[3] = 1
            if RPSHistory[last - i + 1] == 0:
                pattern[0] += 1
            elif RPSHistory[last - i + 1] == 1:
                pattern[1] += 1
            else:
                pattern[2] += 1
    if pattern[3]:
        index = pattern[:3].index(max(pattern[:3]))
        if index == 0:
            return 2
        elif index == 1:
            return 0
        else:
            return 1
    #for most recent move
    for i in range(len(RPSHistory) - 1):
        match = [RPSHistory[last - i]]
        if match[0] == mostRecent[0]:
            pattern[3] = 1
            if RPSHistory[last - i + 1] == 0:
                pattern[0] += 1
            elif RPSHistory[last - i + 1] == 1:
                pattern[1] += 1
            else:
                pattern[2] += 1
    if pattern[3]:
        index = pattern[:3].index(max(pattern[:3]))
        if index == 0:
            return 2
        elif index == 1:
            return 0
        else:
            return 1
    else:
        return random.randint(0,2)

nextm = nextMove(txtToArray("RPS2.txt"))
if nextm == 0:
    print("Rock, Comp: Scissors")
elif nextm == 1:
    print("Paper, Comp: Rock")
else:
    print("Scissors, Comp: Paper")

