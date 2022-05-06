from cmath import inf
import numpy as np
from operator import mod
from copy import copy
state_dict = {
    0: 'athome',
    1: 'onconv',
    2: 'infront',
    3: 'atbin',
    4: 'unknown'
}

oloc_dict = {      # By feeding state dict idx, we can get mdp idx
    0 : 3,      
    1 : 1,
    2 : 2,
    4 : 0,
}

eefloc_dict = {
    0 : 3,
    1 : 1,
    2 : 2,
    3 : 0,
}

pred_dict = {
    0: 'unknown',
    1: 'bad',
    2: 'good'
}

action_dict = {
    0: 'Detect',
    1: 'Pick',
    2: 'Inspect',
    3: 'PlaceOnConveyor',
    4: 'PlaceInBin',
    5: 'Noop'
}

mdp_action_dict = {
    0 : 1,
    1 : 2,
    2 : 3,
    3 : 4,
    4 : 5,
    5 : 0,
}

nOnionLoc = len(oloc_dict)
nEEFLoc = len(eefloc_dict)
nPredict = len(pred_dict)



def isValidState(onionLoc, eefLoc, pred):
    '''
    @brief - Checks if a given state is valid or not.
    '''
    if (onionLoc == 2 and eefLoc != 2) or (onionLoc == 3 and eefLoc != 3) or \
        (onionLoc == 0 and pred != 0) or (onionLoc != 0 and pred == 0):
        return False
    return True

def isValidAction(onionLoc, eefLoc, pred, action):
    '''
    @brief - For each state there are a few invalid actions, returns only valid actions.
    '''
    assert action <= 5, 'Unavailable action. Check if action is within num actions'
    if action == 0: # Noop, this can be done from anywhere.
        return True
    elif action == 1:   # Detect
        if pred == 0 or onionLoc == 0:  # Only when you don't know about the onion
            return True
        else: return False
    elif action == 2:   # Pick
        if onionLoc == 1:   # As long as onion is on conv
            return True
        else: return False
    elif action == 3 or action == 4 or action == 5:   # Inspect # Placeonconv # Placeinbin
        if pred != 0 and onionLoc != 0: # Pred and onion loc both are known. 
            if onionLoc == eefLoc and eefLoc != 1:    # Onion is in hand and hand isn't on conv
                return True
            else: return False
        else: return False

def sid2vals(s):
    '''
    @brief - Given state id, this func converts it to the 3 variable values. 
    '''
    sid = s
    onionloc = int(mod(sid, nOnionLoc))
    sid = (sid - onionloc)/nOnionLoc
    eefloc = int(mod(sid, nEEFLoc))
    sid = (sid - eefloc)/nEEFLoc
    predic = int(mod(sid, nPredict))
    return onionloc, eefloc, predic

def vals2sid(nxtS):
    '''
    @brief - Given the 3 variable values making up a state, this converts it into state id 
    '''
    ol = nxtS[0]
    eefl = nxtS[1]
    pred = nxtS[2]
    return (ol + nOnionLoc * (eefl + nEEFLoc * pred))


names = np.loadtxt('names.csv', dtype=str)
test_pred_onion = np.loadtxt('test_pred_onion.csv')
test_pred_eef = np.loadtxt('test_pred_eef.csv')
yolo_detections = np.load('yolo_detections.npy', allow_pickle=True)
test_pred_action = np.loadtxt('test_pred_action.csv')

flag = False
tlx = -np.inf
agent = 'human'
prev_oloc, prev_eefloc, prev_pred, prev_act, curr_pred, yolo_pred, prev_coord = -1, -1, -1, -1, -1, -1, -np.inf

with open(f'human_test_results.txt' , 'a') as f:
    for n , onion, eef, yolo_preds, action in zip(names,test_pred_onion,test_pred_eef,yolo_detections,test_pred_action):
        n = n.split('/')[-1]

        if not flag:
            tlx = yolo_preds[0][0]
        # print(state_dict[onion], state_dict[eef], action_dict[action])
        
        oloc, eefloc, act = oloc_dict[onion], eefloc_dict[eef], mdp_action_dict[action]

        if prev_act == 1 and prev_pred == 0:
            if agent == 'human':
                if yolo_preds[0][0] >= tlx:
                    yolo_pred = yolo_preds[0][-1]
                    prev_coord = yolo_preds[0][0]
                else: yolo_pred = -1
            else: 
                if yolo_preds[0][0] <= tlx:
                    yolo_pred = yolo_preds[0][-1]
                    prev_coord = yolo_preds[0][0]
                else: yolo_pred = -1
            # print("At pick. Yolo det is: ", yolo_pred)
            if yolo_pred: curr_pred = 2
            elif not yolo_pred: curr_pred = 1
            else: curr_pred = 0

        if act == 4 and oloc == eefloc == 1:   # Placeonconv and onion infront
            curr_pred = 2   # 2 means good for MDP
            yolo_pred = 1   # 1 means good for yolo

        elif act == 5 and oloc == eefloc == 3: # Placeinbin and onion athome
            curr_pred = 1
            yolo_pred = 0 
        
        elif act == 1:   # Detect
            onion = 4
            curr_pred = 0
        
        elif act == 0:   # Noop
            if prev_pred != -1:
                curr_pred = copy(prev_pred)
            else: curr_pred = 0

        elif act == 2:  # Pick, onion onconv, eef not onconv
            if agent == 'human':
                if yolo_preds[0][0] >= tlx:
                    yolo_pred = yolo_preds[0][-1]
                    prev_coord = yolo_preds[0][0]
                else: yolo_pred = -1
            else: 
                if yolo_preds[0][0] <= tlx:
                    yolo_pred = yolo_preds[0][-1]
                    prev_coord = yolo_preds[0][0]
                else: yolo_pred = -1
            # print("At pick. Yolo det is: ", yolo_pred)
            if yolo_pred: curr_pred = 2
            elif not yolo_pred: curr_pred = 1
            else: curr_pred = 0
            
        elif act == 3 and oloc == eefloc == 3: # Inspect and onion athome
            curr_pred = 2
            yolo_pred = 1

        else:
            # print(state_dict[onion], state_dict[eef], pred_dict[curr_pred], action_dict[action])
            # raise ValueError
            pass

        oloc, eefloc, act = oloc_dict[onion], eefloc_dict[eef], mdp_action_dict[action]     # Generate updated values after inference

        validS = isValidState(oloc, eefloc, curr_pred)
        validA = isValidAction(oloc, eefloc, curr_pred, act)
        if validS and validA:
            # print(f"{n}, State: {state_dict[onion], state_dict[eef], pred_dict[curr_pred]}, Valid state: {validS}, Action: {action_dict[action]}, Valid action: {validA}")
            sid = vals2sid([oloc, eefloc, curr_pred])
            f.write(f'{sid},{act}\n')
        # else: 

        prev_oloc, prev_eefloc, prev_pred, prev_act = onion, eef, curr_pred, action
