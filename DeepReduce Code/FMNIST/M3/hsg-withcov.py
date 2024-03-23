import os
import copy
import random
import math
import numpy as np
#import keras
#from keras import optimizers
#from keras.datasets import cifar10
#from keras.models import Sequential
#from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
#from keras.callbacks import LearningRateScheduler, TensorBoard
#from keras.models import load_model
from collections import defaultdict
import numpy as np
import sys
import time


def tree(): return defaultdict(tree) 
sys.setrecursionlimit(1000000)

def readFile(filepath):
    f = open(filepath)
    content = f.read()
    f.close()
    return content.splitlines()

predict = readFile('Cov/predict')
for i in range(len(predict)):
    predict[i] = eval(predict[i])

#print(predict)

def calculate_acc(model1,RST):
    count = 0
    acc = 0
    predict = []
    for i in range(len(RST)):
        #test_image = x_test[RST[i]].reshape([1,32,32,3])
        #y = model1.predict_classes(test_image)
        y = model1[RST[i]][0]
        if y == 1:
            acc += 1
            #predict.append((1,y[0],np.argmax(y_test[RST[i]])))
        else:
            #predict.append((0,y[0],np.argmax(y_test[RST[i]])))
            pass
        count += 1
    return acc/count

global KN
KN = 20


def hsg(mapdict,tnum):
    # initialization
    marked = [False] * len(mapdict.keys())
    MAX_CARD = 0
    RS = []
    for i in mapdict.keys():
        MAX_CARD = max(MAX_CARD,len(mapdict[i]))
        if len(mapdict[i]) == 1:
            #RS.append(mapdict[i][0]) 
            RS.extend(list(mapdict[i]))
            marked[i] = True
    for i in mapdict.keys():
        if mapdict[i] & set(RS):
            marked[i] = True
    cur_card = 1
    
    curdict = {}
    for i in mapdict.keys():
        tlen = len(mapdict[i])
        if tlen not in curdict.keys():
            curdict[tlen] = [i]
        else:
            curdict[tlen].append(i)
    print('initiation : %s'%RS)
    # compute RS
    while MAX_CARD > cur_card:
        #List = []
        cur_card += 1
        if cur_card not in curdict.keys():
            #print('%s not exists ...'%cur_card)
            continue         
        while True:
            if cur_card not in curdict.keys():
                break
            #print('size : %s '%cur_card)
            List = []
            for item in curdict[cur_card]:
                if marked[item] == False:
                    List.extend(mapdict[item])
            List = list(set(List))
            if len(List)== 0:
                break          
            next_test = SelectTest(cur_card,List,mapdict,curdict,tnum,marked,MAX_CARD)
            #print('*************************')
            #print('list : %s'%List)
            #print('candidate list : %s , candidate test : %s' %(List,next_test))
            RS.append(next_test)
            may_reduce = False
            for i in mapdict.keys():
                if next_test in mapdict[i]:
                    marked[i] = True
                    if len(mapdict[i]) == MAX_CARD:
                        may_reduce = False
                else:
                    continue
            if may_reduce:
                for i in mapdict.keys():
                    if marked[i] == False:
                        MAX_CARD = max(MAX_CARD,len(mapdict[i]))
            #print('marked : %s'%marked)
            print('cur_card : %d, reduced test : %s'%(cur_card,RS))
            #print('max_card : %d, cur_card : %d'%(MAX_CARD,cur_card))
            #input('check...')
        #input('cry!!!')
    return RS


def SelectTest(SIZE,LIST,mapdict,curdict,tnum,marked,MAX_CARD):
    count = [0]*tnum
    maxcount = 0
    #if SIZE not in curdict.keys():
    #    return SelectTest(SIZE+1,LIST,mapdict,curdict,tnum,marked,MAX_CARD)
    for ti in LIST:
        temp = 0
        try:
            for ni in curdict[SIZE]:
                if (marked[ni] == False) and (ti in mapdict[ni]):
                    temp += 1
            count[ti]
            try:
                count[ti] = temp
            except:
                print('error ...')
                print('length : %d : %s'%(len(count),count))
                print(ti)
                input('SelectTest check...')
        except:
            print(SIZE)
            #print(curdict.keys())
            cc = list(curdict.keys())
            cc.sort()
            print(cc)
            input('key error check...')
        maxcount = max(maxcount,count[ti])
    testlist = []
    for ti in LIST:
        if count[ti] == maxcount:
            testlist.append(ti)
    if len(testlist) == 1:
        return testlist[0]
    elif SIZE == MAX_CARD:
        try:
            return testlist[random.randint(0,len(testlist)-1)]
        except:
            print('error2 ...')
            print(LIST)
            print(testlist)
            input('check...')
    else:
        #print('debug')
        cc = list(curdict.keys())
        cc.sort()
        ccindex = cc.index(SIZE)
        return SelectTest(cc[ccindex+1],testlist,mapdict,curdict,tnum,marked,MAX_CARD) 


def SelectTest_cross_entropy(SIZE,LIST,mapdict,curdict,tnum,marked,MAX_CARD,maxdepth):
    count = [0]*tnum
    maxcount = 0
    #if SIZE not in curdict.keys():
    #    return SelectTest(SIZE+1,LIST,mapdict,curdict,tnum,marked,MAX_CARD)
    for ti in LIST:
        temp = 0
        #try:
        for ni in curdict[SIZE]:
            if (marked[ni] == False) and (ti in mapdict[ni]):
                temp += 1
        count[ti]
        try:
            count[ti] = temp
        except:
            print('error ...')
            print('length : %d : %s'%(len(count),count))
            print(ti)
            input('SelectTest check...')
        #except:
        #    print('*************************')
        #    print(tnum)
        #    print(ti)
        #    print(SIZE)
        #     print(curdict.keys())
        #    cc = list(curdict.keys())
        #    cc.sort()
        #    print(cc)
        #    input('key error check...')
        maxcount = max(maxcount,count[ti])
    testlist = []
    for ti in LIST:
        if count[ti] == maxcount:
            testlist.append(ti)
    if len(testlist) == 1:
        return testlist[0]
    elif SIZE == MAX_CARD:
        try:
            return testlist[random.randint(0,len(testlist)-1)]
        except:
            print('error2 ...')
            print(LIST)
            print(testlist)
            input('check...')
    else:
        #print('debug')
        cc = list(curdict.keys())
        cc.sort()
        ccindex = cc.index(SIZE)
        if maxdepth != 0:
            maxdepth = maxdepth - 1
            return SelectTest_cross_entropy(cc[ccindex+1],testlist,mapdict,curdict,tnum,marked,MAX_CARD,maxdepth)
        else:
            return testlist[random.randint(0,len(testlist)-1)] 



def update(mapdict,tnum,RST,RS,RSTDist,outputs,NIndex,NMap):
    for i in mapdict.keys():
        temp = list(mapdict[i])
        for j in RST:
            if j in temp:
                temp.remove(j)
        mapdict[i] = set(copy.deepcopy(temp)) 
    #for i in range(len(RSTDist)):
    #    for k in range(K):
    for i in RS:
        output = NIndex[i]
        for j in range(len(RSTDist)):
            tempindex = output[j]
            RSTDist[j][tempindex] += 1
        #print('**********************************')
        #print('output : %s'%output)
        #print('-----------------')
        #for j in range(len(RSTDist)):
        #    print('%s : %d'%(RSTDist[j],output[j]))
    return mapdict,tnum-len(RST),RSTDist

def update_dist(mapdict,tnum,RST,RS,RSTDist,outputs,NIndex,NMap):
    for i in RS:
        output = NIndex[i]
        for j in range(len(RSTDist)):
            tempindex = output[j]
            RSTDist[j][tempindex] += 1
    return mapdict,tnum-len(RST),RSTDist


def getCoverage(mapdict,RST):
    if RST == []:
        tempcount = 0
        for i in mapdict.keys():
            if len(mapdict[i]) != 0:
                tempcount += 1
        return tempcount/len(mapdict.keys())
    else:
        tempcount = 0
        #print(RST)
        for i in mapdict.keys():
            #print(RST)
            #print(mapdict[i]) 
            if (len(mapdict[i]) != 0) and (set(RST)&set(mapdict[i])):
                tempcount += 1
                #print('yes')
                #input('coverage check ...')
        return tempcount/len(mapdict.keys())

# ds : distribution of sample set; dt : distribution of test set
def calculate_cross_entropy(ds,dt,snum):
    eps = 1e-15
    if snum == 0:
        #eps = 1e-15
        cesum = 0
        print(len(ds))
        for i in range(len(ds)):
            for k in range(KN):
                cesum -= dt[i][k] * np.log(eps)
    else:
        cesum = 0
        for i in range(len(ds)):
            for k in range(KN):
                if ds[i][k] != 0 and dt[i][k] != 0:
                    cesum += dt[i][k] * np.log(dt[i][k]/(ds[i][k]/snum))
                elif ds[i][k] == 0:
                    if dt[i][k] == 0:
                        cesum += dt[i][k] * np.log(eps/eps)
                    else:
                        cesum += dt[i][k] * np.log(dt[i][k]/eps)
    '''
    print('******************************')
    print('original distribution : ')
    for i in dt:
        print(i)
    print('reduced distribution : ')
    for j in ds:
        print([c/snum for c in j])
    print('end *************************')
    '''
    return cesum/len(ds)

def cce(ds,dt):
    eps = 1e-15
    cesum = 0
    for i in range(len(ds)):
        for k in range(KN):
            cesum += max(dt[i][k],eps) * np.log(max(dt[i][k],eps)/max(ds[i][k],eps)) 
            #if ds[i][k]!= 0:
            #    cesum += dt[i][k] * np.log(dt[i][k]/ds[i][k])
            #else:
            #    cesum += dt[i][k] * np.log(eps/eps)
    return cesum/len(ds)

def hsg_crossentropy(mapdict,tnum,NDist,outputs,NIndex,NMap,filepath):
    f = open(filepath,'w')
    RST = []
    log = []
    #print('the number of test inputs : %s' %tnum)
    #input('check...')
    #mapdict_ori = copy.deepcopy(mapdict)
    default_cov = getCoverage(mapdict,[])
    dnum = len(NDist)
    #print(default_cov)
    # empty distribute set
    RSTDist = []
    #print('nnum : %s'%dnum)
    for i in range(dnum):
        RSTDist.append([0] * KN)
    default_ce = cce(NDist,NDist) 
    #Distribute = getDistrubute()
    print('the default cross entropy : %s'%default_ce)
    default_acc = calculate_acc(predict,range(testnumber))
    print('the default accuracy : %s'%default_acc)
    #input('check...') 
    terminal_cov = 0 
    terminal_ce =  float('inf')
    
    # first iteration to guarantee coverage for reduction

    for first_iteration in range(1):
        #mapdict,tnum = update(mapdict,tnum,RTT)
        # initialization
        marked = [False] * len(mapdict.keys())
        MAX_CARD = 0
        RS = []
        for i in mapdict.keys():
            MAX_CARD = max(MAX_CARD,len(mapdict[i]))
            if len(mapdict[i]) == 1:
                #RS.append(mapdict[i][0])
                if list(mapdict[i])[0] not in RS:
                    RS.extend(list(mapdict[i]))
                marked[i] = True
        for i in mapdict.keys():
            if mapdict[i] & set(RS):
                marked[i] = True
        cur_card = 1

        curdict = {}
        for i in mapdict.keys():
            tlen = len(mapdict[i])
            if tlen not in curdict.keys():
                curdict[tlen] = [i]
            else:
                curdict[tlen].append(i)
        #print('initiation : %s'%RS)

        while MAX_CARD > cur_card:
        #List = []
            cur_card += 1
            if cur_card not in curdict.keys():
                #print('%s not exists ...'%cur_card)
                continue
            while True:
                if cur_card not in curdict.keys():
                    break
                #print('size : %s '%cur_card)
                List = []
                for item in curdict[cur_card]:
                    if marked[item] == False:
                        List.extend(mapdict[item])
                List = list(set(List))
                if len(List)== 0:
                    break
                next_test = SelectTest_cross_entropy(cur_card,List,mapdict,curdict,tnum,marked,MAX_CARD,3)
                #print('*************************')
                #print('list : %s'%List)
                #print('candidate list : %s , candidate test : %s' %(List,next_test))
                RS.append(next_test) 
                may_reduce = False
                for i in mapdict.keys():
                    if next_test in mapdict[i]:
                        marked[i] = True
                        if len(mapdict[i]) == MAX_CARD:
                            may_reduce = False
                    else:
                        continue
                if may_reduce:
                    for i in mapdict.keys():
                        if marked[i] == False:
                            MAX_CARD = max(MAX_CARD,len(mapdict[i]))
                #print('marked : %s'%marked)
                #print('cur_card : %d, reduced test : %s'%(cur_card,RS))
                #print('max_card : %d, cur_card : %d'%(MAX_CARD,cur_card))
            
                #input('check...')
            #input('cry...')
        RST.extend(copy.deepcopy(RS))
        RST = list(set(RST))
        RS = list(set(RS))
        mapdict,tt1,RSTDist = update_dist(mapdict,tnum,RST,RS,RSTDist,outputs,NIndex,NMap)
        terminal_cov = getCoverage(mapdict,RST)
        terminal_ce = calculate_cross_entropy(RSTDist,NDist,len(RST))
        terminal_acc = calculate_acc(predict,RST)
        print('reduced test : %s, coverage : %s, cross entropy : %s, accuracy: %s'%(len(RST),terminal_cov,terminal_ce,terminal_acc))
        log.append('reduced test : %s, coverage : %s, cross entropy : %s, accuracy : %s'%(len(RST),terminal_cov,terminal_ce,terminal_acc))
        f.write(str(RST) + '\n')
        #input('iteration check...')
    
    # the following iteration to guarantee test distribution
    print('starting second iteration ...')
    RSTR = copy.deepcopy(RST)
    RSTRDist = copy.deepcopy(RSTDist)
    flag = 1
    while len(RST) < 5000:
        RS = []
        #RS = getCandidate(RSTDist,NDist,NMap,mapdict,len(RST),RST)
        if flag == 1:
            RS = getCandidate_new(RSTDist,NDist,NMap,mapdict,len(RST),RST,NIndex,outputs)
        else:
            RS = getCandidate_negative(RSTDist,NDist,NMap,mapdict,len(RST),RST,NIndex)
            input('kl below zero error ...')
        #RSR = []
        #RSR = random.sample(list(set(range(10000)) - set(RST)),1)
        #print('NIndex :')
        #for rs in RS:
        #    print(NIndex[rs])
        #print(candidateTest)
        mapdict,tt1,RSTDist = update_dist(mapdict,tnum,RST,RS,RSTDist,outputs,NIndex,NMap)
        #mapdict,tt2,RSTRDist = update_dist(mapdict,tnum,RSTR,RSR,RSTRDist,outputs,NIndex,NMap)
        RST.extend(RS)
        #RSTR.extend(RSR)
        #terminal_cov = getCoverage(mapdict,RST)
        terminal_ce = calculate_cross_entropy(RSTDist,NDist,len(RST))
        terminal_acc = calculate_acc(predict,RST)
        #terminal_cer = calculate_cross_entropy(RSTRDist,NDist,len(RSTR))
        #terminal_accr = calculate_acc(predict,RSTR)
        if terminal_ce > 0:
            flag = 1
        else:
            flag = 0
        print('reduced test : %s, coverage : %s, cross entropy : %s, accuracy: %s'%(len(RST),terminal_cov,terminal_ce,terminal_acc))
        #print('random test : %s, coverage : %s, cross entropy : %s, accuracy: %s'%(len(RSTR),terminal_cov,terminal_cer,terminal_accr))
        print('************************************************************************************************')
        log.append('reduced test : %s, coverage : %s, cross entropy : %s, accuracy : %s'%(len(RST),terminal_cov,terminal_ce,terminal_acc))
        f.write(str(RST) + '\n')
        #input('iteration check...')
    f.close()
    return RST,log



def getCandidate_new(RSTDist,NDist,NMap,mapdict,tnum,RST,NIndex,outputs):
    eps = 1e-15
    relist = []
    for i in range(len(RSTDist)):
        maxindex = (i,0)
        maxdiff = NDist[i][0]*tnum/max(RSTDist[i][0],eps)
        for j in range(len(RSTDist[i])):
            tempdiff = NDist[i][j]*tnum/max(RSTDist[i][j],eps)
            if tempdiff > maxdiff:
                maxdiff = tempdiff
                maxindex = (i,j)
        relist.append(maxindex[1])
    maxsim = 0
    remain = list(set(range(testnumber))- set(RST))
    simlist = [0] * len(remain)
    simdict = {}
    for i in range(len(remain)):
        simlist[i] = calSimilarity(NIndex[remain[i]],relist)
        maxsim = max(simlist[i],maxsim)
        if simlist[i] in simdict.keys():
            simdict[simlist[i]].append(remain[i])
        else:
            simdict[simlist[i]] = [remain[i]]
    minkl = 10000
    minindex = -1
    for i in range(len(simdict[maxsim])):
        exp = copy.deepcopy(RST)
        exp.append(simdict[maxsim][i])
        mapdict,tt2,exdist =  update_dist(mapdict,tnum,exp,[simdict[maxsim][i]],copy.deepcopy(RSTDist),outputs,NIndex,NMap)
        tempkl = calculate_cross_entropy(exdist,NDist,len(exp))
        if tempkl < minkl:
            minkl = tempkl
            minindex = i
    #print('length of candidate : %s, maxsimilarity : %s'%(len(simdict[maxsim]),maxsim))
    #return random.sample(simdict[maxsim],1)
    return [simdict[maxsim][minindex]]

 

def calSimilarity(a,b):
    resim = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            resim += 1
    return resim 

def getCandidate_negative(RSTDist,NDist,NMap,mapdict,tnum,RST,NIndex):
    relist = []
    for i in range(len(RSTDist)):
        maxindex = (i,0)
        maxdiff = RSTDist[i][0] - NDist[i][0]*tnum
        for j in range(len(RSTDist[i])):
            tempdiff = RSTDist[i][0] - NDist[i][0]*tnum
            if tempdiff > maxdiff:
                maxdiff = tempdiff
                maxindex = (i,j)
        relist.append(maxindex[-1])
    maxsim = 0
    remain = list(set(range(10000))- set(RST))
    simlist = [0] * len(remain)
    simdict = {}
    for i in range(len(remain)):
        simlist[i] = calSimilarity(NIndex[remain[i]],relist)
        maxsim = max(simlist[i],maxsim)
        if simlist[i] in simdict.keys():
            simdict[simlist[i]].append(remain[i])
        else:
            simdict[simlist[i]] = [remain[i]]
    return random.sample(simdict[maxsim],1)


def getSection(outputs):
    NSec = []
    #print(outputs[0])
    #print(len(outputs[0]))
    nnum = len(outputs[0])
    tnum = len(outputs)
    for i in range(nnum):
        omax = outputs[0][i]
        omin = outputs[0][i]
        for j in range(tnum):
            try:
                omax = max(omax,outputs[j][i])
                omin = min(omin,outputs[j][i])
            except:
                print(nnum)
                print(len(outputs[j]))
                print("%s : %s"%(j,i))
                print(outputs[j])
                input('getSection error check...')
        NSec.append((omin,omax))
    return NSec

def getK(tmi,tma,tm):
    step = (tma-tmi)/KN
    index = math.ceil((tm-tmi)/step)
    if index == 0:
        return index+1
    else:
        return index
    

def getDistribute(outputs,NSec):
    nnum = len(outputs[0])
    tnum = len(outputs)
    NDist = []
    for i in range(nnum):
        temps = [0] * KN
        for j in range(tnum):
            tempv = outputs[j][i]
            tempmin = NSec[i][0]
            tempmax = NSec[i][1]
            tempindex = getK(tempmin,tempmax,tempv)
            temps[tempindex-1] += 1
        for k in range(KN):
            temps[k] = temps[k]/tnum
        NDist.append(copy.deepcopy(temps))
        #print(sum(temps))
        #input('check...')
    return NDist
            

def getSectionIndex(outputs,NSec):
    nnum = len(outputs[0])
    tnum = len(outputs)
    NIndex = []
    for j in range(tnum):
        temps = [0] * nnum
        for i in range(nnum):
            tempv = outputs[j][i]
            tempmin = NSec[i][0]
            tempmax = NSec[i][1]
            tempindex = getK(tempmin,tempmax,tempv)
            temps[i] = tempindex -1
        NIndex.append(copy.deepcopy(temps))
    return NIndex

def getNeuronMap(NIndex):
    relist = tree()
    for i in range(len(NIndex[0])):
        for j in range(KN):
            relist[i][j] = set()
    for i in range(len(NIndex)):
        for j in range(len(NIndex[i])):
            tempindex = NIndex[i][j]
            relist[j][tempindex].add(i)
    return relist 

testdict = {0:{0,4},
            1:{4},
            2:{0,1,2},
            3:{2,5},
            4:{0,3},
            5:{0,5},
            6:{2,3,6},
            7:{1,2,3,6}}


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    threshold = str(sys.argv[1])
    path = os.getcwd() + '/Cov/activeneuron/' + threshold + 'ase/'
    path_res = os.getcwd() + '/result/DLR/'
    if os.path.exists(path_res) == False:
        os.makedirs(path_res)
    cov = readFile(path + 'test_cov')
    outputs_ori = readFile(os.getcwd() + '/Cov/cross_entropy')
    tnum_ori = len(cov[0])
    nnum_ori = len(cov)
    for i in range(len(outputs_ori)):
        outputs_ori[i] = eval(outputs_ori[i])
    mapdict_ori = {}
    for i in range(nnum_ori):
        mapdict_ori[i] = []
        for j in range(tnum_ori):
            if cov[i][j] == '1':
                mapdict_ori[i].append(j)
            else:
                continue
        mapdict_ori[i] = set(mapdict_ori[i])
    global testnumber
    testnumber = len(outputs_ori)
    NSec = getSection(outputs_ori)
    NDist = getDistribute(outputs_ori,NSec)
    NIndex = getSectionIndex(outputs_ori,NSec)
    NMap = getNeuronMap(NIndex)
    #print(NMap)
    #input('check...')
    
    tt,tlog = hsg_crossentropy(mapdict_ori,tnum_ori,NDist,outputs_ori,NIndex,NMap,path_res + 'hsg-cov-'+str(threshold)+'.iteration')
    #tt = hsg(mapdict_ori,tnum_ori)
    #tt = hsg(testdict,7)
    f = open(path_res + 'hsg-cov-'+str(threshold)+'.result','w')
    f.write(str(tt) + '\n')
    f.close()
    print('length : %d, %s' %(len(tt),tt))
    
    f = open(path_res + 'hsg-cov-'+str(threshold)+'.log','w')
    for item in tlog:
        f.write(str(item) + '\n')
    f.close# Record the end time
    
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Save the execution time to a file
    with open("execution_time_hsg.txt", "w") as file:
        file.write("Execution time: {} seconds".format(execution_time))

    print("Execution time:", execution_time, "seconds")