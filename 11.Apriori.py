import numpy as np
import pandas as pd
import pickle
'''
apriori原理： 
频繁项集的规则：候选集不是频繁项集（支持度没有超过我们设定的值），那么他的超集也不是频繁项集
关联规则的规则：某条规则不满足（置信度没有超过我们设定的值），那么左部为他的子集的左部形成的规则也都不是关联规则
'''
def loadDataSet():
    # file = pd.read_csv('D:/03LIULU/WeChat Files/LL-961126/FileStorage/File/2019-09/final.csv',delimiter='\t', header=None,encoding='gb2312')
    # file = file.values

    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
    # return file



def createC1(dataSet):
    '''
    在数据中的单个元素摘出来
    :param dataSet: 例子：[[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
    :return: C1: [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
    '''
    C1 =[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))

def scanD(D,Ck,minSupport):
    '''
    计算候选集出现的频率
    :param D:dataSet
    :param Ck:生成的候选集
    :param minSupport:最小支持度
    :return:retList候选集，supportData,支持度的字典
    '''
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                # if not ssCnt.has_key(can):
                if not can in ssCnt:
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1

    # 开始计算支持度
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support>=minSupport:
            retList.insert(0,key)   #在字典的头部插入键
        supportData[key]=support
    return retList,supportData

def aprioriGen(Lk,k):
    '''
    由k-1个元素的候选集生成由k个元素组成的候选集，这里有用到一个技巧就是将前k-2个元素相同的集合合并，可以减少扫描次数，这里需要看书上的例子
    :param Lk: 有k个元素的候选集
    :param k:
    :return:
    '''
    retList= []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            #前k-2个项相同时，将两个集合合并
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 ==L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet,minSupport=0.5):
    '''
    完整的apriori算法
    :param dataSet:
    :param minSupport:
    :return:L是得到的频繁项集，supportData 是频繁项集对应的频率
    '''
    C1 = createC1(dataSet)  # 得到第一个的候选集
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport) # 得到第一个的频繁项集L1
    L = [L1]
    k = 2
    while (len(L[k-2])>0):
        Ck = aprioriGen(L[k-2],k)
        Lk ,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L,supportData


def calConf(freqSet,H,supportData,brl,minconf=0.7):
    '''
    计算支持度
    :param freqSet:一个频繁项集（有1，2，3几种长度）
    :param H: 一个频繁项集中的一个元素
    :param supportData: 之前得到的所有候选集的频率
    :param brl: 得到的最后的关联规则
    :param minconf: 最小置信度
    :return:所有可以当作右部的候选集（我觉得）
    '''
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf>minconf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    print('prunedH:',prunedH)
    return prunedH

def rulesFromConseq(freqSet,H,supportData,brl,minconf = 0.7):
    '''
    没太理解是干什么的i>1 就调用这个函数，表示这个函数是用来解决频繁项集里的元素超过两个时怎么生成关联规则的
    :param freqSet:
    :param H:
    :param supportData:
    :param brl:
    :param minconf:
    :return:
    '''
    m = len(H[0])
    if (len(freqSet)>(m+1)):
        Hmpl = aprioriGen(H,m+1)
        Hmpl = calConf(freqSet,Hmpl,supportData,brl,minconf)
        if (len(Hmpl)>1):
            rulesFromConseq(freqSet,Hmpl,supportData,brl,minconf)

def generateRules(L,supportData,minConf = 0.7):
    '''
    生成关联规则和对应的频率组成list
    :param L:
    :param supportData:
    :param minConf:
    :return:
    '''
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            # print('H1:',H1)
            # print(len(H1[0]))
            if (i>1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)   # 不太理解对有超过两个元素的频繁项集是怎么处理的
            else:
                calConf(freqSet,H1,supportData,bigRuleList,minConf)
    # print('bigRuleList:',bigRuleList)

if __name__ == '__main__':
    dataSet = loadDataSet()
    # C1 = createC1(dataSet)
    # print(C1)
    # retList,supportData=scanD(dataSet,C1,0.01)
    # print(retList)
    # print(supportData)
    # retList2 = aprioriGen(retList,2)
    # print(retList2)
    L,supportData =apriori(dataSet,0.0)
    # pickle.dump(supportData,open('support.txt','w'))
    # 写到文件里
    print(L)
    print(supportData)
    generateRules(L,supportData,0.0)
