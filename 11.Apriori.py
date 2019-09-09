

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1 =[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))

def scanD(D,Ck,minSupport):
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
            retList.insert(0,key)
        supportData[key]=support
    return retList,supportData

def aprioriGen(Lk,k):
    #生成候选集
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


if __name__ == '__main__':
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    print(C1)
    retList,supportData=scanD(dataSet,C1,0.5)
    print(retList)
    # print(supportData)
    # retList2 = aprioriGen(retList,2)
    # print(retList2)
    L,supportData =apriori(dataSet,0.5)
    print(L)
    print(supportData)