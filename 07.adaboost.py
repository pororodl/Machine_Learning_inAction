import numpy as np
from sklearn.datasets import load_iris
# 这个是最基本的Adaboost 只能用于二分类
def loadSimpData():
    dataMat = np.matrix([[1.,2.1],[2,1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classLabel = np.mat([[1.0],[1.0],[-1.0],[-1.0],[1.0]])
    return dataMat,classLabel

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# 构建单层决策树
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]<=threshVal]= -1.0    # dataMatrix[:,dimen]的size 要和retArray的一样，并且只能是一列吗 好像不一定
    else:
        retArray[dataMatrix[:,dimen]>threshVal]= -1.0
    return retArray                                       # 返回预测结果

def buildStump(dataArr,classLabel,D):
    m,n = np.shape(dataArr)
    #设置一个步长
    numStep = 10.0
    minError = 1000
    bestClassEst = np.mat(np.zeros((m,1)))
    bestStump = {}

    for i in range(n):   #对于每一个维度，也就是对于每个特征
        rangeMin = dataArr[:,i].min();rangeMax = dataArr[:,i].max()
        stepSize = (rangeMax-rangeMin)/numStep
        for j in range(-1,int(numStep)+1):
            for threshIneq in ['lt','gt']:
                threshVal = rangeMin + float(j) * stepSize
                predLable = stumpClassify(dataMat,i,threshVal,threshIneq)
                errArr = np.mat(np.ones((m,1)))
                # print(errArr.shape)
                # print(predLable.shape)
                # print(classLabel.shape)
                errArr[predLable ==classLabel] = 0
                weightError = D.T*errArr
                # print('spilt:dim %d,thresh %.2f,thresh ineqal:%s,the weighterror is %.3f'%(i,threshVal,threshIneq,weightError))
                if weightError<minError:
                    minError = weightError
                    bestClassEst = predLable.copy()
                    bestStump['dim'] = i
                    bestStump['thresh']=threshVal
                    bestStump['ineq'] = threshIneq
    return bestStump,minError,bestClassEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weekClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))    # 用来存每次预测结果的加权和
    for i in range(numIt):
        bestStump, Error, classEst = buildStump(dataMat, classLabels, D)
        print('D:',D.T)
        alpha = float(0.5*np.log((1-Error)/max(Error,np.exp(-16))))     # 为了避免除0
        bestStump['alpha']=alpha
        weekClassArr.append(bestStump)
        print('classEst:',classEst.T)
        # print('classEst:',classEst)
        # print('shape of classLabels',np.shape(-1*alpha*(classLabels.T)))
        # print('shape of calssEst',np.shape(classEst))
        expon = np.multiply(-1*alpha*classLabels,classEst)     # 对应元素相乘，形状是一样的，和*一样，但是和np.dot不一样
        # print('shape of expon',np.shape(expon))
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()                      # 更新D，D表示的是样本的权重
        aggClassEst += alpha*classEst       # 这是对预测出来的结果也进行加权，具体操作是：用每个结果乘以alpha
        print('aggClassEst',aggClassEst.T)
        aggErrors  = np.multiply(np.sign(aggClassEst)!=classLabels,np.ones((m,1)))     # np.sign 函数是对大于0 的输出1，小于0的输出-1，等于0 的输出0，来判断预测和实际的是不是相符
        print('aggError',aggErrors.T)
        print('aggErrors.sum是什么意思',aggErrors.sum())    # 表示错了几个
        errorRate = aggErrors.sum()/m                  # 再除以总的样本数来表示错误率
        print('total error:',errorRate)
        if errorRate == 0.0: break
    return weekClassArr

def adaClassify(dataToClass,classifierArr):
    m = np.shape(dataToClass)[0]
    aggClassEst = np.zeros((m,1))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataToClass,classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+= classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

if __name__ == '__main__':
    # dataMat,classLabel = loadSimpData()
    dataMat,classLabel = load_data()
    print(np.shape(dataMat))
    print(np.shape((np.mat(classLabel))))
    # print(dataMat)
    # print(classLabel)
    # ret = stumpClassify(dataMat,0,1.5,'lt')
    # print(ret)

    # D = np.mat(np.ones((5,1))/5)
    # bestStump, minError, bestClassEst=buildStump(dataMat,classLabel,D)
    # print(bestStump)
    # print(minError)
    # print(bestClassEst)

    weekClassArr = adaBoostTrainDS(dataMat,classLabel,9)
    # print(weekClassArr)

    adaResult = adaClassify(np.mat([0,0]),weekClassArr)
    print(adaResult)
