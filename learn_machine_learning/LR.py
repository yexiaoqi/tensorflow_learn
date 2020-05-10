import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def model(w,x):
    z=np.sum(w.T*x,axis=1)[:,np.newaxis]
    return sigmoid(z)

def cross_entropy(y,y_pre):
    n_samples=y.shape[0]
    return -(np.sum(y*np.log(y_pre)+(1-y)*np.log(1-y_pre)))/n_samples

def cost_function(w,x,y):
    y_pre=model(w,x)
    return cross_entropy(y,y_pre)

def optimize(w,x,y):
    n=x.shape[0]
    learning_rate=1e-1
    y_pre=model(w,x)
    dw=(1/n)*(np.sum((y-y_pre)*x,axis=0)[:,np.newaxis])
    w=w+dw*learning_rate
    return w


def predict(x,w):
    y_pre=model(w,x)
    result=(y_pre>0.5)*1
    return result

def accuracy(w,x,y):
    y_pre=predict(x,w)
    correct_prediction=np.equal(y_pre,y)
    result=np.mean(correct_prediction)
    return result

def iterate(w,x,y,times):
    costs=[]
    accs=[]
    for i in range(times):
        w=optimize(w,x,y)
        costs.append(cost_function(w,x,y))
        accs.append(accuracy(w,x,y))
    return w,costs,accs



#https://www.zhihu.com/question/34932576
'''不是对应有个偏置么？？
回归问题都有的吧
最简单的y=w0+w1*x1
相当于y=w0*1+w1*x1
就是说x0=1
X0假设为1，为了求出weights的第一个参数，所以加了一列数据1'''
def add_ones(x):
    ones=np.ones((x.shape[0],1))
    x_with_ones=np.hstack((ones,x))
    return x_with_ones


if __name__=="__main__":
    dataset=load_breast_cancer()
    data=pd.DataFrame(data=dataset.data,columns=dataset.feature_names)
    data['cancer']=[dataset.target_names[t] for t in dataset.target]

    x=dataset.data
    y=dataset.target
    n_features=x.shape[1]

    std=x.std(axis=0)
    mean=x.mean(axis=0)
    x_norm=(x-mean)/std

    x_with_ones=add_ones(x_norm)

    x_train,x_test,y_train,y_test=train_test_split(x_with_ones,y,test_size=0.3,random_state=12345)
    y_train=y_train[:,np.newaxis]
    y_test=y_test[:,np.newaxis]

    w=np.ones((n_features+1,1))
    w,costs,accs=iterate(w,x_train,y_train,1500)



    plt.subplot(2,2,1)
    plt.plot(costs)
    plt.subplot(2, 2, 2)
    plt.plot(costs[:100])
    plt.subplot(2, 2, 3)
    plt.plot(accs)
    plt.subplot(2, 2, 4)
    plt.plot(accs[:100])



    # plt.figure()
    # plt.plot(costs)
    # plt.figure()
    # plt.plot(costs[:100])
    # plt.figure()
    # plt.plot(accs)
    # plt.figure()
    # plt.plot(accs[:100])
    plt.show()
    print(costs[-1],accs[-1])
    print(accuracy(w,x_test,y_test))