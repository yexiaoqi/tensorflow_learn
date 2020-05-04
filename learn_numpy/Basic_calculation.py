import numpy as np


a=np.array([10,20,30,40])
b=np.arange(4)
minus=a-b
plus=a+b
mul=a*b
power=b**2
sin=10*np.sin(a)
print(b<3)


a=np.array([[1,1],[0,1]])
b=np.arange(4).reshape((2,2))
c_dot=np.dot(a,b)
c_dot_2=a.dot(b)

a=np.random.random((2,4))
print(a)
print(np.sum(a))
print(np.min(a))
print(np.max(a))

print(np.sum(a,axis=1))
print(np.min(a,axis=0))
print(np.max(a,axis=1))

a=np.arange(2,14).reshape((3,4))
print(np.argmin(a))
print(np.argmax(a))

print(np.mean(a))
print(np.average(a))
print(a.mean())
print(np.median(a))
print(np.cumsum(a))
print(np.diff(a))
print(np.nonzero(a))

a=np.arange(14,2,-1).reshape((3,4))
print(np.sort(a))
print(np.transpose(a))
print(a.T)
print(np.clip(a,5,9))
