import numpy as np

a=np.array([1,2,3],dtype=np.int)
print(a)
print('dim:',a.ndim)
print('shape',a.shape)
print('size',a.size)
print(a.dtype)
b=np.zeros((3,4),dtype=np.int)
c=np.ones((3,4),dtype=np.int)
d=np.empty((3,4))
e=np.arange(12).reshape((3,4))
f=np.linspace(1,10,20).reshape(((5,4)))
g=1