import numpy as np

a=np.arange(4)
b=a
a[0]=11
print(b is a)
b[1:3]=[22,33]
print(a)

b=a.copy()
print(b)
a[3]=111
print(a)
print(b)