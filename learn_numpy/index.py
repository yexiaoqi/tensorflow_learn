import numpy as np
a=np.arange(3,15).reshape((3,4))
print(a)
print(a[1][1])
print(a[1,1:3])
for row in a:
    print(row)
for column in a.T:
    print(column)
print(a.flatten())
for item in a.flat:
    print(item)