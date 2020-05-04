import numpy as np

a=np.array([1,1,1])
b=np.array([2,2,2])
c=np.vstack((a,b))
print(c)
print(a.shape,c.shape)
d=np.hstack((a,b))
print(d)
print(d.shape)


print("1111111111111111111")
print(a[np.newaxis,:])
print(a[np.newaxis,:].shape)
print(a[:,np.newaxis])
print(a[:,np.newaxis].shape)

print('222222222222222222222222')
a=np.array([1,1,1])[:,np.newaxis]
b=np.array([2,2,2])[:,np.newaxis]
print(a.shape,b.shape)
c=np.concatenate((a,b,b,a),axis=0)
print(c)
d=np.concatenate((a,b,b,a),axis=1)
print(d)