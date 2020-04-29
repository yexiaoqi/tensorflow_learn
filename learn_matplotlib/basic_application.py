import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-1,1,300)
y1=np.square(x)
y2=2*x+1

plt.figure()
plt.plot(x,y1)

plt.figure(num=3,figsize=(8,3))
plt.plot(x,y1)
# plt.plot(x,y2,c='red',lw=1.0,ls='--')
plt.plot(x,y2,color='red',linewidth=1.0,linestyle='--')#这个和上面等价


plt.show()

