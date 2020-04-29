import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-1,1,300)
y2=np.square(x)
y1=2*x+1

plt.figure()
plt.plot(x,y1)
l1,=plt.plot(x,y1,label='linear line')
l2,=plt.plot(x,y2,color='red',linewidth=1.0,linestyle='--',label='square line')

new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3],['really bad',r'$bad$',r'$normal$',r'$good$',r'$really\ good$'])

#plt.legend(loc='upper right')
plt.legend(handles=[l1,l2],labels=['up','down'],loc='best')
plt.show()