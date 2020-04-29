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

plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am x')
plt.ylabel('I am y')

new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)

# 格式化转义 字符串首尾 r'$...$' （matplotlib中），就是把\ 变成了空格，实际上直接'really bad'这种字符串也行
plt.yticks([-2,-1.8,-1,1.22,3],['really bad',r'$bad$',r'$normal$',r'$good$',r'$really\ good$'])

plt.show()

