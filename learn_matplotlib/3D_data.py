import numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=Axes3D(fig)

x=np.arange(-4,4,0.25)
y=np.arange(-4,4,0.25)
x,y=np.meshgrid(x,y)
r=np.sqrt(x**2+y**2)
z=np.sin(r)

ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'),edgecolor='black')

ax.contourf(x,y,z,zdir='z',offset=-2,cmap=plt.get_cmap('rainbow'))

ax.set_zlim(-2,2)
plt.show()