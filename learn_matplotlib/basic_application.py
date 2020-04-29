import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-1,1,300)
y=np.square(x)
plt.figure()
plt.plot(x,y)
plt.show()