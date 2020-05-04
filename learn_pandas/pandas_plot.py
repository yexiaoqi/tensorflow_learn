import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.Series(np.random.randn(1000),index=np.arange(1000))
data2=data.cumsum()
data2.plot()
plt.show()



data=pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),columns=list("ABCD"))
data2=data.cumsum()
data2.plot()
plt.show()

print(list("abcd"))
ax=data.plot.scatter(x='A',y='B',color='DarkBlue',label='class1')
data.plot.scatter(x='A',y='C',color='LightGreen',label='class2',ax=ax)
plt.show()