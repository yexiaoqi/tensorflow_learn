import numpy as np
import pandas as pd

dates=pd.date_range('20200501',periods=6)
df=pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['a','b','c','d'])
df.iloc[0,1]=np.nan
df.iloc[1,2]=np.nan


df2=df.dropna(axis=0,how='any')#必须赋值非df2，df本身没变
print(df2)

df3=df.fillna(value=0)
print(df3)

print(df.isnull())
print(np.any(df.isnull())==True)