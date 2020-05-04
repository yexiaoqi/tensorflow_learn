import numpy as np
import pandas as pd


dates=pd.date_range('20160101',periods=6)
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
print(df)
#print(df['a'])
#print(df.a)
#print(df[0:3])
#print(df['20160102':'20160104'])


print(df.loc['20160101'])
print(df.loc[:,['a','b']])
print(df.loc['20160101',['a','b']])

print(df.iloc[3,1])
print(df.iloc[3:5,1:3])
print(df.iloc[[1,3,5],1:3])

#print(df.ix[:3,['a','c']])#新版本中已经没有ix
print(df[df.a>0.03])