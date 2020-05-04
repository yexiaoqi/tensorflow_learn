import numpy as np
import pandas as pd


dates=pd.date_range('20200501',periods=6)
df=pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['a','b','c','d'])

df.iloc[2,2]=111
df.loc['20200501','b']=2222

df.b[df.a>4]=0
df['F']=np.nan
df['e']=pd.Series([1,2,3,4,5,6],index=pd.date_range('20200501',periods=6))
print(df)