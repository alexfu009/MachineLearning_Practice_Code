# -*- coding: utf-8 -*-
"""
Created on Thu  13:59:04 

@author: alex
"""

import pandas as pd

import numpy as np


"lesson 11"

print('############### lesson11 ################')

s = pd.Series([1,3,6,np.nan,44,1])

print(s)

dates = pd.date_range(start = "20170505",periods = 5)

print(dates)

df = pd.DataFrame(np.random.randn(5,4),index=dates,columns=['ee','ff','gg','tt'])

print(df)


df1 = pd.DataFrame(np.random.randn(4,5))

print(df1)

df2= pd.DataFrame({'A':1.0,
                   'B':pd.Timestamp('20130102'),
                   'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                   'D':np.array([3]*4,dtype='int32'),
                    'E':pd.Categorical([" ","trani","train","ee"]),
                    'F':'foo'})

print(df2)

print(df2.dtypes)
print(df2.index)
print(df2.columns)
print(df2.values)

print(df2.describe())

print(df2.T)

print(df2.sort_index(axis=0,ascending=False))

print(df2.sort_values(by='E'))




'''lesson12 '''

print('############### lesson12 ################')

dates12 = pd.date_range('20130101',periods=7)

df12 = pd.DataFrame(np.arange(28).reshape((7,4)),index=dates12,columns=['A','B','C','D'])

print(df12['A'],df12.A) 

print(df12[0:3],df12['20130102':])

#select by label:loc
print(df12.loc['20130102'])
print(df12.loc[:,['A','B']])

print('##########')

#select by position:iloc
print(df12.iloc[[3],1:3])

print(df12[3:4])


#mixed selection:ix
print(df12.ix[:3,['A','C']])

#Boolean indexing
print(df12[df12.A > 8])



'''lesson13 '''

print('############### lesson13 ################')

dates13 = pd.date_range('20130101',periods=7)

df13 = pd.DataFrame(np.arange(28).reshape((7,4)),index=dates13,columns=['A','B','C','D'])

df13.iloc[2,2] = 1111
df13.loc['20130101','B'] = 2222

df13.A[df13.A > 4] = 666

df13['E'] = pd.Series([3]*7,index=pd.date_range('20130101',periods=7))

print(df13)




'''lesson14  '''

print('############### lesson14 ################')

df13.iloc[0,1] = np.nan

print(df13)

print(df13.dropna(axis=1,how='any'))#how = {'any','all'}


print(df13.fillna(value = 1.0))

print(df13.isnull())

print(np.any(df13.isnull())==True)





'''lesson15  '''





'''lesson16  '''

print('############### lesson16 ################')

#concatenating
df161 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df162 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
df163 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])


res16 = pd.concat([df161,df162,df163],axis=0,ignore_index=True)

print(res16)

#join [inner,outer]

df164 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[2,3,4])
df165 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[1,2,3])

res161 = pd.concat([df164,df165],axis=0)

print(res161)

res162 = pd.concat([df164,df165],axis=0,join='outer')

print(res162)

res163 = pd.concat([df164,df165],axis=0,join='inner')

print(res163)

#join_axes

res164 = pd.concat([df164,df165],axis=1)

print(res164)

res165 = pd.concat([df164,df165],axis=1,join_axes=[df164.index])

print(res165)

res166 = pd.concat([df164,df165],axis=0,join_axes=[df164.columns])

print(res166)

# append
df167 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df168 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df169 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d', 'e'], index=[2,3,4])
res160 = df167.append(df168, ignore_index=True)
res160 = df167.append([df168, df169])
print(res160)

s161 = pd.Series([1,2,3,4], index=['a','b','c','d'])
res160 = df167.append(s161, ignore_index=True)

print(res160)





'''lesson17  '''

print('############### lesson17 ################')

# merging two df by key/keys. (may be used in database)

# consider two keys
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                             'key2': ['K0', 'K1', 'K0', 'K1'],
                             'A': ['A0', 'A1', 'A2', 'A3'],
                             'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                              'key2': ['K0', 'K0', 'K0', 'K0'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)
res171 = pd.merge(left, right, on=['key1', 'key2'], how='inner')  # default for how='inner'
print(res171)
# how = ['left', 'right', 'outer', 'inner']
res172 = pd.merge(left, right, on=['key1', 'key2'], how='left')
print(res172)


# indicator
df171 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df172 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
print(df171)
print(df172)
res173 = pd.merge(df171, df172, on='col1', how='outer', indicator=True)
# give the indicator a custom name
res174 = pd.merge(df171, df172, on='col1', how='outer', indicator='indicator_column')
print(res173)
print(res174)


# merged by index
left17 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                                  'B': ['B0', 'B1', 'B2']},
                                  index=['K0', 'K1', 'K2'])
right17 = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                                     'D': ['D0', 'D2', 'D3']},
                                      index=['K0', 'K2', 'K3'])
print(left17)
print(right17)
# left_index and right_index
res175 = pd.merge(left17, right17, left_index=True, right_index=True, how='outer')
res176 = pd.merge(left17, right17, left_index=True, right_index=True, how='left')

print(res175)
print(res176)


# handle overlapping
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
res177 = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
print(res177)






'''lesson18  '''

print('############### lesson18 ################')


import matplotlib.pyplot as plt

#plot data

data = pd.Series(np.random.randn(1000))

data = data.cumsum()
print(data.head())
data.plot()

#DataFrame

data1 = pd.DataFrame(np.random.randn(100,4),index=np.arange(100),
                     columns=['A','B','C','D'])
print(data1.head(10))
data1.plot()

# plot methods:
# 'bar', 'hist', 'box', 'kde', 'area', scatter', hexbin', 'pie'
ax1 = data1.plot.scatter(x='A', y='B', color='DarkBlue', label="Class 1")
data1.plot.scatter(x='A', y='C', color='LightGreen', label='Class 2',ax=ax1)


plt.show()





















































