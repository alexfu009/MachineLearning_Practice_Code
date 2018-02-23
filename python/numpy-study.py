# -*- coding: utf-8 -*-
"""
Created on Fri 09:57:29 2017

@author: alex
"""

import numpy as np

'''lesson3'''
array = np.array([[1,1,3],[2,3,4]])
array666 = np.array([1]*5)

print(array)
print(array666)

print('number of dim: ',array.ndim)
print('number of shape is ',array.shape)
print('number of size is ',array.size)



'''lesson4 '''

a = np.array([2,23,24],dtype=np.int64)

print(a.dtype)

b = np.zeros((3,5))

print(b)

c =np.ones((3,5),dtype = np.int)

print(c)

d = np.empty((3,5))

print(d)

e = np.arange(10,20,2)

print(e)

f = np.arange(12).reshape((3,4))

print(f)

l = np.linspace(1,10,4)

print(l)



'''lesson5'''
aa= np.array([10,11,12,13])
bb= np.arange(4)

print(aa,bb)

cc= bb**2

print(cc)

print(bb<3)
print(bb==3)

dd= np.array([[2,2],
             [0,1]])

ee = np.arange(4).reshape((2,2))

print(dd)
print(ee)

#每个矩阵中的相同位置的数字逐个相乘
ff = dd*ee
#matrix 乘法
cc_dot = np.dot(dd,ee)

rr = np.random.random((2,5))

print(rr)

print(np.sum(rr))
print(np.min(rr))
print(np.max(rr))







'''lesson6'''

A = np.arange(2,14).reshape((3,4))

print(np.argmin(A))
print(np.argmax(A))

print(np.mean(A))
print(np.median(A))
print(np.cumsum(A))
print(np.diff(A))

B = np.array([[1,1,0],
              [1,0,1]
              ])
print(np.nonzero(B))

print(np.sort(B))

print((B.T).dot(B))

print(np.clip(B,5,9))

print(np.mean(B,axis=1))





'''lesson7'''

C = np.arange(4,16).reshape(3,4)
print(C)
print(C[2][3])
print(C[2,0:])
print(C.flatten())

'''C.flat代表了C的元素的迭代器'''
for i in C.flat: 
    print(i)
    
for column in C.T:
    print(column)
    

    
'''lesson8'''
print('//////////////////lesson8/////////////////')

D = np.array([1,1,1])
E = np.array([2,2,2])

F=np.vstack((D,E)) #vertical stack 
G=np.hstack((D,E)) #horizontal stack
print(F)
print(G)   

print(D.shape,F.shape,G.shape) 

'''使用newaxis可以为一个数组增加维度，
   下面第一种的写法就是为D增加一个row方向的一个维度。
   于是D数组就变成了维度为（1，3）的矩阵
   第二种写法，即在列方向增加维度，于是变为（3，1）维度'''
print(D[np.newaxis,:])
print(D[np.newaxis,:].shape)

print(D[:,np.newaxis])
print(D[:,np.newaxis].shape)

d8 =  np.array([[1,1,1]])
list = [d8,d8,d8]

H = np.concatenate(list)

print(H)
print(H.shape)

H8 = np.hstack((H,H)) 

print(H8)
print(H8.shape)




'''lesson9'''
print('//////////////////lesson9/////////////////')

AA = np.array([[1,1,1],
              [2,2,2],
              [3,3,3]
              ])

print(np.split(AA,3,axis=1))

print(np.array_split(AA,2,axis=1))
    
print(np.vsplit(AA,3))
print(np.hsplit(AA,3))




'''lesson10'''
print('//////////////////lesson10/////////////////')

aaa = np.arange(4,dtype=float)
bbb = aaa
ccc = aaa
ddd = bbb

aaa[0] = 0.3

print(aaa)

print(ddd)

print(ddd is aaa)

fff = aaa.copy() #deep copy

print(fff is aaa)

aaa[1] = 11

print(fff)






























