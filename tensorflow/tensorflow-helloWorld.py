# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

#from sklearn.learning_curve import learning_curve

import tensorflow as tf

import matplotlib.pyplot as plt

hello = tf.constant('Hello,Auburn!')
sess = tf.Session()
print(sess.run(hello))

matrix1 = tf.constant([[3,4]])

matrix2 = tf.constant([[2],
                       [3]])

product = tf.matmul(matrix1,matrix2,name='auburn')

'''
sess = tf.Session()

result = sess.run(product)

print(result)

sess.close()


with tf.Session() as sess:
    result = sess.run(product)
    print(result)
    
state = tf.Variable(3,name='counter')

one = tf.constant(1)

new_value = tf.add(state , one)

update = tf.assign(state , new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))
        
input1 = tf.placeholder(tf.float32)

input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess2:
    print(sess2.run(output,feed_dict={input1:[7.],input2:[3.]}))
'''

def add_layer(inputs,in_size,out_size,activation_function=None):
   with tf.name_scope('layer'):
     with tf.name_scope('Weights'):
       Weights = tf.Variable(tf.random_normal([in_size,out_size]))
     with tf.name_scope('biases'):
       biases = tf.Variable(tf.zeros([1,out_size])+0.1) 
     with tf.name_scope('Wx_plus_b'):
       Wx_plus_b = tf.matmul(inputs,Weights) + biases
     if activation_function is None:
         outputs = Wx_plus_b
     else:
         outputs = activation_function(Wx_plus_b)
     return outputs
   

x_data = np.linspace(-1,1,300, dtype=np.float32)[:,np.newaxis]

noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)

y_data = np.square(x_data)-0.5 + noise

with tf.name_scope('inputs'):
  xs = tf.placeholder(tf.float32,[None,1],name='x_input')
  ys = tf.placeholder(tf.float32,[None,1],name='y_input')

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

prediction = add_layer(l1,10,1,activation_function=None)

with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
with tf.name_scope('train'):
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

writer = tf.summary.FileWriter("logs/", sess.graph)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 ==0:
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
        lines = ax.plot(x_data,prediction_value,'g-',lw=5)
        plt.pause(1)

        
        



    
    


    
