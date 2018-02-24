# -*- coding: utf-8 -*-
"""
Created on Mon 23:33:50

@author: alex
"""

import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence

from sklearn.cross_validation import train_test_split

import tensorflow as tf




#执行程序的时间记录
import time

start = time.clock()    

################################读入数据

df_train = pd.read_csv('./movie/train.tsv', sep='\t', header=0)
df_test = pd.read_csv('./movie/test.tsv', sep='\t', header=0)

#print(df_train.head())
#print(df_train.shape)

#print(df_test.head())
#print(df_test.shape)

train_data_size = 155000

X_train_pre = df_train.ix[:train_data_size,['Phrase']].values.flatten()
print(type(X_train_pre))
print(X_train_pre.shape)

#X_test_pre = df_test['Phrase'].values
X_test_pre = df_test.ix[:,['Phrase']].values.flatten()
print(type(X_test_pre))
print(X_test_pre.shape)

#Y_train = df_train.ix[:155000,['Sentiment']].values
#num_labels = len(np.unique(Y_train))
#print(type(Y_train))
#print(Y_train.shape)

Y_train = df_train.ix[:train_data_size,['Sentiment']].values
print(type(Y_train))
print(Y_train.shape)
T_Y_train = tf.convert_to_tensor(Y_train)
print(type(T_Y_train))
print(T_Y_train.shape)
aa = tf.one_hot(T_Y_train, depth=5, on_value=None, off_value=None, axis=None, dtype=None, name=None) 
print("aa is : ",aa)  
bb = tf.reshape(aa, (len(Y_train.flatten()), 5))  
print ("bb is : ",bb)  
sess_1=tf.Session()
sess_1.run(tf.global_variables_initializer())
#转化为numpy数组
Y_train_np = bb.eval(session=sess_1)
print("Y_train_np = ",type(Y_train_np))
print('Y_train_np shape is: ',Y_train_np.shape)
#print(Y_train_np[:5])
#Y_train_list = [ele.flatten().tolist() for ele in Y_train_np]
#print('Y_train_list lenght is: ',len(Y_train_list))
#print(Y_train_list[1:2])
#Y_train_final = np.array(Y_train_list)
#print('Y_train_final shape is: ',Y_train_final.shape)
#print(Y_train_final[:5])

sess_1.close



######################################文本预处理        
          
# 停止词
from nltk.corpus import stopwords
stop = stopwords.words('english')

# 数字
import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

# 特殊符号
def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))

# lemma
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def check(word):
    """
    如果需要这个单词，则True
    如果应该去除，则False
    """
    word= word.lower()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True

# 把上面的方法综合起来
def preprocessing(sen):
    res = []
    for word in sen:
        if check(word):
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res

##句子的最大长度
maxlen_sentence = 0
    
X_train_features=[]
    
for list_item in X_train_pre:
    token = word_tokenize(list_item)
    temp_token = preprocessing(token)
    if(len(temp_token)) > maxlen_sentence:
        maxlen_sentence = len(temp_token)
    X_train_features.append(temp_token)

X_train_num = len(X_train_features)   


X_test_features=[]
    
for list_item in X_test_pre:
    token = word_tokenize(list_item)
    temp_token = preprocessing(token)
    if(len(temp_token)) > maxlen_sentence:
        maxlen_sentence = len(temp_token)
    X_test_features.append(temp_token)

X_test_num = len(X_test_features)   


    
print('max length of sentence include train and test data is: ',maxlen_sentence)   
    

#X_train_np = np.array([' '.join(x) for x in X_train])
#X_test_np = np.array([' '.join(x) for x in X_test])

#print(type(X_train_np))
#print(X_train_np.shape)
#print(X_train_np)


    

################################特征提取（对句子进行编号）

##查找字典中某个值对应的键
def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

# 创建词语字典，并返回word2vec模型中词语的索引，词向量
def create_dictionaries(p_model,w2v_dem):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec_np = np.zeros([len(w2indx)+1, w2v_dem],dtype=np.float32)
    for i in range(len(w2indx)+1):
        if i==0:
            continue
        temp_list = get_keys(w2indx,i)
        if len(temp_list)>0:
           w2vec_np[i] = p_model[temp_list[0]]
    #w2vec = {word: p_model[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec_np


corpus = X_train_features+X_test_features
print(type(corpus))
print(len(corpus))

#corpus_w2v = [' '.join(x) for x in corpus]
#print(len(corpus_w2v))
#print(corpus_w2v[:10])

w2v_size = 128

model_w2v = Word2Vec(corpus,size=w2v_size,window=5,min_count=5,workers=4)

# 索引字典、词向量字典
index_dict, word_vectors_np= create_dictionaries(model_w2v,w2v_size)
print('The last element of word_vectors_np is: ')
print(word_vectors_np[-1]) 
print("The element dtype of word_vectors_np is: ",word_vectors_np.dtype)       

print("converting to token ids...")
word_id_train = []
for doc in X_train_features:
    word_ids = []
    for word in doc:
        try:
            word_ids.append(index_dict[word])  # 单词转索引数字
        except:
            word_ids.append(0)  # 索引字典里没有的词转为数字0
    word_id_train.append(word_ids)

print("word_id_train's length is: ",len(word_id_train))    

word_id_test = []
for doc in X_test_features:
    word_ids = []
    for word in doc:
        try:
            word_ids.append(index_dict[word])  # 单词转索引数字
        except:
            word_ids.append(0)  # 索引字典里没有的词转为数字0
    word_id_test.append(word_ids)
        
print("word_id_test's length is: ",len(word_id_test))    
    
 

#####################LSTM模型搭建#################

batchSize = 300
lstmUnits = 64
numClasses = 5
learningRate = 0.01

#不加这个函数，会报错：
#valueError: Variable rnn/basic_lstm_cell/weights already exists, disallowed
tf.reset_default_graph()

##占位
input_data = tf.placeholder(tf.int32,[batchSize,maxlen_sentence])
output_labels = tf.placeholder(tf.float32,[batchSize,numClasses])

##用词向量来构建完成模型的三维输入数据
data = tf.Variable(tf.zeros([batchSize, maxlen_sentence, w2v_size]),dtype=tf.float32)
data = tf.nn.embedding_lookup(word_vectors_np,input_data)

##构建隐藏层
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob = 0.75)
initial_state = lstmCell.zero_state(batchSize,tf.float32)
outputs,final_states = tf.nn.dynamic_rnn(lstmCell, data,initial_state=initial_state,dtype=tf.float32)
#outputs,final_states = tf.nn.dynamic_rnn(lstmCell, data,dtype=tf.float32)

##构建输出层
weight_out = tf.Variable(tf.truncated_normal([lstmUnits,numClasses]))
bias_out = tf.Variable(tf.constant(0.1,shape=[numClasses]))
# unpack to list [(batchSize, lstmUnits)..] * maxlen_sentence
outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

prediction = tf.matmul(outputs[-1], weight_out) + bias_out

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_labels))
train_op = tf.train.AdamOptimizer(learningRate).minimize(loss)

###估算正确率的公式搭建
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


###########################训练模型############

#pad sequences
X_train_seq = sequence.pad_sequences(np.array(word_id_train), maxlen=maxlen_sentence)
X_test_seq = sequence.pad_sequences(np.array(word_id_test), maxlen=maxlen_sentence)

test_split_size =0.1
X_train_data,X_CV_data,Y_train_data,Y_CV_data=train_test_split(X_train_seq,Y_train_np,test_size=test_split_size,random_state=42)


from random import randint

def getTrainBatch(X,Y,batch,length):
    temp_max = train_data_size*(1-test_split_size)
    labels = []
    arr = np.zeros([batch, length])
    for i in range(batch):
        num = randint(1,temp_max-2)
        arr[i] = X[num-1:num]
        labels.append(Y[num-1:num].flatten().tolist())       
    return arr, labels

def getCVBatch(X,Y,batch,length):
    temp_max = train_data_size*test_split_size
    labels = []
    arr = np.zeros([batch, length])
    for i in range(batch):
        num = randint(1,temp_max-2)
        arr[i] = X[num-1:num]
        labels.append(Y[num-1:num].flatten().tolist())       
    return arr, labels


sess = tf.Session()
sess.run(tf.global_variables_initializer())

iterations = 15000

for i_itr in range(iterations):
    #if(i_itr%1000 == 0):
     #   start_one = time.clock()    
    
    #Next Batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch(X_train_data,Y_train_data,batchSize,maxlen_sentence);
    #sess.run(train_op, {input_data: nextBatch, output_labels: nextBatchLabels})
    loss_val , _ = sess.run([loss, train_op], {input_data: nextBatch, output_labels: nextBatchLabels})
    if(i_itr%1000 == 0):
        print("Program has finished the training numbers of ",i_itr)
        print("Current loss is: ",loss_val)
    
    ###在交叉验证集上试验正确率
    if(i_itr%1000 == 0):
        val_acc = []
        #val_state = sess.run(lstmCell.zero_state(batchSize, tf.float32))
        for i in range(10):
            x_cv, y_cv = getCVBatch(X_CV_data,Y_CV_data,batchSize,maxlen_sentence);
            feed={input_data: x_cv, output_labels: y_cv}
            batch_acc = sess.run(accuracy,feed_dict=feed)
            val_acc.append(batch_acc)
        print("Current CV accuracy is: {:.3f}".format(np.mean(val_acc)))        
        #stop_one = time.clock()
        #print('The time of Current Training and CV once is: ' + str(stop_one-start_one) + 's')
 

#######################输出测试数据的结构       

test_predict_all = np.ones([len(X_test_seq.tolist()),])*2

range_num = len(X_test_seq.tolist())%batchSize
for i in range(range_num):
    startNum = i*batchSize
    input_test = X_test_seq[startNum:startNum+batchSize]
    if(len(input_test.tolist()) != 300):
        print('input_test lenght is not 300, but is: ',len(input_test.tolist()))
        break
    feed_test={input_data: input_test}
    result = sess.run(prediction,feed_dict=feed_test)
    test_predict = sess.run(tf.argmax(result, 1))
    #print("test_predict type is: ",type(test_predict))
    print('test_predict shape is: ',test_predict.shape)
    #print(test_predict[:299])
    test_predict_all[startNum:startNum+batchSize] = test_predict
    
#make a submission
df_test['Sentiment'] = test_predict_all.reshape(-1,1) 
header = ['PhraseId', 'Sentiment']
df_test.to_csv('./w2v_lstm_tensor.csv', columns=header, index=False, header=True)
       
        
sess.close


##输出程序执行时间
stop = time.clock()

print('program running time is: ' + str(stop-start) + 's')

   





