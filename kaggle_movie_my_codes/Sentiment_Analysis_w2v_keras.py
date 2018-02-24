# -*- coding: utf-8 -*-
"""
Created on Mon 23:39:53

@author: alex
"""

import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM

from sklearn.cross_validation import train_test_split



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

X_train_pre = df_train.ix[:155000,['Phrase']].values.flatten()
print(type(X_train_pre))
print(X_train_pre.shape)

#X_test_pre = df_test['Phrase'].values
X_test_pre = df_test.ix[:,['Phrase']].values.flatten()
print(type(X_test_pre))
print(X_test_pre.shape)

Y_train = df_train.ix[:155000,['Sentiment']].values
num_labels = len(np.unique(Y_train))
#print(type(Y_train))
#print(Y_train.shape)
#print(Y_train)


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

# 创建词语字典，并返回word2vec模型中词语的索引，词向量
def create_dictionaries(p_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.wv.vocab.keys(), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: p_model[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec


corpus = X_train_features+X_test_features
print(type(corpus))
print(len(corpus))

#corpus_w2v = [' '.join(x) for x in corpus]
#print(len(corpus_w2v))
#print(corpus_w2v[:10])

w2v_size = 128

model_w2v = Word2Vec(corpus,size=w2v_size,window=5,min_count=5,workers=4)

# 索引字典、词向量字典
index_dict, word_vectors= create_dictionaries(model_w2v)

        

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
    
    
#################嵌入层的权重我使用自己训练的词向量

print("Setting up Arrays for Keras Embedding Layer...")
n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
embedding_weights = np.zeros((n_symbols, w2v_size))  # 创建一个n_symbols * 100的0矩阵
for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
    embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）


################################模型训练

#pad sequences
X_train_seq = sequence.pad_sequences(np.array(word_id_train), maxlen=maxlen_sentence)
X_test_seq = sequence.pad_sequences(np.array(word_id_test), maxlen=maxlen_sentence)
y_train_enc = np_utils.to_categorical(Y_train, num_labels)

X_train_data,X_CV_data,Y_train_data,Y_CV_data=train_test_split(X_train_seq,y_train_enc,test_size=0.1,random_state=42)


#LSTM
print("fitting LSTM ...")
EMBEDDING_SIZE=w2v_size
HIDDEN_LAYER_SIZE = 64

model = Sequential()
#model.add(Embedding(n_symbols, EMBEDDING_SIZE, weights=[embedding_weights],input_length = maxlen_sentence,dropout=0.2,trainable=False))
model.add(Embedding(n_symbols, EMBEDDING_SIZE,input_length = maxlen_sentence,dropout=0.2))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_data, Y_train_data, nb_epoch=6, batch_size=256, verbose=1)

score, acc = model.evaluate(X_CV_data, Y_CV_data,batch_size=64)
print('Test score:', score)
print('Test accuracy:', acc)

test_predict = model.predict_classes(X_test_seq)
#make a submission
df_test['Sentiment'] = test_predict.reshape(-1,1) 
header = ['PhraseId', 'Sentiment']
df_test.to_csv('./w2v_keras.csv', columns=header, index=False, header=True)


##输出程序执行时间
stop = time.clock()

print('program running time is: ' + str(stop-start) + 's')





