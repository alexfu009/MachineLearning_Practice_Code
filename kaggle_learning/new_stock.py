# -*- coding: utf-8 -*-
"""
Created on Sun  12:38:55 2017

@author: alex
"""

'''quantifying the quality of predictions in this problem is roc_auc_score.
   auc是Area Under the Curve，ROC曲线下面积。
   auc应大于0.5至少，值越大代表ROC偏离参考曲线的距离越大，即：
   TPR（True Positive Rate）越大，FPR越小，模型更有用。'''
 

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date


data = pd.read_csv('./Combined_News_DJIA.csv')

print(type(data))
#print(data.head())

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

X_train = train[train.columns[2:]]
corpus = X_train.values.flatten().astype(str)
#print(X_train.head(1))
#print(corpus[:1])

X_train = X_train.values.astype(str)
X_train = np.array([' '.join(x) for x in X_train])
X_test = test[test.columns[2:]]
X_test = X_test.values.astype(str)
X_test = np.array([' '.join(x) for x in X_test])

y_train = train['Label'].values
y_test = test['Label'].values

print(type(y_train))

print(corpus[:3])

print(X_train[:1])

print(y_train[:5])


###########  分词  ####################
from nltk.tokenize import word_tokenize

corpus = [word_tokenize(x) for x in corpus]
X_train = [word_tokenize(x) for x in X_train]
X_test = [word_tokenize(x) for x in X_test]
 
print(type(X_train))
                   
#数据预处理：数据清洗（剔除离心点、不可信的样本除去、缺省值的处理）、调权、数据采样、保证样本均衡等。
#本次案例中：进行了以下预处理：小写化、删除停止词、删除数字与符号、lemma
        
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
            # 这一段的用处仅仅是去除python里面byte存str时候留下的标识。。之前数据没处理好，其他case里不会有这个情况
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res
 
          
corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]
            
print(corpus[0:2])

          
           

######## 特征处理 #################

      
from gensim.models.word2vec import Word2Vec

model = Word2Vec(corpus, size=128, window=5, 
                 min_count=5, workers=4)           
           
#print(model['china'])
           
#由于文本本身的量很小，可以把所有的单词的vector拿过来取个平均值：

# 先拿到全部的vocabulary
vocab = model.wv.vocab

# 得到任意text的vector
def get_vector(word_list):
    # 建立一个全是0的array
    res =np.zeros([128])
    count = 0
    for word in word_list:
        if word in vocab:
            res += model[word]
            count += 1
    return res/count    
#此时，我们得到了一个取得任意word list平均vector值得方法：

#print(get_vector(['hello', 'from', 'the', 'other', 'side']))          
           
           
           
wordlist_train = X_train
worllist_test = X_test

X_train = [get_vector(x) for x in X_train]
X_test = [get_vector(x) for x in X_test]           

#print(X_train[50])           
           
           
############# 建立ML模型 #############

from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

params = [0.1,0.5,1,3,5,7,10,12,16,20,25,30,35,40]   

train_scores = []

for param in params:
    clf = SVC(gamma = param)
    train_score = cross_val_score(clf,X_train,y_train,cv=5,scoring='roc_auc')
    train_scores.append(np.mean(train_score))

Cs = [1,3,5,7,10,15,20,25,30,35,40,50,100]   

train_scores_c = []

for C_item in Cs:
    clf = SVC(C = C_item)
    train_score_c = cross_val_score(clf,X_train,y_train,cv=5,scoring='roc_auc')
    train_scores_c.append(np.mean(train_score_c))    
    
import matplotlib.pyplot as plt

plt.figure(num=1)

plt.plot(params,train_scores,color='red',label='Param vs CV AUC Score')

plt.plot(Cs,train_scores_c,color='DarkGreen',label='C vs CV AUC Score')

plt.legend(loc='best')

plt.show()
    



final_clf = SVC(C=10,kernel='rbf',gamma=20)

final_clf.fit(X_train,y_train)

y_pre = final_clf.predict(X_test)

print(type(y_pre))

print(y_pre)

plt.figure(num=2)

plt.scatter(np.arange(20),y_test[0:20],c='red',label='y_test')

plt.scatter(np.arange(20),y_pre[0:20],c='green',label='y_pre')

plt.xlim((0, 20))
plt.ylim((-1, 2))

plt.figure(num=3)

plt.scatter(np.arange(20),y_test[20:40],c='DarkBlue',label='y_test')

plt.scatter(np.arange(20),y_pre[20:40],c='yellow',label='y_pre')

plt.xlim((0, 20))
plt.ylim((-1, 2))

plt.legend(loc = 'best')

plt.show()














    



