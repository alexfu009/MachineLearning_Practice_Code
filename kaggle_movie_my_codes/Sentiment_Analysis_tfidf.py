# -*- coding: utf-8 -*-
"""
Created on Fri 15:55:30

@author: alex Fu
"""


import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB


################################读入数据

df_train = pd.read_csv('./movie/train.tsv', sep='\t', header=0)
df_test = pd.read_csv('./movie/test.tsv', sep='\t', header=0)

#print(df_train.head())
#print(df_train.shape)

#print(df_test.head())
#print(df_test.shape)

X_train_pre = df_train.ix[:150000,['Phrase']].values.flatten()
print(type(X_train_pre))
print(X_train_pre.shape)

#X_test_pre = df_test['Phrase'].values
X_test_pre = df_test.ix[:60000,['Phrase']].values.flatten()
print(type(X_test_pre))
print(X_test_pre.shape)


Y_train = df_train.ix[:150000,['Sentiment']].values.flatten()

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

    
X_train=[]
    
for list_item in X_train_pre:
    token = word_tokenize(list_item)
    X_train.append(preprocessing(token))

#print(X_train)    

X_test=[]
    
for list_item in X_test_pre:
    token = word_tokenize(list_item)
    X_test.append(preprocessing(token))




X_train_np = np.array([' '.join(x) for x in X_train])
X_test_np = np.array([' '.join(x) for x in X_test])

print(type(X_train_np))
print(X_train_np.shape)
#print(X_train_np)




#########################################特征提取

####TD-IDF

feature_extraction = TfidfVectorizer()
X_train_tfidf = feature_extraction.fit_transform(X_train_np)
X_test_tfidf = feature_extraction.transform(X_test_np)



#print(X_train_tfidf)
print(type(X_train_tfidf))
print(X_train_tfidf.shape)







############################################模型训练

'''



#####LogisticRegression

LR_classifier =LogisticRegression(multi_class='multinomial',solver='newton-cg',C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)    
test_score = cross_val_score(LR_classifier, X_train_tfidf, Y_train, cv=3, scoring='accuracy')
print("LR_classifier's test_score are:")
print(test_score)
print(np.mean(test_score))  

'''

#####naive bayes



for clf_item, name in (
        (MultinomialNB(),'MultinomialNB'),
        (BernoulliNB(),'BernoulliNB'),
        ):
    test_score = cross_val_score(clf_item, X_train_tfidf, Y_train, cv=3, scoring='accuracy')
    print("%s test_score are:" %name)
    print(test_score)
    print(np.mean(test_score))  







