# -*- coding: utf-8 -*-
"""
Created on Fri 15:55:30

@author: alex Fu
"""


import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec


from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing as prep
from sklearn.naive_bayes import BernoulliNB, MultinomialNB,GaussianNB


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

print(type(Y_train))
print(Y_train.shape)
#print(Y_train)


'''
############爆米花数据
train = pd.read_csv('./popcorn/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test = pd.read_csv('./popcorn/testData.tsv', header=0, delimiter="\t", quoting=3)

print(train.head())
print(train.shape)

print(test.head())
print(test.shape)

X_train_pre = train.ix[:24000,['review']].values.flatten()
print(type(X_train_pre))
print(X_train_pre.shape)

X_test_pre = test.ix[:20000,['review']].values.flatten()
print(type(X_test_pre))
print(X_test_pre.shape)

Y_train = train.ix[:24000,['sentiment']].values.flatten()
'''


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

  

    
import gensim

LabeledSentence = gensim.models.doc2vec.LabeledSentence 

def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v,[label]))
    return labelized
    

X_train = labelizeReviews(X_train, 'TRAIN')
X_test = labelizeReviews(X_test,'TEST')




#X_train_np = np.array([' '.join(x) for x in X_train])
#X_test_np = np.array([' '.join(x) for x in X_test])  
#print(type(X_train_np))
#print(X_train_np.shape)
#print(X_train_np)




#########################################特征提取


####doc2vec


corpus = X_train+X_test
print(type(corpus))
print(len(corpus))

size=128

model_dm = gensim.models.Doc2Vec(size=size, alpha=0.025, min_alpha=0.025,window=10, negative=5,min_count=1, workers=3)
model_dbow = gensim.models.Doc2Vec(dm=0,size=size,alpha=0.025, min_alpha=0.025, window=10, negative=5,min_count=1, workers=3)


#build vocab over all reviews
model_dm.build_vocab(corpus)
model_dbow.build_vocab(corpus)

#X_train = np.array(X_train)

for epoch in range(10):
    model_dm.train(X_train,total_examples=model_dm.corpus_count,epochs=model_dm.iter)
    model_dbow.train(X_train,total_examples=model_dbow.corpus_count,epochs=model_dbow.iter)
    model_dm.alpha -= 0.002  # decrease the learning rate
    model_dm.min_alpha = model_dm.alpha  # fix the learning rate, no decay
    model_dbow.alpha -= 0.002  # decrease the learning rate
    model_dbow.min_alpha = model_dbow.alpha 
    #model_dm.train(random.shuffle(X_train),total_examples=model_dm.corpus_count,epochs=model_dm.iter)
    #model_dbow.train(random.shuffle(X_train),total_examples=model_dm.corpus_count,epochs=model_dm.iter)

#model_dm.train(X_train,total_examples=model_dm.corpus_count,epochs=model_dm.iter)
#model_dbow.train(X_train,total_examples=model_dbow.corpus_count,epochs=model_dbow.iter)
 
#print(type(model_dm.wv.vocab))
#print(list(model_dm.wv.vocab.values())[:150])

#z = X_train[0]
#print(z.words)
#print(type(model_dm.docvecs[z.tags[0]]))
#print(model_dm.docvecs[z.tags[0]].shape)
#print(model_dm.docvecs[z.tags[0]])
   

sims = model_dm.docvecs.most_similar(0)
print(sims)
# get similarity between doc1 and doc2 in the training data
simss = model_dm.docvecs.similarity(1,2)
print(simss)
  
def getVecs(model,corpus,size):
    vecs = [model_dm.docvecs[z.tags[0]].reshape((1,size)) for z in corpus]
    return np.concatenate(vecs)
    
train_vecs_dm = getVecs(model_dm,X_train,size)
train_vecs_dbow = getVecs(model_dbow,X_train,size)

train_vecs = np.hstack((train_vecs_dm,train_vecs_dbow))

#print(train_vecs[:1,:])
print(train_vecs.shape)



############################################模型训练


scaler = prep.StandardScaler().fit(train_vecs)
X_train_features_transformed = scaler.transform(train_vecs)

'''

C=1.0
model_svc = svm.SVC(C=C,kernel='rbf',decision_function_shape='ovo')

test_score = cross_val_score(model_svc, train_vecs, Y_train, cv=4, scoring='accuracy')
    
print(test_score)
print(np.mean(test_score))

'''

LR_classifier =LogisticRegression(multi_class='multinomial',solver='newton-cg',C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)    
test_score = cross_val_score(LR_classifier, X_train_features_transformed, Y_train, cv=3, scoring='accuracy')
print("LR_classifier's test_score are:")
print(test_score)
print(np.mean(test_score))   

'''


scaler = prep.MinMaxScaler().fit(train_vecs)
X_train_features_transformed = scaler.transform(train_vecs)


for clf_item, name in (
        (MultinomialNB(),'MultinomialNB'),
        (BernoulliNB(),'BernoulliNB'),
        (GaussianNB(),'GaussianNB')
        ):
        test_score = cross_val_score(clf_item, X_train_features_transformed, Y_train, cv=3, scoring='accuracy')
        print("%s test_score are:" %name)
        print(test_score)
        print(np.mean(test_score)) 


'''

