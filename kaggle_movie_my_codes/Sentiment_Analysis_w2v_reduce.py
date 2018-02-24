# -*- coding: utf-8 -*-
"""
Created on Fri 15:55:30

@author: alex Fu
"""


import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
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

#print(type(X_train_np))
#print(X_train_np.shape)
#print(X_train_np)




########################################降维


df_train_cut = df_train.ix[:150000,['Phrase','Sentiment']]


##提取结果是积极评语的单词库，存为列表
postive_series = df_train_cut.Phrase[df_train_cut.Sentiment>2]

pos_list_token=[]
    
for list_item in postive_series:
    token = word_tokenize(list_item)
    pos_list_token.append(preprocessing(token))

print(len(pos_list_token))    
    
postive_list_token = [y for x in pos_list_token for y in x]

print(len(postive_list_token))


##提取结果是消极评语的单词库，存为列表
negative_series = df_train_cut.Phrase[df_train_cut.Sentiment<2]

neg_list_token=[]
    
for list_item in negative_series:
    token = word_tokenize(list_item)
    neg_list_token.append(preprocessing(token))

print(len(neg_list_token))    
    
negative_list_token = [y for x in neg_list_token for y in x]

print(len(negative_list_token))


##提取结果是中性评语的单词库，存为列表
objective_series = df_train_cut.Phrase[df_train_cut.Sentiment==2]

obj_list_token=[]
    
for list_item in objective_series:
    token = word_tokenize(list_item)
    obj_list_token.append(preprocessing(token))

print(len(obj_list_token))    
    
objective_list_token = [y for x in obj_list_token for y in x]

print(len(objective_list_token))


##词频和卡方来求取每个单词的信息量，并进行选取一定维度的信息量大的单词作为特征

def create_word_scores(posWords,negWords,objWords):

    word_fd = FreqDist() #可统计所有词的词频
    print(type(word_fd))
    cond_word_fd = ConditionalFreqDist() #可统计积极文本中的词频和消极文本中的词频
    for word in posWords:
        #word_fd.inc(word)
        word_fd[word] += 1
        #cond_word_fd['pos'].inc(word)
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        #word_fd.inc(word)
        word_fd[word] += 1
        #cond_word_fd['neg'].inc(word)
        cond_word_fd['neg'][word] += 1
    for word in objWords:
        #word_fd.inc(word)
        word_fd[word] += 1
        #cond_word_fd['neg'].inc(word)
        cond_word_fd['obj'][word] += 1

    pos_word_count = cond_word_fd['pos'].N() #积极词的数量
    neg_word_count = cond_word_fd['neg'].N() #消极词的数量
    obj_word_count = cond_word_fd['obj'].N() #中性词的数量
    total_word_count = pos_word_count + neg_word_count + obj_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        #计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count) 
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count) 
        obj_score = BigramAssocMeasures.chi_sq(cond_word_fd['obj'][word], (freq, obj_word_count), total_word_count) 
        #一个词的信息量等于积极卡方统计量加上消极卡方统计量
        word_scores[word] = pos_score + neg_score + obj_score

    return word_scores #包括了每个词和这个词的信息量

#根据信息量进行倒序排序，选择排名靠前的信息量的词
def find_best_words(word_scores, number):
    #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    #best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_values = sorted(word_scores.items(), key=lambda item:item[1], reverse=True)[:1500]
    best_words = set([w for w, s in best_values])
    return best_words

#根据best_words,选择参数words中信息量丰富的单词成为特征，完成降维
def best_word_features(words, best_words):
    return [ word for word in words if word in best_words]    
  

##################################根据信息量，进行特征选择
  
X_train_word_scores = create_word_scores(postive_list_token, negative_list_token, objective_list_token)
print(len(X_train_word_scores))

X_train_best_words = find_best_words(X_train_word_scores, 1500)
print(len(X_train_best_words))
#print(X_train_best_words)

X_train_features=[]

for item in X_train:
    temp_best_word = best_word_features(item,X_train_best_words)
    X_train_features.append(temp_best_word)
    
X_test_features=[]

for item in X_test:
    temp_best_word = best_word_features(item,X_train_best_words)
    X_test_features.append(temp_best_word)


X_train_feature_np = np.array([' '.join(x) for x in X_train_features])
X_test_feature_np = np.array([' '.join(x) for x in X_test_features])

print(X_train_feature_np.shape)
    
    
#########################################特征提取



####word2vec


corpus = X_train_features+X_test_features
print(type(corpus))
print(len(corpus))

#corpus_w2v = [' '.join(x) for x in corpus]
#print(len(corpus_w2v))
#print(corpus_w2v[:10])


model_w2v = Word2Vec(corpus,size=110,window=5,min_count=5,workers=4)

vocab = model_w2v.wv.vocab
print(type(vocab))
print(len(vocab))
#print(vocab.keys())
#print(model_w2v['good'])

# 得到任意text的vector
def get_vector(text):
    # 建立一个全是0的array
    res =np.zeros([110])
    count = 0
    for word in word_tokenize(text):
        if word in vocab:
            res += model_w2v[word]
            count += 1
    if count != 0:
        res = res/count          
    return res            

print(get_vector('life is like a box of chocolate'))


X_train_w2v_list = [get_vector(x) for x in X_train_feature_np]
print(len(X_train_w2v_list))
X_train_w2v_np = np.array(X_train_w2v_list)
print(X_train_w2v_np.shape)
#print(X_train_w2v_np.T.shape)




############################################模型训练
'''

scaler = prep.StandardScaler().fit(X_train_w2v_np)
X_train_features_transformed = scaler.transform(X_train_w2v_np)



C=[1.0,5.0]

for c_item in C:
    model_svc = svm.SVC(C=c_item,kernel='rbf',decision_function_shape='ovo')
    test_score = cross_val_score(model_svc, X_train_w2v_np, Y_train, cv=4, scoring='accuracy')
    print(test_score)
    print(np.mean(test_score))




LR_classifier =LogisticRegression(multi_class='multinomial',solver='newton-cg',C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)    
test_score = cross_val_score(LR_classifier, X_train_features_transformed, Y_train, cv=3, scoring='accuracy')
print("LR_classifier's test_score are:")
print(test_score)
print(np.mean(test_score))   

'''

scaler = prep.MinMaxScaler().fit(X_train_w2v_np)
X_train_features_transformed = scaler.transform(X_train_w2v_np)


for clf_item, name in (
        (MultinomialNB(),'MultinomialNB'),
        (BernoulliNB(),'BernoulliNB'),
        (GaussianNB(),'GaussianNB')
        ):
    test_score = cross_val_score(clf_item, X_train_features_transformed, Y_train, cv=3, scoring='accuracy')
    print("%s test_score are:" %name)
    print(test_score)
    print(np.mean(test_score)) 




