# -*- coding: utf-8 -*-
"""
Created on Mon 09:55:20

@author: alex Fu
"""


import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures

from gensim.models.word2vec import Word2Vec

from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing as prep

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


import gensim

LabeledSentence = gensim.models.doc2vec.LabeledSentence 

def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v,[label]))
    return labelized
    

X_train_d2v = labelizeReviews(X_train, 'TRAIN')
X_test_d2v = labelizeReviews(X_test,'TEST')



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

#####TextBlob

polarity_train_list = [] 
subjectivity_train_list = []

for item in X_train_feature_np:
    testimonial = TextBlob(item)
    polarity_train_list.append(testimonial.sentiment.polarity)
    subjectivity_train_list.append(testimonial.sentiment.subjectivity)

print('length of polarity_train_list is:')    
print(len(polarity_train_list))
print("polarity_train_list's first two elements are:")
print(polarity_train_list[:2])
print("subjectivity_train_list's length is:") 
print(len(subjectivity_train_list))

polarity_train_np = np.array(polarity_train_list)
subjectivity_train_np = np.array(subjectivity_train_list)

polarity_test_list = [] 
subjectivity_test_list = []
    
for item in X_test_feature_np:
    testimonial = TextBlob(item)
    polarity_test_list.append(testimonial.sentiment.polarity)
    subjectivity_test_list.append(testimonial.sentiment.subjectivity)

print("polarity_test_list's length is:")    
print(len(polarity_test_list))
print("subjectivity_test_list's length is:") 
print(len(subjectivity_test_list))

polarity_test_np = np.array(polarity_test_list)
subjectivity_test_np = np.array(subjectivity_test_list)
   

 
####TD-IDF

feature_extraction = TfidfVectorizer()
X_train_tfidf = feature_extraction.fit_transform(X_train_feature_np)
X_test_tfidf = feature_extraction.transform(X_test_feature_np)

#print(X_train_tfidf)
print(type(X_train_tfidf))
print(X_train_tfidf.shape)


X_train_tfidf_ave = X_train_tfidf.mean(axis=1)
print("X_train_tfidf_ave's shape is:")
print(X_train_tfidf_ave.shape)

X_test_tfidf_ave = X_test_tfidf.mean(axis=1)
print("X_test_tfidf_ave's shape is:")
print(X_test_tfidf_ave.shape)



####word2vec-CBOW  word2vec-Skip-Gram


corpus = X_train_features+X_test_features
print(type(corpus))
print(len(corpus))

#corpus_w2v = [' '.join(x) for x in corpus]
#print(len(corpus_w2v))
#print(corpus_w2v[:10])


model_w2v_cbow = Word2Vec(corpus,sg=0,size=128,window=8,min_count=2,workers=4)
model_w2v_skip = Word2Vec(corpus,sg=1,size=128,window=8,min_count=2,workers=4)

vocab_cbow = model_w2v_cbow.wv.vocab
print(type(vocab_cbow))
print(len(vocab_cbow))
#print(vocab.keys())
#print(model_w2v['good'])
vocab_skip = model_w2v_skip.wv.vocab
print(type(vocab_skip))
print(len(vocab_skip))


# 得到任意text的vector
def get_vector(text,vocab,model_w2v):
    # 建立一个全是0的array
    res =np.zeros([128])
    count = 0
    for word in word_tokenize(text):
        if word in vocab:
            res += model_w2v[word]
            count += 1
    if count != 0:
        res = res/count          
    return res            

#print(get_vector('life is like a box of chocolate'))


X_train_w2v_cbow = [get_vector(x,vocab_cbow,model_w2v_cbow) for x in X_train_feature_np]
print("X_train_w2v_cbow's length is:")
print(len(X_train_w2v_cbow))
X_train_w2v_cbow_np = np.array(X_train_w2v_cbow)
print("X_train_w2v_cbow_np's shape is:")
print(X_train_w2v_cbow_np.shape)
X_train_w2v_cbow_ave = X_train_w2v_cbow_np.mean(axis=1)
print("X_train_w2v_cbow_ave's shape is:")
print(X_train_w2v_cbow_ave.shape)

X_train_w2v_skip = [get_vector(x,vocab_skip,model_w2v_skip) for x in X_train_feature_np]
print("X_train_w2v_skip's length is:")
print(len(X_train_w2v_skip))
X_train_w2v_skip_np = np.array(X_train_w2v_skip)
print("X_train_w2v_skip_np's shape is:")
print(X_train_w2v_skip_np.shape)
X_train_w2v_skip_ave = X_train_w2v_skip_np.mean(axis=1)
print("X_train_w2v_skip_ave's shape is:")
print(X_train_w2v_skip_ave.shape)




X_test_w2v_cbow = [get_vector(x,vocab_cbow,model_w2v_cbow) for x in X_test_feature_np]
print("X_test_w2v_cbow's length is:")
print(len(X_test_w2v_cbow))
X_test_w2v_cbow_np = np.array(X_test_w2v_cbow)
print("X_test_w2v_cbow_np's shape is:")
print(X_test_w2v_cbow_np.shape)
X_test_w2v_cbow_ave = X_test_w2v_cbow_np.mean(axis=1)
print("X_test_w2v_cbow_ave's shape is:")
print(X_test_w2v_cbow_ave.shape)


X_test_w2v_skip = [get_vector(x,vocab_skip,model_w2v_skip) for x in X_test_feature_np]
print("X_test_w2v_skip's length is:")
print(len(X_test_w2v_skip))
X_test_w2v_skip_np = np.array(X_test_w2v_skip)
print("X_test_w2v_skip_np's shape is:")
print(X_test_w2v_skip_np.shape)
X_test_w2v_skip_ave = X_test_w2v_skip_np.mean(axis=1)
print("X_test_w2v_skip_ave's shape is:")
print(X_test_w2v_skip_ave.shape)



####doc2vec


corpus_d2v = X_train_d2v+X_test_d2v
print(type(corpus_d2v))
print(len(corpus_d2v))

size=128

model_dm = gensim.models.Doc2Vec(size=size, alpha=0.025, min_alpha=0.025,window=10,min_count=1, workers=3)
model_dbow = gensim.models.Doc2Vec(dm=0,size=size,alpha=0.025, min_alpha=0.025, window=10,min_count=1, workers=3)


#build vocab over all reviews
model_dm.build_vocab(corpus_d2v)
model_dbow.build_vocab(corpus_d2v)

for epoch in range(10):
    model_dm.train(X_train_d2v,total_examples=model_dm.corpus_count,epochs=model_dm.iter)
    model_dbow.train(X_train_d2v,total_examples=model_dbow.corpus_count,epochs=model_dbow.iter)
    model_dm.alpha -= 0.002  # decrease the learning rate
    model_dm.min_alpha = model_dm.alpha  # fix the learning rate, no decay
    model_dbow.alpha -= 0.002  # decrease the learning rate
    model_dbow.min_alpha = model_dbow.alpha 


sims = model_dm.docvecs.most_similar(0)
print(sims)
# get similarity between doc1 and doc2 in the training data
simss = model_dm.docvecs.similarity(1,2)
print(simss)
  
def getVecs(model,corpus,size):
    vecs = [model_dm.docvecs[z.tags[0]].reshape((1,size)) for z in corpus]
    return np.concatenate(vecs)
    
train_vecs_dm = getVecs(model_dm,X_train_d2v,size)
train_vecs_dm_ave = train_vecs_dm.mean(axis=1)
print("train_vecs_dm_ave's shape is:")
print(train_vecs_dm_ave.shape)

train_vecs_dbow = getVecs(model_dbow,X_train_d2v,size)
train_vecs_dbow_ave = train_vecs_dbow.mean(axis=1)
print("train_vecs_dbow_ave's shape is:")
print(train_vecs_dbow_ave.shape)


test_vecs_dm = getVecs(model_dm,X_test_d2v,size)
test_vecs_dm_ave = test_vecs_dm.mean(axis=1)
print("test_vecs_dm_ave's shape is:")
print(test_vecs_dm_ave.shape)

test_vecs_dbow = getVecs(model_dbow,X_test_d2v,size)
test_vecs_dbow_ave = test_vecs_dbow.mean(axis=1)
print("test_vecs_dbow_ave's shape is:")
print(test_vecs_dbow_ave.shape)



#########################特征构建

length_x_train = len(X_train)
print("X_train features concatenate :")
print(length_x_train)

X_train_features = np.hstack((polarity_train_np.reshape((length_x_train,1)),
                              subjectivity_train_np.reshape((length_x_train,1)), 
                              X_train_tfidf_ave.reshape(length_x_train,1), 
                              X_train_w2v_cbow_ave.reshape(length_x_train,1), 
                              X_train_w2v_skip_ave.reshape(length_x_train,1),
                              train_vecs_dm_ave.reshape(length_x_train,1),
                              train_vecs_dbow_ave.reshape(length_x_train,1)))

print("X_train_features's shape is")
print(X_train_features.shape)
print(X_train_features[:2,:])

max_num=1000
min_max_scaler = prep.MinMaxScaler(feature_range=(0, max_num))
X_train_minmax_f = min_max_scaler.fit_transform(X_train_features)
X_train_minmax = np.rint(X_train_minmax_f)

################################模型训练

maxlen_sentence=7
#pad sequences
X_train_seq = sequence.pad_sequences(X_train_minmax, maxlen=maxlen_sentence)
#X_test_seq = sequence.pad_sequences(np.array(word_id_test), maxlen=maxlen_sentence)
y_train_enc = np_utils.to_categorical(Y_train, num_labels)

X_train_data,X_CV_data,Y_train_data,Y_CV_data=train_test_split(X_train_seq,y_train_enc,test_size=0.1,random_state=42)


#LSTM
print("fitting LSTM ...")
EMBEDDING_SIZE=128
HIDDEN_LAYER_SIZE = 64

model = Sequential()
#model.add(Embedding(n_symbols, EMBEDDING_SIZE, weights=[embedding_weights],input_length = maxlen_sentence,dropout=0.2,trainable=False))
model.add(Embedding(max_num+1, EMBEDDING_SIZE,input_length = maxlen_sentence,dropout=0.2))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_data, Y_train_data, nb_epoch=6, batch_size=256, verbose=1)

score, acc = model.evaluate(X_CV_data, Y_CV_data,batch_size=64)
print('Test score:', score)
print('Test accuracy:', acc)




##输出程序执行时间
stop = time.clock()

print('program running time is: ' + str(stop-start) + 's')






