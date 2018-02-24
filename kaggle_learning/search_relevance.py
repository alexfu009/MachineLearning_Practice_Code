# -*- coding: utf-8 -*-
"""
Created on Sunday  21:17:07

@author: alex
"""



import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

##############读入数据

df_train = pd.read_csv('./input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('./input/test.csv', encoding="ISO-8859-1")

df_desc = pd.read_csv('./input/product_descriptions.csv')

#print(df_train.head())

#print(df_desc.head())

df_train = df_train[:5000]
df_test = df_test[:5000]
df_desc = df_desc[:5000]

#####################文本预处理

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

print(df_all.head())

print(df_all.shape)

df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')

print(df_all.head())



#######################数据清洗


stemmer = SnowballStemmer('english')

def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])
 
    
#为了计算『关键词』的有效性，我们可以naive地直接看『出现了多少次』 
def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())
    
#接下来，把每一个column都跑一遍，以清洁所有的文本内容
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))

df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))

df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))




#####################文本特征提取

#使用Levenshtein，词语的距离
import Levenshtein
print(Levenshtein.ratio('hello', 'hello world'))

#把search_term和product_title进行比较：
df_all['dist_in_title'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_title']), axis=1)

#对产品介绍进行比较：
df_all['dist_in_desc'] = df_all.apply(lambda x:Levenshtein.ratio(x['search_term'],x['product_description']), axis=1)




####TF-IDF

df_all['all_texts']=df_all['product_title'] + ' . ' + df_all['product_description'] + ' . '

print(df_all['all_texts'][:5])


from gensim.utils import tokenize
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(list(tokenize(x, errors='ignore')) for x in df_all['all_texts'].values)
print(dictionary)

#myCorpus类：扫描所有的语料，并且转化成简单的单词的个数计算。（Bag-of-Words）
class MyCorpus(object):
    def __iter__(self):
        for x in df_all['all_texts'].values:
            yield dictionary.doc2bow(list(tokenize(x, errors='ignore')))

# 作用：是为了内存friendly。面对大量corpus数据时，你直接存成一个list，会使得整个运行变得很慢。
# 搞成这样，一次只输出一组。但本质上依旧长得跟 [['sentence', '1'], ['sentence', '2'], ...]一样

corpus = MyCorpus()

from gensim.models.tfidfmodel import TfidfModel
tfidf = TfidfModel(corpus)

print(tfidf[dictionary.doc2bow(list(tokenize('hello world and good moring', errors='ignore')))])

from gensim.similarities import MatrixSimilarity


def to_tfidf(text):
    res = tfidf[dictionary.doc2bow(list(tokenize(text, errors='ignore')))]
    return res

# 然后，我们创造一个cosine similarity的比较方法
def cos_sim(text1, text2):
    tfidf1 = to_tfidf(text1)
    tfidf2 = to_tfidf(text2)
    index = MatrixSimilarity([tfidf1],num_features=len(dictionary))
    sim = index[tfidf2]
    # 本来sim输出是一个array，我们不需要一个array来表示，
    # 所以我们直接cast成一个float
    return float(sim[0])

text1 = 'hello world'
text2 = 'hello world and good morning'
print(cos_sim(text1, text2))

#生成我们的新特征值，由TF-IDF带来的相似度
df_all['tfidf_cos_sim_in_title'] = df_all.apply(lambda x: cos_sim(x['search_term'], x['product_title']), axis=1)

print(df_all['tfidf_cos_sim_in_title'][:5])

df_all['tfidf_cos_sim_in_desc'] = df_all.apply(lambda x: cos_sim(x['search_term'], x['product_description']), axis=1)


print(df_all['tfidf_cos_sim_in_desc'][:5])




####word2vec


import nltk
# nltk也是自带一个强大的句子分割器。
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

print(tokenizer.tokenize(df_all['all_texts'].values[0]))

sentences = [tokenizer.tokenize(x) for x in df_all['all_texts'].values]

sentences = [y for x in sentences for y in x]

print(len(sentences))

from nltk.tokenize import word_tokenize
w2v_corpus = [word_tokenize(x) for x in sentences]


from gensim.models.word2vec import Word2Vec

model = Word2Vec(w2v_corpus, size=128, window=5, min_count=5, workers=4)
#这时候，每个单词都可以像查找字典一样，读出他们的w2v坐标了：

#print(model['right'])

#平均化一个句子的w2v向量，算作整个text的平均vector：

# 先拿到全部的vocabulary
vocab = model.wv.vocab

# 得到任意text的vector
def get_vector(text):
    # 建立一个全是0的array
    res =np.zeros([128])
    count = 0
    for word in word_tokenize(text):
        if word in vocab:
            res += model[word]
            count += 1
    return res/count            

print(get_vector('life is like a box of chocolate'))

#同理，需要计算两个text的平均w2v的cosine similarity

from scipy import spatial


def w2v_cos_sim(text1, text2):
    try:
        w2v1 = get_vector(text1)
        w2v2 = get_vector(text2)
        sim = 1 - spatial.distance.cosine(w2v1, w2v2)
        return float(sim)
    except:
        return float(0)
# 这里加个try exception，以防我们得到的vector是个[0,0,0,...]
print(w2v_cos_sim('hello world', 'hello from the other side'))

#跟刚刚TFIDF一样，计算一下search term在product title和product description中的cosine similarity

df_all['w2v_cos_sim_in_title'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_title']), axis=1)
df_all['w2v_cos_sim_in_desc'] = df_all.apply(lambda x: w2v_cos_sim(x['search_term'], x['product_description']), axis=1)

print(df_all.head(2))

#我们把不能被『机器学习模型』处理的column给drop掉
df_all = df_all.drop(['search_term','product_title','product_description','all_texts'],axis=1)




#################重塑训练/测试集


df_train = df_all.loc[df_train.index]
df_test = df_all.loc[df_test.index]

#记录下测试集的id ,留着上传的时候 能对的上号
test_ids = df_test['id']


y_train = df_train['relevance'].values

#把原集中的label给删去,否则就是cheating了
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values




##################模型选择，超参数调整

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score

#用CV结果保证公正客观性；并调试不同的alpha值

params = [1,3,5,6,7,8,9,10]
test_scores = []
for param in params:
    clf = RandomForestRegressor(n_estimators=30, max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='mean_squared_error'))
    test_scores.append(np.mean(test_score))


import matplotlib.pyplot as plt

plt.figure(num=1)
plt.plot(params, test_scores)
plt.title("Param vs CV Error")
plt.show()


rf = RandomForestRegressor(n_estimators=30, max_depth=6)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(y_pred)

pd.DataFrame({"id": test_ids, "relevance": y_pred}).to_csv('submission.csv',index=False)











