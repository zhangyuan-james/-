
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 
from sklearn import preprocessing
import pandas as pd
def myfun(x):
    try:
        return km.predict(model[x].reshape(1,-1))[0]
    except:
        return -1


# In[2]:


data=pd.read_excel(r'e:\python\jd\book\jdb.xlsx')
print("载入成功")
data.head()


# In[1]:


f=open(r"e:/python/jd/word2.txt",encoding="UTF-8")
content=f.readlines()
wd=[]
st=[]
for i in content[1:]:
    i=i.split()
    wd.append(i[0])
    st.append(i[1:])


# In[3]:


dt=data["分词结果"]
dt=dt.tolist()
st=[]
for i in dt:
    i=str(i)
    if len(i)>0:
        i=i.strip()
        st.append(i.split())
import logging
import numpy as np
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences=word2vec.Text8Corpus(st)
model=word2vec.Word2Vec(st, sg=1, size=50,  window=5,  min_count=1,  negative=3, sample=0.001, hs=1, iter=5,workers=4)
model.wv.save_word2vec_format(r'e:/python/jd/word2.txt', binary=False)
print("完成向量化")


# In[4]:


f=open(r"e:/python/jd/word2.txt",encoding="UTF-8")
content=f.readlines()
wd=[]
st=[]
for i in content[1:]:
    i=i.split()
    wd.append(i[0])
    st.append(i[1:])
import numpy as np
from sklearn.cluster import KMeans
km= KMeans(n_clusters=10)#构造聚类器
km.fit(st)#聚类
print('完成聚类')


# In[6]:


data


# In[33]:


useful=["级别","情感分数","次数","长度","否定词个数","程度副词个数","情感词个数","tfidf最大值单词","tfidf最大值","标签"]
data=data[useful]
col=data.columns
data=np.array(data)
data=data.tolist()
for i in data:
    i[7]=myfun(i[7])
data=np.array(data)
data=pd.DataFrame(data)
data.columns=col
data.to_excel(r"e:\python\jd\book\addword2.xlsx")
print("分析完成")


# In[31]:


km.predict(model['上市'].reshape(1,-1))[0]


# In[8]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE


# In[10]:


df=pd.read_excel(r"e:\python\jd\book\addword2.xlsx")


# In[69]:



df=df.iloc[:5000,:]


# In[80]:


df=pd.read_excel(r"e:\python\jd\book\addword2.xlsx")
df=df[['情感分数','长度','次数','情感词个数','tf-idf最大值单词','tf-idf最大值','标签']]
df.columns=[['qinggan','len','times','qinggannum','tfidfword','tfidf','标签']]
df=df.iloc[:5000,:]
y_set=df['标签']
x_set=df.drop('标签',1)
xtrain, xtest, ytrain, ytest = train_test_split(x_set, y_set, test_size=0.7, random_state=0)
col=xtrain.columns.values.tolist()
#xtrain, ytrain=SMOTE(kind='borderline2').fit_sample(xtrain, ytrain)
scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain=pd.DataFrame(scaler.transform(xtrain))
xtest=pd.DataFrame(scaler.transform(xtest))
x_set=pd.DataFrame(scaler.transform(x_set))
xtrain.columns=col
xtest.columns=col
x_set.columns=col
dtrain=xgb.DMatrix(xtrain,label=ytrain)


# In[13]:


len(xtest)


# In[81]:


dtest=xgb.DMatrix(x_set)
for i in range(1):
    params={'booster':'gbtree',
'objective': 'binary:logistic',
'eval_metric': 'auc',
'max_depth':10,
'lambda':0.82,
'alpha':0.4,
'subsample':0.93,
'colsample_bytree':0.6,
'min_child_weight':1,
'eta': 0.63,
'seed':7,
'nthread':8,
'silent':1,
'scale_pos_weight':10
}
    ##watchlist = [(dtrain,'train')]
    bst=xgb.train(params,dtrain,num_boost_round=17)##evals=watchlist)
    ypred=bst.predict(dtest)
    y_pred = (ypred >= 0.39)*1
    print( i,'Recall: %.4f' % metrics.recall_score( y_set,y_pred))
    print ('Precesion: %.4f' %metrics.precision_score( y_set,y_pred))
    print ('Precesion: %.4f' %metrics.accuracy_score(y_set,y_pred))
    print ('Precesion: %.4f' %metrics.f1_score(y_set,y_pred))
    print(metrics.confusion_matrix(y_set,y_pred),'\n')


# In[110]:


y_pred=pd.DataFrame(y_pred)
y_pred.to_clipboard()


# In[65]:


from xgboost import plot_importance
plt=plot_importance(bst)
show(plt


# In[75]:


for i in range(200):
    y_pred = (ypred >= 0.3+0.001*i)*1
    print (0.3+0.001*i,metrics.f1_score( ytest,y_pred))

