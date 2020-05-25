#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[2]:


# Algumas configurações para o matplotlib.

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[18]:


countries = pd.read_csv("countries.csv")


# In[19]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# In[20]:


countries.info()


# In[6]:


countries.Region[0]


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[14]:


# Sua análise começa aqui.
df = pd.read_csv("countries.csv",decimal = ',', header=0, names=new_column_names)

df['Country'] = df.Country.str.strip()
df['Region'] = df.Region.str.strip()

df.head()


# In[8]:


df.info()


# In[9]:


df.Region[0]


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[6]:


def q1():
    unique_regions = df.Region.unique()
    return list(np.sort(unique_regions))

q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[7]:


from sklearn.preprocessing import KBinsDiscretizer

def q2():
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    X = df['Pop_density'].values.reshape(-1,1)
    pop_density_bins = est.fit_transform(X)
    # Since we have 10 bins from 0 to 9, bins >8 (bins == 9) represent countries over 90 percentile
    return  len((df[pop_density_bins == 9]['Country']).unique())

q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[17]:


df[["Region","Climate"]].isnull().sum()


# In[21]:


def q3():
    df_aux = df.copy()
    df_aux["Region"] = df["Region"].astype('category')
    df_aux["Climate"] = df["Climate"].fillna(-1).astype('category')
    
    new_features = pd.get_dummies(df_aux[['Region', 'Climate']])
    new_cols = new_features.shape[1]

    return new_cols

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[9]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[10]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[22]:


def q4():
    numeric_features = list(df.select_dtypes(include=['int64', 'float64']).columns)
    categorical_features = list(df.select_dtypes(include=['object','bool']).columns)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("standart_scaler", StandardScaler()) ])

    #preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)], remainder = 'passthrough')
    preprocessor = ColumnTransformer(
                    transformers=[('cat', 'passthrough', categorical_features),
                                 ('num', numeric_transformer, numeric_features)])
    
    pipeline_transformation = preprocessor.fit_transform(df)
    df_test_country = pd.DataFrame(test_country, index= list(df.columns))
    
    test_country_transformation = preprocessor.transform(df_test_country.transpose())
    df_test_country_transformation = pd.DataFrame(test_country_transformation , columns = df.columns)
    
    return round(df_test_country_transformation.loc[0,'Arable'],3)

q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[23]:


def pd_of_stats(df,col):
    '''
    Create a dataframe of descriptive Statistics
    '''
    stats = dict()
    stats['Mean']  = df[col].mean()
    stats['Std']   = df[col].std()
    stats['Var'] = df[col].var()
    stats['Kurtosis'] = df[col].kurtosis()
    stats['Skewness'] = df[col].skew()
    stats['CoefVar'] = stats['Std'] / stats['Mean']
    
    return pd.DataFrame(stats, index = col).T.round(2)


def pd_of_stats_quantile(df,col):
    '''
    Create a dataframe of quantile Statistics
    '''
    df_no_na = df[col].dropna()
    stats_q = dict()

    stats_q['Min'] = df[col].min()
    label = {0.25:"Q1", 0.5:'Median', 0.75:"Q3"}
    for percentile in np.array([0.25, 0.5, 0.75]):
        stats_q[label[percentile]] = df_no_na.quantile(percentile)
    stats_q['Max'] = df[col].max()
    stats_q['Range'] = stats_q['Max']-stats_q['Min']
    stats_q['IQR'] = stats_q['Q3']-stats_q['Q1']

    return pd.DataFrame(stats_q, index = col).T.round(2)    


# In[52]:


pd_of_stats(df,['Net_migration'])


# In[53]:


pd_of_stats_quantile(df,['Net_migration'])


# In[82]:


sns.distplot(df.Net_migration.dropna())


# In[24]:


def q5():
    var = df.Net_migration.dropna().copy()
    q1, q3 = var.quantile([0.25, 0.75])
    iqr = q3 - q1
    cutoff = 1.5*iqr
    lowerbound, upperbound = q1 - cutoff, q3 + cutoff
    print(f'Lower Bound: {lowerbound}, Upper Bound:{upperbound}')
    n_outliers = len(var[~var.between(lowerbound, upperbound, inclusive=True)])
    outliers_abaixo = len(var[var < lowerbound])
    outliers_acima  = len(var[var > upperbound])

    percent_outlier = n_outliers/len(var)
    print(f'Outliers/Observations: {round(percent_outlier*100,3)}, Outliers: {n_outliers}, Below threshold: {outliers_abaixo}, Above threshold: {outliers_acima}')
    return (outliers_abaixo, outliers_acima, False)

q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[11]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[28]:


print('Groups :', newsgroups.target_names)
print('Length :',len(newsgroups.data))


# In[34]:


#first entry
print(f'{newsgroups.data[0]}')


# In[30]:


newsgroups.target_names[newsgroups.target[0]]


# In[92]:


def q6():
    count_vect = CountVectorizer()
    newsgroups_counts = count_vect.fit_transform(newsgroups.data)
    print('Shape', newsgroups_counts.shape)
    #get the position that corresponds to phone
    phone_idx = count_vect.vocabulary_.get(u"phone")
    tf_phone = np.sum(newsgroups_counts[:,phone_idx]).item()
    n_doc_phone = np.sum(newsgroups_counts[:,phone_idx]!=0)
    print(f'Number of times the word phone appeared: {tf_phone}')
    print(f'Number of documents the word phone appeared: {n_doc_phone}')
    
    return tf_phone

q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[12]:


def q7():
    tfidf_vectorizer = TfidfVectorizer()
    newsgroups_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroups.data)
    
    phone_idx = tfidf_vectorizer.vocabulary_.get(u"phone")
    tf_idf_phone = np.sum(newsgroups_tfidf_vectorized[:,phone_idx]).item()
    
    print('TF-IDF', tf_idf_phone)  
    return round(tf_idf_phone,3)

q7()


# In[ ]:




