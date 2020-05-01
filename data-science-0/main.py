#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# In[4]:


black_friday.describe().T


# In[5]:


black_friday.info()


# In[6]:


df_info = pd.DataFrame({'nome':black_friday.columns,'types':black_friday.dtypes,'na': black_friday.isna().sum(),'na%':(black_friday.isna().sum()/len(black_friday))*100})
df_info


# In[7]:


black_friday.shape


# In[8]:


black_friday.groupby(['Gender','Age']).agg({'User_ID':'nunique'})


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[3]:


def q1():
    # Retorne aqui o resultado da questão 1.
    '''Retorna # de observações e variáveis no dataset
     input [pd.DataFrame]
     output [int] 
    '''
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[4]:


def q2():
    # Retorne aqui o resultado da questão 2.
    ''' Calcula # de  mulheres com idade entre 26-35 anos no dataset
     input [pd.DataFrame]
     output [int] 
    '''
    #black_friday[(black_friday['Age']=='26-35') & (black_friday['Gender']=='F')]['User_ID'].nunique()
    return len(black_friday[(black_friday['Age']=='26-35') & (black_friday['Gender']=='F')])


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    # Retorne aqui o resultado da questão 3.
    ''' Calcula # de User_ID no dataset
     input [pd.DataFrame]
     output [int] 
    '''
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    # Retorne aqui o resultado da questão 4.
    ''' Calcula # de diferentes tipos de dados presentes no dataset
     input [pd.DataFrame]
     output [int] 
    '''
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    # Retorne aqui o resultado da questão 5.
    ''' Calcula % de registros com ao menos um NAs
     input [pd.DataFrame]
     output [float] 
    '''
    aux = black_friday.isna().sum(axis=1)
    aux1 = [1 if (x!=0) else 0 for x in aux]
    # sum(aux!=0)/len(black_friday) 
    return sum(aux1)/len(black_friday)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    # Retorne aqui o resultado da questão 6.
    ''' Número de NAs na coluna com maior Missing Values
     input [pd.DataFrame]
     output [int] 
    '''    
    na_max = black_friday.isna().sum().max()
    return int(na_max)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[11]:


def q7():
    # Retorne aqui o resultado da questão 7.
    '''Moda em Product_Category_3 desconsiderando NA 
     input [pd.DataFrame]
     output [float] mode
    '''

    #black_friday['Product_Category_3'].value_counts().index[0]
    return black_friday['Product_Category_3'].mode()[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# ![image.png](attachment:image.png)

# In[12]:


def q8():
    # Retorne aqui o resultado da questão 8.
    '''Ao Normalizar os tados obtemos uma variável entre 0 e 1
     Lembre-se: A escolha entre normalizar e padronizar sempre 
     deve levar me conta o tipo de dados que se dispõe
     
     input [pd.DataFrame]
     output [float] média da variável PURCHASE normalizada
     
     '''
    
    p_min = black_friday['Purchase'].min()
    p_max = black_friday['Purchase'].max()
    p_norm = (black_friday['Purchase']-p_min)/(p_max-p_min)
    return float(p_norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# ![image.png](attachment:image.png)

# In[13]:


def q9():
    # Retorne aqui o resultado da questão 9.
    ''' Ao Padronizar os dados obtemos uma variável com 
    média igual a 0 e um desvio padrão igual a 1.
    Lembre-se: A escolha entre normalizar e padronizar 
    sempre deve levar me conta o tipo de dados que se dispõe
     
     input [pd.DataFrame]
     output [int] ocorrências entre -1 e 1 da  variável PURCHASE padronizada
     
     '''
    
    p_mean = black_friday['Purchase'].mean()
    p_std = black_friday['Purchase'].std()
    z =( black_friday['Purchase']-p_mean)/p_std
    z_interval = z.between(-1,1, inclusive = True).sum()
    return int(z_interval)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[14]:


def q10():
    # Retorne aqui o resultado da questão 10.
    ''' Analisa se a variável 'Product_Category_3' 
    também é nula toda às vezes que 'Product_Category_2' é nula.
     
    input [pd.DataFrame]
    output [bool] 
     
    '''
    
    aux1 = black_friday[black_friday['Product_Category_2'].isna()]['Product_Category_2']
    aux2 = black_friday[black_friday['Product_Category_2'].isna()]['Product_Category_3']
    return aux1.equals(aux2)

