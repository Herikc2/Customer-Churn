#!/usr/bin/env python
# coding: utf-8

# # Definição do Problema de Negócio

# Será criado um modelo preditivo, para a probabilidade de um cliente cancelar o seu plano. Será utilizado dados históricos fornecidos pelo Operadora de Telecom.
# 
# O dataset possui dados anonimos de mais de 3000 mil clientes, separado pela operadora em dois datasets, o primeiro para treino e o segundo para testes.
# 
# A coluna "churn" é a variável a ser prevista. Sendo possivel dois valores "no" e "yes".
# 
# A tarefa é prever a probabilidade de cada cliente cancelar o seu plano.

# # Import

# In[22]:


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer
import pickle
import warnings
warnings.filterwarnings("ignore")


# # Extraindo e Carregando os Dados

# Os arquivos foram carregados no formato CSV, a Operadora forneceu um arquivo para treino e outro para testes.

# In[2]:


# Carregando o dataset de treino e teste em formato CSV


# In[3]:


arquivo_treino = 'data/projeto4_telecom_treino.csv'
arquivo_teste = 'data/projeto4_telecom_teste.csv'
dados_treino = pd.read_csv(arquivo_treino)
dados_teste = pd.read_csv(arquivo_teste)
print(dados_treino.shape)
print(dados_teste.shape)


# # Análise Explorátoria de Dados

# Em uma primeira analise já foi identificado que a primeira coluna não é necessário para trabalhar com esse conjunto de dados, visto que é somente um ID. Entrou em questionamento também a necessidade da colna "area_code", visto que a coluna "state" já trás o senso de localização e "area_code" só iria aumentar a restrição, poderia ser muito util se a gama de dados fosse grande, porém para esse conjunto aparenta ser desnecessário.

# In[4]:


# Visualizando as 20 primeiras linhas
dados_treino.head(20)


# In[5]:


# Tipo de dados de cada atributo
dados_treino.dtypes


# In[6]:


# Sumário Estatísticoo
dados_treino.describe()


# In[7]:


# Disribuição das classes
dados_treino.groupby("churn").size()


# # PROCESSAMENTO DE DADOS

# Como visualizado acima possui um grande numero de clientes na classe "no", ou seja clientes que não pretendem desistir de seu plano de operadora. Para melhores resultados irá ser necessário um balanceamento do dataframe. Também é visualizado que seria ideal aplicar binarização nas colunas 'international_plan', 'voice_mail_plan' e 'churn' visto que possuem valores binário "no" ou "yes".

# In[8]:


def tratamento_dados(df):
    df = df.drop(columns=['Unnamed: 0'])
    df['international_plan'] = df['international_plan'].map(lambda x: '0' if x == 'no' else ('1' if x == 'yes' else 'NA') )
    df['voice_mail_plan'] = df['voice_mail_plan'].map(lambda x: '0' if x == 'no' else ('1' if x == 'yes' else 'NA') )
    df['churn'] = df['churn'].map(lambda x: '0' if x == 'no' else ('1' if x == 'yes' else 'NA') )
    
    df = df.astype({"international_plan": int, "voice_mail_plan": int, "churn": int})
    
    return df


# In[9]:


dados_treino = tratamento_dados(dados_treino)
dados_teste = tratamento_dados(dados_teste)


# In[10]:


dados_treino.dtypes


# In[11]:


dados_treino.head(20)


# In[12]:


# Dividindo por classe
dados_classe_0 = dados_treino[dados_treino['churn'] == 0]
dados_classe_1 = dados_treino[dados_treino['churn'] == 1]

contador_classe_0 = dados_classe_0.shape[0]
contador_classe_1 = dados_classe_1.shape[0]

dados_classe_0_sample = dados_classe_0.sample(contador_classe_1)
dados_treino = pd.concat([dados_classe_0_sample, dados_classe_1], axis = 0)


# In[13]:


# Disribuição das classes
dados_treino.groupby("churn").size()


# In[14]:


# Correlação de Pearson
dados_treino.corr(method = 'pearson')


# In[15]:


# Encontrando a correlação entre a variável target com as variáveis preditoras
corr = dados_treino[dados_treino.columns[1:]].corr()['churn'][:].abs()


# In[16]:


minima_correlacao = 0.05
corr2 = corr[corr > minima_correlacao]
corr2.shape
corr2


# In[17]:


corr_keys = corr2.index.tolist()
dados_filter = dados_treino[corr_keys]
dados_filter.head(20)
dados_filter.dtypes


# In[18]:

# Filtrando colunas que possuem correlação acima do estipulado
array_treino = dados_treino[corr_keys].values
# Separandooo o array em X (input) e y (output) para dados de treino
X_treino = array_treino[:, 0:array_treino.shape[1] - 1]
y_treino = array_treino[:, array_treino.shape[1] - 1]

array_teste = dados_teste[corr_keys].values
X_teste = array_teste[:, 0:array_teste.shape[1] - 1]
y_teste = array_teste[:, array_teste.shape[1] - 1]

# Gerando os dados normalizados
scaler = Normalizer().fit(X_treino)
normalizedX_treino = scaler.transform(X_treino)

scaler2 = Normalizer().fit(X_teste)
normalizedX_teste = scaler2.transform(X_teste)

y_treino = y_treino.astype('int')
y_teste = y_teste.astype('int')


# In[19]:

# Treinamento

# Preparando a lista de modelos
modelos = []
modelos.append(('LR', LinearRegression()))
modelos.append(('Ridge', Ridge()))
modelos.append(('Lasso', Lasso()))
modelos.append(('ElasticNet', ElasticNet()))
modelos.append(('KNN', KNeighborsRegressor()))
modelos.append(('CART', DecisionTreeRegressor()))
modelos.append(('SVR', SVR()))

for nome, modelo in modelos:
    modelo_treinado = modelo
    modelo_treinado.fit(X_treino, y_treino)
    Y_pred = modelo_treinado.predict(X_teste)
    
    mse = mean_squared_error(y_teste, Y_pred)
    mae = mean_absolute_error(y_teste, Y_pred)
    r2 = r2_score(y_teste, Y_pred)
    print("O MSE do modelo", nome, "é:", mse)
    print("O MAE do modelo", nome, "é:", mae)
    print("O R2 do modelo", nome, "é:", r2)
    print("\n")
    
''' 
    Após uma analise concluiu-se que os melhores modelos sem uma configuração, seriam o SVM e  CART. Os outros modelos,
    apresentaram incosistências altas entre as métricas e/ou resultados piores.
'''

# In[ ]:
    
modelo = SVR()
modelo.fit(X_treino, y_treino)
Y_pred = modelo.predict(X_teste)    

mse = mean_squared_error(y_teste, Y_pred)
mae = mean_absolute_error(y_teste, Y_pred)
r2 = r2_score(y_teste, Y_pred)
print("O MSE do modelo SVR é:", mse)
print("O MAE do modelo SVR é:", mae)
print("O R2 do modelo SVR é:", r2)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




