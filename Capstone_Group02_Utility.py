#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE


# In[35]:


def Logisticsmodel(x,y,x1,y1):
    mdl=LogisticRegression()
    mdl.fit(x,y)
    mdl.predict(x1)
    score=mdl.score(x1,y1)
    return score
    


# In[42]:


def DTClassifier(x,y,x1,y1,a,b):
    DT=DecisionTreeClassifier(max_depth= a, criterion=b)
    DT.fit(x,y)
    DT.predict(x1)
    score=DT.score(x1,y1)
    return score


# In[51]:


def RFClassifier(x,y,x1,y1):
    RF=RandomForestClassifier()
    RF.fit(x,y)
    RF.predict(x1)
    score=RF.score(x1,y1)
    features=RF.feature_importances_
    return score, features


# In[66]:


def GBClassifier(x,y,x1,y1):
    XGB=GradientBoostingClassifier(learning_rate=1, n_estimators=1000, subsample=1.0,max_depth=3)
    XGB.fit(x,y)
    XGB.predict(x1)
    score=XGB.score(x1,y1)
    return score


# In[88]:


from sklearn.neural_network import MLPClassifier


# In[89]:


def NNClassifier(x,y,x1,y1):
    NN=MLPClassifier(hidden_layer_sizes=(100,150,100),solver='sgd',activation='relu',learning_rate_init=0.01, max_iter=5000)
    NN.fit(x,y)
    NN.predict(x1)
    score=NN.score(x1,y1)
    return score

def KNNClassifier(x,y,x1,y1,n,m,w):
    KNN=KNeighborsClassifier(n_neighbors=n, metric=m, weights=w)
    KNN.fit(x,y)
    KNN.predict(x1)
    score=KNN.score(x1,y1)
    return score

import plotly.graph_objs as go
def model_evaluation(a,b,c,d,e,f):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Gradient Boost','Random Forest','Neural Network','KNN','Decision Tree','Logistic Regression'],y=[a,b,c,d,e,f]))
    fig.update_layout(title='Performance evaluation of models',xaxis_title='Model',yaxis_title='Accuracy Score')
    modelplot=fig.show()
    return modelplot