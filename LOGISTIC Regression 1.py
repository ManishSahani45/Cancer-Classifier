#!/usr/bin/env python
# coding: utf-8

# In[213]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[222]:


d=pd.read_csv("framingham cancer prediction LR dataset.csv")

d.head()
d.fillna(0)


# In[223]:


X=d [["male","age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol",
    "sysBP","diaBP","BMI","heartRate","glucose"]]


# In[224]:


X.head()


# In[225]:


y=d[["TenYearCHD"]]


# In[226]:


""""scalar= StandardScaler()
scalar.fit(X=d [["age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol",
    "sysBP","diaBP","BMI","heartRate","glucose"]])
X=d [["age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol",
    "sysBP","diaBP","BMI","heartRate","glucose"]]=scalar.transform(X=d [["age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol",
    "sysBP","diaBP","BMI","heartRate","glucose"]])
X"""


# In[227]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,shuffle= True )


# In[228]:


Scaler=StandardScaler()
#Scaler.fit(X_train)


# In[229]:


X_test


# In[230]:


X_train=Scaler.fit_transform(X_train)


# In[231]:


X_test=Scaler.fit_transform(X_test)


# In[237]:



X_train = np.nan_to_num(X_train)
X_train


# In[238]:


X_test = np.nan_to_num(X_test)


# In[239]:


X_test


# In[240]:


#training model 
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state= 0)


# In[ ]:





# In[241]:


#fittting the model withh training data
classifier.fit(X_train,y_train)


# In[242]:


y_pred=classifier.predict(X_test)


# In[243]:


y_pred


# In[246]:


#performance calculation by Evaluation metrics
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
cm
#here we can find the accuracy score by true result by total result i.e 713(true positive)+ 6(false positive)/total


# In[249]:


# performance calculation by acccuracy score
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac


# In[ ]:




