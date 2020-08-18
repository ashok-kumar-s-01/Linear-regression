#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv('.../House_Data.csv')


# In[3]:


dataset


# In[4]:


X=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]


# In[5]:


X


# In[6]:


y


# In[7]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# In[8]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[9]:


y_pred=regressor.predict(X_test)


# In[10]:


y_pred


# In[11]:


plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')


# In[12]:


plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("SquareFeet vs Price($)")
plt.xlabel("SquareFeet")
plt.ylabel('Price')
plt.show()


# In[ ]:




