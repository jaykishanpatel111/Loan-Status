#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


cd F:\Dataset\Done projects\Loan Status


# # Data Exploration and Pre-processing

# In[ ]:


# load the given dataset


# In[3]:


df = pd.read_csv('Python_Project_7_Nai.csv')
df.head()


# In[ ]:


#  check the null values


# In[4]:


df.isnull().sum()


# In[5]:


# print the column names


# In[6]:


df.columns


# In[ ]:


# create list for all the columns which have null values columns


# In[7]:


lis = ['ID', 'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
       'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
       'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'Default Status']


# In[ ]:


# fill all the null values with mean using for loops


# In[8]:


for i in lis:
    df[i]=df[i].fillna(df[i].mean())


# In[9]:


df.isnull().sum()


# In[ ]:


# get data information


# In[10]:


df.info()


# In[11]:


# describe dataset


# In[12]:


df.describe()


# In[13]:


# display box plot for LIMIT_BAL


# In[14]:


df['LIMIT_BAL'].plot(kind='box')


# In[15]:


# display box plot for age


# In[16]:


df['AGE'].plot(kind='box')


# In[ ]:


# drop all the null values


# In[17]:


df = df.dropna()


# # perform encoding on default status

# In[19]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['Default Status'] = label.fit_transform(df['Default Status'])


# In[20]:


df.head()


# # Correlation between each features

# In[44]:


corr_data = df.corr()


# In[45]:


corr_data


# # Ploting heat map of the correlated data

# In[46]:


plt.figure(figsize = (10,5))
sns.heatmap(corr_data, annot = True, cmap = 'RdYlGn')


# In[ ]:





# # Working with Models

# In[ ]:


# Create a features and target dataset


# In[21]:


X = df.drop('Default Status',axis=1)
Y = df['Default Status']


# In[22]:


# Split data into training and testing


# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 10)


# In[24]:


# Fit the Gaussian naive bayes classifier


# In[25]:


from sklearn.naive_bayes import GaussianNB


# In[26]:


model = GaussianNB()
model.fit(X_train,Y_train)


# In[27]:


Y_train


# # Print the testing score

# In[43]:


model.score(X_test,Y_test)


# In[ ]:





# # Checking accuracy score of our model

# In[38]:


def run_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train,Y_train.ravel())
    accuracy = accuracy_score(Y_test, Y_pred)
    print("pricison_score: ",precision_score(Y_test, Y_pred))
    print("recall_score: ",recall_score(Y_test, Y_pred))
    print("Accuracy = {}".format(accuracy))
    print(classification_report(Y_test,Y_pred,digits=5))
    print(confusion_matrix(Y_test,Y_pred))


# In[39]:


run_model(model, X_train, Y_train, X_test, Y_test)


# In[40]:


cm = confusion_matrix(Y_test, Y_pred)
cm


# In[41]:


# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)


# # Classification Report

# In[42]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

