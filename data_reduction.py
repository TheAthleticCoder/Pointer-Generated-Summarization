
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv("train.csv")
df.head()


# In[3]:


print(len(df))


# In[4]:


df = df[:50000]
df.to_csv('train_50k.csv', index=False)


# In[5]:


df_1 = pd.read_csv("train_50k.csv")
df_1.head()


# In[6]:


print(len(df_1))

