#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[71]:


df = pd.read_csv('Train.csv')
df.head()


# In[72]:


df.describe()


# In[73]:


df.isnull().sum()


# In[74]:


mean_weight = df['Item_Weight'].mean()
median_weight = df['Item_Weight'].median()


# In[75]:


print(mean_weight,median_weight)


# In[76]:


df['Item_Weight_mean']=df['Item_Weight'].fillna(mean_weight)
df['Item_Weight_median']=df['Item_Weight'].fillna(median_weight)


# In[77]:


df.head(1)


# In[78]:


print("Original Weight variable variance",df['Item_Weight'].var())
print("Item Weight variance after mean imputation",df['Item_Weight_mean'].var())
print("Item Weight variance after median imputation",df['Item_Weight_median'].var())


# In[79]:


df['Item_Weight'].plot(kind = "kde",label="Original")
df['Item_Weight_mean'].plot(kind = "kde",label = "Mean")
df['Item_Weight_median'].plot(kind = "kde",label = "Median")
plt.legend()
plt.show()


# In[80]:


df[['Item_Weight','Item_Weight_mean','Item_Weight_median']].boxplot()


# In[81]:


df['Item_Weight_interpolate']=df['Item_Weight'].interpolate(method="linear")


# In[82]:


df['Item_Weight'].plot(kind = "kde",label="Original")
df['Item_Weight_interpolate'].plot(kind = "kde",label = "interpolate")
plt.legend()
plt.show()


# In[83]:


from sklearn.impute import KNNImputer


# In[84]:


knn = KNNImputer(n_neighbors=10,weights="distance")


# In[85]:


df['knn_imputer']= knn.fit_transform(df[['Item_Weight']]).ravel()


# In[86]:


df['Item_Weight'].plot(kind = "kde",label="Original")
df['knn_imputer'].plot(kind = "kde",label = "KNN imputer")
plt.legend()
plt.show()


# In[87]:


df=df.drop(['Item_Weight','Item_Weight_mean','Item_Weight_median','knn_imputer'],axis=1)


# In[88]:


df.head(1)


# In[89]:


df.isnull().sum()


# In[90]:


df['Outlet_Size'].value_counts()


# In[91]:


df['Outlet_Type'].value_counts()


# In[92]:


mode_outlet=df.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode()[0]))


# In[93]:


mode_outlet


# In[94]:


missing_values=df['Outlet_Size'].isnull()


# In[95]:


missing_values


# In[96]:


df.loc[missing_values,'Outlet_Size']=df.loc[missing_values,'Outlet_Type'].apply(lambda x:mode_outlet[x])


# In[97]:


df.isnull().sum()


# In[98]:


df.columns


# In[99]:


df['Item_Fat_Content'].value_counts()


# In[100]:


df.replace({'Item_Fat_Content':{'Low Fat':'LF','low fat':'LF','reg':'Regular'}},inplace=True)


# In[101]:


df['Item_Fat_Content'].value_counts()


# In[102]:


df.columns


# In[103]:


df['Item_Visibility'].value_counts()


# In[104]:


df['Item_Visibility_interpolate']=df['Item_Visibility'].replace(0,np.nan).interpolate(method='linear')


# In[105]:


df.head(1)


# In[106]:


df['Item_Visibility_interpolate'].value_counts()


# In[107]:


df['Item_Visibility'].plot(kind="kde",label="Original")
df["Item_Visibility_interpolate"].plot(kind="kde",color="red",label="Interpolate")
plt.legend()
plt.show()


# In[108]:


df=df.drop('Item_Visibility',axis=1)
df.head(1)


# In[109]:


df.columns


# In[110]:


df['Item_Type'].value_counts()


# In[111]:


df.columns


# In[112]:


df['Item_Identifier'].value_counts().sample(5)


# In[113]:


df['Item_Identifier'] =df['Item_Identifier'].apply(lambda x:x[:2])


# In[114]:


df['Item_Identifier'].value_counts()


# In[115]:


df['Outlet_Establishment_Year']


# In[116]:


import datetime as dt


# In[117]:


current_year=dt.datetime.today().year


# In[118]:


current_year


# In[119]:


df['Outlet_age']=current_year-df['Outlet_Establishment_Year']


# In[120]:


df.head(1)


# In[121]:


df = df.drop('Outlet_Establishment_Year',axis=1)


# In[122]:


df.head()


# In[124]:


df_encoded.head(3)


# In[125]:


from sklearn.preprocessing import OrdinalEncoder

df_encoded = df.copy()

cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    oe = OrdinalEncoder()
    df_encoded[col]=oe.fit_transform(df_encoded[[col]])
    print(oe.categories_)


# In[126]:


df_encoded.head(3)


# In[127]:


X = df_encoded.drop('Item_Outlet_Sales',axis=1)
y = df_encoded['Item_Outlet_Sales']


# In[128]:


y


# In[ ]:




