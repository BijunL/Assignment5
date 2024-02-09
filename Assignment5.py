#!/usr/bin/env python
# coding: utf-8

# # 0.) Import the Credit Card Fraud Data From CCLE

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df = pd.read_csv("fraudTest.csv")


# In[3]:


df.head()


# In[6]:


df_select = df[["trans_date_trans_time", "category", "amt", "city_pop", "is_fraud"]].copy()

df_select["trans_date_trans_time"] = pd.to_datetime(df_select["trans_date_trans_time"])


df_select["time_var"] = df_select["trans_date_trans_time"].dt.second


X = pd.get_dummies(df_select, columns=["category"]).drop(["trans_date_trans_time", "is_fraud"], axis=1)
y = df_select["is_fraud"]


# # 1.) Use scikit learn preprocessing to split the data into 70/30 in out of sample

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)


# In[9]:


X_test, X_holdout, y_test, y_holdout = train_test_split(X_test, y_test, test_size = .5)


# In[10]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# In[11]:


X_test = scaler.transform(X_test)
X_holdout = scaler.transform(X_holdout)


# # 2.) Make three sets of training data (Oversample, Undersample and SMOTE)¶

# In[12]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


# In[13]:


ros = RandomOverSampler()
over_X, over_y = ros.fit_resample(X_train, y_train)

rus = RandomUnderSampler()
under_X, under_y = rus.fit_resample(X_train, y_train)

smote = SMOTE()
smote_X, smote_y = smote.fit_resample(X_train, y_train)


# # 3.) Train three logistic regression models

# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


over_log = LogisticRegression().fit(over_X, over_y)

under_log = LogisticRegression().fit(under_X, under_y)

smote_log = LogisticRegression().fit(smote_X, smote_y)


# # 4.) Test the three models

# In[16]:


over_log.score(X_test, y_test)


# In[17]:


under_log.score(X_test, y_test)


# In[18]:


smote_log.score(X_test, y_test)


# We see SMOTE performing with higher accuracy but is ACCURACY really the best measure?

# # 5.) Which performed best in Out of Sample metrics?

# In[19]:


from sklearn.metrics import confusion_matrix


# In[20]:


y_true = y_test


# In[21]:


y_pred = over_log.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm


# In[22]:


print("Over Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))


# In[23]:


y_pred = under_log.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm


# In[24]:


print("Under Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))


# In[25]:


y_pred = smote_log.predict(X_test)
cm = confusion_matrix(y_true, y_pred)
cm


# In[26]:


print("SMOTE Sample Sensitivity : ", cm[1,1] /( cm[1,0] + cm[1,1]))


# # 7.) We want to compare oversampling, Undersampling and SMOTE across our 3 models (Logistic Regression, Logistic Regression Lasso and Decision Trees).
# 
# # Make a dataframe that has a dual index and 9 Rows.
# # Calculate: Sensitivity, Specificity, Precision, Recall and F1 score. for out of sample data.
# # Notice any patterns across perfomance for this model. Does one totally out perform the others IE. over/under/smote or does a model perform better DT, Lasso, LR?
# # Choose what you think is the best model and why. test on Holdout

# In[29]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


# In[30]:


resampling_methods = {
    "over": RandomOverSampler(),
    "under": RandomUnderSampler(),
    "smote": SMOTE()
}

model_configs = {
    "LOG":LogisticRegression(),
    "LASSO": LogisticRegression(penalty = "l1", C = 2., solver = "liblinear"),
    "DTREE": DecisionTreeClassifier()
}


# In[47]:


def calc_perf_metric(y_true, y_pred):
    tn,fp,fn,tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_pred),
    recall = recall_score(y_true, y_pred),
    f1 = f1_score(y_true, y_pred)
    
    
    return(sensitivity, specificity, precision, recall, f1)


# In[32]:


trained_models = {}
results = []


# In[48]:


for resample_key, resampler in resampling_methods.items():
    resample_X, resample_y = resampler.fit_resample(X_train, y_train)
    
    for model_key, model in model_configs.items():
        combined_key = f"{resample_key}_{model_key}"
        
        m = model.fit(resample_X, resample_y)
        
        trained_models[combined_key] = m
        
        y_pred = m.predict(X_test)
        
        sensitivity, specificity, precision, recall, f1 = calc_perf_metric(y_test,y_pred)
        
        results.append({"Model": combined_key,
                       "Sensitivity" : sensitivity,
                       "Specificity": specificity,
                       "Precision": precision,
                       "Recall": recall,
                       "F1" : f1})


# In[49]:


result_df = pd.DataFrame(results)


# In[50]:


result_df

