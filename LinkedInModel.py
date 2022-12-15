#!/usr/bin/env python
# coding: utf-8

#import packages
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st


# add header
st.markdown("LinkedIn User?")



#import data
import pandas as pd
s = pd.read_csv("social_media_usage.csv")




# In[27]:


#create function 1 = LinkedIn user, 0 = is not
import numpy as np
def clean_sm(x):
    return np.where(x==1,1,0)
   



# In[30]:


# Clean data
ss = pd.DataFrame({
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"]>98, np.nan, s["age"]),
    "sm_li":clean_sm(s["web1h"])})

# In[58]:
#drop NAs
ss = ss.dropna()


# In[59]:
# check if NA is eliminated
ss.isnull().sum()

income = st.slider("Income:", 1, 9)
education = st.slider("Education:", 1, 8)
parent = st.slider("Parent:", 0,1)
married = st.slider("Married:", 0,1)
female = st.slider("Female:", 0,1)
age = st.slider("Age:", 1, 98)



# In[66]:


# create target column sm_li
y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]


# In[67]:


# split data into training and test
x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify=y,
                                                   test_size = 0.2,
                                                   random_state = 987)
# x_train carries 80% of feature data used to predict sm_li. y_train carries 80% of target (sm_li) data that
# corresponds with the 80% of feature. This is to train the model. 
# x_test carries 20% of feature data used to test the model on new data. y_test carries 20% of data and 
# contains the target we will predict when testing the model on new data to evaluate performance.
                                        




# oversample
from imblearn.over_sampling import RandomOverSampler


# In[69]:


#Instantiate
over = RandomOverSampler(random_state=987)


# In[70]:


# Fit features and target
x_over, y_over = over.fit_resample(x, y)


# In[71]:


# Check target classes
print(f"Original:\n{y.value_counts()}\n")

print(f"Oversampled:\n{y_over.value_counts()}\mZ")# Check target classes
print(f"Original:\n{y.value_counts()}\n")

print(f"Oversampled:\n{y_over.value_counts()}\mZ")


# In[72]:


#create algorithm
lr = LogisticRegression()


# In[73]:


#fit algorithm into training dta
lr.fit(x_train, y_train)


# In[74]:


#make prediction with test data
y_pred = lr.predict(x_test)


# In[86]:


# compare predictions to test data
confusion_matrix(y_test, y_pred)
#True Negative (as predected) is 145. False Negative (predicted as positive) is 50. True Positive (as predicted) is 34. 
#False Positive (predicted as negative) is 23.


# In[80]:


#create new dataframe
pd.DataFrame(confusion_matrix(y_test, y_pred),
           columns = ["Predicted Negative", "Predicted Positive"],
           index = ["Actual Negative", "Actual Positive"]).style.background_gradient(cmap="PiYG")



# In[83]:


# statistical metrics
print(classification_report(y_test, y_pred))
#accuracy of 71%


# In[98]:








# In[110]:


#make prediction with new data
person = [income, education, parent, married, female, age]
predicted_class = lr.predict([person])
probs = lr.predict_proba([person])

if st.button("Predict:"):
    person = predicted_class[0]
    st.success(f"Probability this person is a LinkedIn User: {probs[0][1]}")



# In[112]:


#print prediction outputs
print(f"Predicted class: {predicted_class[0]}")
print(f"Probability that this person is a LinkedIn user: {probs[0][1]}")

