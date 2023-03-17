#!/usr/bin/env python
# coding: utf-8

# In[1]:


Kealean Biggs
Capstone Project One


# In[ ]:


link: https://www.kaggle.com/datasets/osuolaleemmanuel/ad-ab-testing

Columns Description
auction_id: the unique id of the online user who has been presented the BIO. In standard terminologies this is called an impression id. The user may see the BIO questionnaire but choose not to respond. In that case both the yes and no columns are zero.

experiment: which group the user belongs to - control or exposed.

control: users who have been shown a dummy ad
exposed: users who have been shown a creative, an online interactive ad, with the SmartAd brand.
date: the date in YYYY-MM-DD format

hour: the hour of the day in HH format.

device_make: the name of the type of device the user has e.g. Samsung

platform_os: the id of the OS the user has.

browser: the name of the browser the user uses to see the BIO questionnaire.

yes: 1 if the user chooses the “Yes” radio button for the BIO questionnaire.

no: 1 if the user chooses the “No” radio button for the BIO questionnaire.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('AdSmartABdata - AdSmartABdata.csv')
print(df)


# In[13]:


print(df.head())


# In[24]:


print(df.columns)


# In[15]:


print(df.info())


# In[25]:


print(df.describe())


# In[26]:


print(df.describe(include = "O"))


# In[28]:


answer_no = df.loc[(df.yes == 0) & (df.no == 0)].index


# In[29]:


df.drop(answer_no, inplace = True)


# In[30]:


df["choose_yes"] = df.yes.apply(lambda x: True if x == 1 else False)


# In[31]:


print(df)


# In[36]:


print(df.corr())


# In[39]:


cross_tab = pd.crosstab(index = df['experiment'],
                             columns = df['choose_yes'],
                             normalize = "index")
cross_tab.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))

plt.legend(loc="upper left", ncol=2)
plt.ylabel("Proportion")
plt.title("Number of Users pressing Yes button in Control Vs. Exposed Group")
plt.show()


# In[40]:


df.groupby(["experiment","choose_yes"]).size().to_frame()


# In[ ]:


A/B Testing
KPI: User pressing the "Yes" radio button in the BIO questionnaire

Prior Assumption: 50 % chance, since there are two buttons: yes or no. Relatively high uncertainty due to not accounting for drop off, i.e. not pressing any button.

Prior Distribution Parameters: alpha = 10, beta = 10

1. Goal
See whether or not being exposed to the creative, online interactive ad, with the SmartAd brand, increases the probability of pressing the "Yes" button in the questionnaire


# In[42]:


#2. Setting Prior
prior_alpha = 30
prior_beta = 30
prior = np.random.beta(prior_alpha, prior_beta, 10000)


# In[43]:


plt.figure(figsize = (12,6))
sns.kdeplot(prior, fill = True, alpha = 0.2)
plt.title("Prior Distribution")
plt.xlim(0,1)
plt.axvline(prior.mean(), linestyle='--', alpha = 0.6)
plt.show()


# In[46]:


#Data Collection
df.groupby(["experiment","choose_yes"]).size()


# In[47]:


control_true = df.loc[df.experiment == "control"].choose_yes.sum()
print(control_true)


# In[48]:


control_false = len(df.loc[df.experiment == "control"].choose_yes) - control_true
print(control_false)


# In[49]:


exposed_true = df.loc[df.experiment == "exposed"].choose_yes.sum()
print(exposed_true)


# In[50]:


exposed_false = len(df.loc[df.experiment == "exposed"].choose_yes) - exposed_true
print(exposed_false)


# In[ ]:


Simulate Distribution


# In[51]:


control_posterior = np.random.beta(prior_alpha + control_true, prior_beta + control_false, 10000)
exposed_posterior = np.random.beta(prior_alpha + exposed_true, prior_beta + exposed_false, 10000)


# In[61]:


plt.figure(figsize = (12,6))

sns.kdeplot(control_posterior, label = "control", fill = True, alpha = 0.2, color = "red")
plt.axvline(control_posterior.mean() - control_posterior.std() * 1.96, color = "red")
plt.axvline(control_posterior.mean() + control_posterior.std() * 1.96, color = "red")

sns.kdeplot(exposed_posterior, label = "exposed", fill = True, alpha = 0.2, color = "blue")
plt.axvline(exposed_posterior.mean() - exposed_posterior.std() * 1.96, color = "blue")
plt.axvline(exposed_posterior.mean() + exposed_posterior.std() * 1.96, color = "blue")

plt.legend()
plt.title("Posterior Distribution of Control and Exposed Group")
plt.show()


# In[62]:


diff = exposed_posterior - control_posterior
prob = (diff > 0).sum() / len(diff)


# In[63]:


lift = (diff / control_posterior).mean()


# In[64]:


print(f"Probability of Exposed Group 'success' versus Control Group of pressing Yes on Questinnaire is {round(prob*100, 2)}%")


# In[65]:


print(f"Exposed Group is {round(lift*100,2)}% more likely to press Yes on Questionnaire")


# In[ ]:


Decision:
    We cannot statistically determine whether the new version increases the probability of visitors on the site actually pressing the 'yes' button in the questionnaire.
    Therefore, we should reccomend that the client not make the changes to the site.

