#!/usr/bin/env python
# coding: utf-8

# ## About the Project
# 
# ####  This project makes use of ANN( Artificial Neural Network) of predict the memory test when under certian medication. The data is based on the experiment of anti-anxiety medicine on memory recall when being primed with happy and sad memories. The drugs that are used are ( Alprazolam (label =A)), (Triazolam)(label= T), (Sugar Tablet)(label = S). The label for happy and sad memores are ( S for sadness) and (H for happyness). The participants were from various age and were given various doses of drug. 
# #### The project has a correlation chart that gives good idea about the relationship between age and testscore, effect of medication on test socre and its effectiveness
# 
# #### Source of data: http://www.jstor.org/stable/43854146, https://www.sciencedirect.com/science/article/pii/S0896627314008484,
# #### http://www.jstor.org/stable/40064315

# # 1 Imports

# In[214]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
# neural network imports
import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import confusion_matrix

from keras.callbacks import TensorBoard
from time import strftime


from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix


# In[215]:


# Creating a folder of tensorboard performace logs

LOG_DIR='tensorboard_logs/'


# # 2 Cleaning our data

# In[216]:


org_data=pd.read_csv("Islander_data.csv")


# In[217]:


# Viewing the first 5 rows of our original data
org_data[:5]


# ## 2.1 Converting expressions and drugs into a dummy variable

# In[218]:


# converting happy_sad_gropy and Drug to a dummy variable
# This is really important step as we don't want the srting or variable 1, 2 or 3 to affect our prediction
dummy_Hap_Sad=pd.get_dummies(org_data['Happy_Sad_group'])
dummy_Drug=pd.get_dummies(org_data['Drug'])

# concating the dummy variable to our original data
new_org_data=pd.concat([dummy_Hap_Sad,dummy_Drug,org_data], axis=1)
new_org_data.head()


# In[219]:


# Removing first and last name as it doesn't play any role in data prediction
# Also removing Drug and happy_sad gorup as they have been converted to a dummy variable
# we are also removing the drug and happy sad grop 
features=new_org_data.drop(['first_name','last_name','Mem_Score_After','Diff','Drug','Happy_Sad_group'], axis=1)
target=new_org_data['Mem_Score_After']


# In[220]:


features.head()


# In[221]:


# This is the coulumn we are going to predict so its named target
target.head()


# In[222]:


# the description of our feature data
features.describe()


# # 3 Visualizing data

# ## 3.1 Visualizing data using Correlation

# In[223]:


# finding the correlation between all our features 
# we concat both features and mean score after to find correlatin between all our data

filtered_all_data=pd.concat([features,target], axis=1)
filtered_all_data.head()


# In[224]:


# now we find correlation using heat map
# we know that the uppertrange of the correlation table is the same as lower traiange so we get rid of them using mask

mask=np.zeros_like(filtered_all_data.corr())
traingle_indices=np.triu_indices_from(mask)
mask[traingle_indices]= True

#one our mask is all set
#we find the correlation and plot it into a head map
plt.figure(figsize=[12,12])
sns.heatmap(filtered_all_data.corr(), mask=mask, annot=True, annot_kws= {"size":14})
sns.set_style('white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# ## Skew Distribution 

# In[225]:


# skew of mean score before distribution
# we got a positve skew and most of the score was distributed around 45-50
plt.figure(figsize=[12,8])
plt.title(f" Distribution of mean score before drug. skew= ({new_org_data['Mem_Score_Before'].skew()})", fontsize=16)
sns.distplot(new_org_data['Mem_Score_Before'], hist=False)

plt.show()


# In[226]:


# skew of age
plt.figure(figsize=[12,8])
plt.title(f" Distribution of mean score before drug. skew= ({new_org_data['age'].skew()})", fontsize=16)
sns.distplot(new_org_data['age'], hist=False)

plt.show()


# In[ ]:





# # 4 Dividing trainining and testing data

# In[227]:


# This is a important phase where our data is divided into a training and testing data
# It is divided in a ration of 8:2 where 8 is training data and 2 is testing data
# we are using the training data to pass in our neural network
# the test data will be used to comapre the prediction that we got form our neural network
x_train, x_test, y_train, y_test= train_test_split(features,target, test_size=0.2)


# In[228]:


# comparign our training dataset to a array as our neural network doesn't take dataframe
x_train=np.array(x_train)
y_train=np.array(y_train)


# In[229]:


# Feature scaling
# Feature scaling is really important as it helps to minimize the gap between data
# it also enhances our calculation 
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
x_train[:5]


# # 5 Using ANN 

# ## 5.1 Initializing ANN

# In[230]:


# classifiying our model
model=Sequential()


# In[231]:


# Adding different layers in our ANN
# init= where we want to start and it should be uniform. Usually start between 0-1
#our first layer
model.add(Dense(output_dim=128,activation='relu', input_dim=8))

#our second layer
model.add(Dense(output_dim=64, activation='relu'))

#our third layer
model.add(Dense(output_dim=32, activation='relu'))


#our output layer
model.add(Dense(output_dim=1, activation='linear'))


# ## 5.2 Compiling and Predicting the ANN

# In[232]:


model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])
model.summary()


# # 5.2 Using Tensor Board
# ### Helps us visualize the training process

# In[233]:


def get_tensorboard(model_name):
    folder_name=f'{model_name} at {strftime("%H %M")}'
    
    dir_path=os.path.join(LOG_DIR,folder_name)
    try:
        os.makedirs(dir_path)
    except osError as err:
        print(err.stererror)
    else:
        print(" Successfully created directiory")
    return TensorBoard(log_dir=dir_path)


# ## 5.3 Training the data

# In[280]:


# predicting our ANN
# xtrain and y train must be a list so convert it
#we have epoch of 1500 i.e it will pass batch size of 32 1500 times
# the data will stored in model 1 to see visulaize in tensor board
model.fit(x_train,y_train, batch_size=32, nb_epoch=3500, validation_split=0.2, callbacks=[get_tensorboard('Model 1')])

#saving our model
model.save("model_test_score.h5")


# # 5.4 predicting the test data using our model

# In[283]:


#loading our model
model= keras.models.load_model("model_test_score.h5")

#predicting the model using our testing data
#Predicing entire test data
prediction=model.predict(x_test)
prediction[:5]


# In[284]:


y_test[:5]


# # 5.5 Comaring predicted data with actual data

# In[285]:


# comparing our actual test score vs predicted score of x_test data
predict_vs_actual=pd.DataFrame(prediction, y_test, columns=["PREDICTED_SCORE"])
print(predict_vs_actual)


# In[286]:


#single test row
x_test[1:2]


# In[287]:


#passing in a single data
prediction=model.predict(x_test[1:2])
print(prediction)


# # Building a costom data function
# 

# In[288]:


our_stats=np.zeros((1, 8))

def score_predictor(status, Drug, age, dosage, score_before):
    if status=='H':
        our_stats[0][0]=1
    else:
        our_stats[0][1]=1
    if Drug=='A':
        our_stats[0][2]=1
    elif Drug=='S':
        our_stats[0][3]=1
    else:
        our_stats[0][4]=1
        
        #updating our last 3 elements
    our_stats[:,5:]=[age,dosage,score_before]

    return sc.transform(our_stats)


# In[291]:


#our manual data
new_stats=score_predictor('H','A',30,1,70.3)
#our model Before prediction
print(new_stats)


# In[292]:


new_prediction=model.predict(new_stats)
print(new_prediction)


# In[ ]:




