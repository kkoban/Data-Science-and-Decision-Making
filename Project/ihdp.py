#!/usr/bin/env python
# coding: utf-8

# ## IHDP

# In[1]:


from sklearn.model_selection import train_test_split, cross_val_score # to split the data into train and test datasets 
from sklearn.preprocessing import StandardScaler # use of StandardScaler to standarise the dataset
import numpy as np # library of mathematical operations
import pandas as pd  # for data anlysis and manipulation
import matplotlib.pyplot as plt # to display charts
import seaborn as sns # data visualisation library
from econml.metalearners import XLearner # advanced CATE estimator
from sklearn.model_selection import GridSearchCV # library function for cross-validation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier # meta estimator for fitting decision trees

from helper_functions import feat_imp, get_ps_weights, pehe # self-defined funtions 


# ### Loading the dataset

# In[2]:


# to load the dataset
ihdp = pd.read_csv("https://raw.githubusercontent.com/dmachlanski/CE888_2022/main/project/data/ihdp.csv", delimiter=",")
ihdp


# ### Exploring the dataset

# In[3]:


# to print information about the dataset
ihdp.info()


# In[4]:


nrow, ncol = ihdp.shape
print(f'There are {nrow} rows and {ncol} columns')


# In[13]:


ihdp['t'].value_counts(1)


# In[7]:


X = ihdp[["x1", 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
         'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 
         'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25' ]].copy()


# In[8]:


# reshaping the single dimension vectors into 2D vectors as some methods prefer this representaion of data
T = ihdp['t'].values.reshape(-1, 1)
Y = ihdp['yf'].values.reshape(-1, 1)
ite = ihdp['ite']


# In[9]:


# to plot a boxplot of each feature in the dataset and check if the features vary in scale
plt.figure(figsize=(30,15))
sns.boxplot(data = X)


# In[ ]:


plt.savefig("ihdp_boxplot.pdf")


# In[10]:


# to draw histogram and look at the distribution of values of each feature
ihdp.hist(bins=50, figsize=(20,20))


# In[11]:


plt.savefig("ihdp_histogram.pdf")


# In[15]:


# to calculate the correlations between each pair of variables
cor = X.corr()

# to plot a heatmap of the correlations between pairs of features
sns.set(rc = {'figure.figsize':(20,15)})
sns.heatmap(cor, annot = True)


# In[ ]:


plt.savefig("ihdp_heatmap.pdf")


# ### Data pre-processing

# In[16]:


# to split the data into train and test datasets
train_set, test_set = train_test_split(ihdp, test_size=0.2, random_state=50)  
print(len(train_set), len(test_set))


# In[17]:


x_train = train_set.drop(['t', 'yf', 'ycf', 'ite'], axis = 1)
x_test = test_set.drop(['t', 'yf', 'ycf', 'ite'], axis = 1)


# In[18]:


y_train = train_set['yf']
y_test = test_set['yf']


# In[19]:


t_train = train_set['t']
t_test = test_set['t']


# In[20]:


ite_train = train_set['ite']
ite_test = test_set['ite']


# In[21]:


# to standarise the dataset i.e. mean = 0 and s.d. = 1
scaler_x = StandardScaler() 
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)


# In[22]:


x_train = pd.DataFrame(x_train)
t_train = pd.DataFrame(t_train)


# In[23]:


x_test = pd.DataFrame(x_test)
t_test = pd.DataFrame(t_test)


# ### Modelling

# #### Simple learners

# In[24]:


# to fit a simple Random Forest regression model
reg = RandomForestRegressor() 


# In[25]:


scores = cross_val_score(reg, x_train, y_train, cv=10, scoring = "neg_mean_squared_error")
print("Negative mean squared error: %.2f +/- %.2f" % (scores.mean(), scores.std()))


# In[26]:


param_grid = { 
    'n_estimators': [25,50,100,150],
    'max_depth' : [3,4,5,6,7,8],
}


# In[27]:


# for hyperparameter tuning
CV_reg = GridSearchCV(estimator=reg, param_grid=param_grid, cv= 10) 
CV_reg.fit(x_train, y_train)


# In[28]:


# to find the best suited parameters for the regression model
CV_reg.best_params_ 


# In[29]:


simple_reg = RandomForestRegressor(n_estimators = 100, max_depth = 6)


# In[30]:


xt_train = np.concatenate([x_train, t_train], axis=1) # to merge x and t

# to build the model on the traing data
simple_reg.fit(xt_train, y_train) 


# In[31]:


scores = cross_val_score(simple_reg, xt_train, y_train, cv = 10, scoring = "neg_mean_squared_error")
print("Negative mean squared error: %.2f +/- %.2f" % (scores.mean(), scores.std()))


# In[32]:


# to merge x and t = 0
xt0_test = np.concatenate([x_test, np.zeros_like(t_test)], axis=1) 

# to predict y based on features and no treatmnent effects
simple_reg_y0_test = simple_reg.predict(xt0_test)


# In[33]:


# to merge x and t = 1
xt1_test = np.concatenate([x_test, np.ones_like(t_test)], axis=1) 

# to predict y based on features and treatmnent effects
simple_reg_y1_test = simple_reg.predict(xt1_test)


# In[34]:


# to compute ITEs
simple_reg_te_test = simple_reg_y1_test - simple_reg_y0_test 


# In[35]:


# to use the pre-defined 'pehe' function
pehe_test = pehe(ite_test, simple_reg_te_test)


# In[36]:


results = []
results.append(['Simple Learner', pehe_test])

cols = ['Method', 'PEHE test']

df = pd.DataFrame(results, columns=cols)
df


# In[37]:


# to plot feature importance by calling pre-defined function
importances, indices = feat_imp(simple_reg)


# In[38]:


names = ihdp.drop(['yf', 'ycf', 'ite'], axis=1).columns.values


# In[39]:


# to print the feature ranking
print("Feature ranking:")
for f in range(x_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, indices[f],  importances[indices[f]]))


# In[81]:


# to plot the feature importances of the forest
plt.rcParams["figure.figsize"] = (12, 6)
plt.title("Simple Learner - Feature importances")
plt.bar(names,importances, color="g", edgecolor="black")


# #### Propensity score re-weighting

# In[41]:


# to train a classifier to predict propensity scores
prop_clf = RandomForestClassifier()


# In[42]:


scores = cross_val_score(prop_clf, x_train, t_train['t'], cv=10, scoring = "f1")
print("f1 score: %.2f" % (scores.mean()))


# In[43]:


param_grid = { 
    'n_estimators': [15, 20, 25, 50, 100],
    'max_depth' : [2, 3, 4, 5, 6],
}


# In[44]:


# for hyperparameter tuning
CV1_model = GridSearchCV(estimator=prop_clf, param_grid=param_grid, cv= 10)
CV1_model.fit(x_train, t_train['t'])


# In[45]:


# to find the best suited parameters for the classification model
CV1_model.best_params_


# In[46]:


prop = RandomForestClassifier(n_estimators = 20, max_depth = 5)


# In[47]:


scores = cross_val_score(prop, x_train, t_train['t'], cv=10, scoring = "f1")
print("f1 score: %.2f" % (scores.mean()))


# In[48]:


# to get the sample weights
weights = get_ps_weights(prop, x_train, t_train)


# In[49]:


# to build a Random Forest regression model
model = RandomForestRegressor()


# In[50]:


scores = cross_val_score(model, x_train, y_train, cv=10, scoring = "neg_mean_squared_error")
print("Negative mean squared error: %.2f +/- %.2f" % (scores.mean(), scores.std()))


# In[51]:


param_grid_ = { 
    'n_estimators': [25, 50, 100, 150, 200],
    'max_depth' : [4, 5, 6, 7, 8, 9, 10, 11, 12],
}


# In[52]:


# for hyperparameter tuning
CV_model = GridSearchCV(estimator=model, param_grid=param_grid_, cv= 10)
CV_model.fit(x_train, y_train)


# In[53]:


# to find the best suited parameters for the classification model
CV_model.best_params_


# In[54]:


# to train the regressor based on the propensity model
ipsw = RandomForestRegressor(n_estimators = 200, max_depth = 5)
ipsw.fit(xt_train, y_train, sample_weight= weights)


# In[55]:


scores = cross_val_score(ipsw, x_train, y_train, cv=10, scoring = "neg_mean_squared_error")
print("Negative mean squared error: %.2f +/- %.2f" % (scores.mean(), scores.std()))


# In[56]:


# to make predictions
ipsw_y0_test = ipsw.predict(np.concatenate([x_test, np.zeros_like(t_test)], axis=1))
ipsw_y1_test = ipsw.predict(np.concatenate([x_test, np.ones_like(t_test)], axis=1))


# In[57]:


# to compute ITEs
ipsw_te_test = ipsw_y1_test - ipsw_y0_test


# In[58]:


# to use the pre-defined 'pehe' function
ipsw_pehe_test = pehe(ite_test, ipsw_te_test)


# In[59]:


results = []
results.append(['IPSW', ipsw_pehe_test])

cols = ['Method', 'PEHE test']

df = pd.DataFrame(results, columns=cols)
df


# In[60]:


# to plot feature importance by calling pre-defined function
importances, indices = feat_imp(ipsw)


# In[61]:


# to print the feature ranking
print("Feature ranking:")
for f in range(x_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, indices[f],  importances[indices[f]]))


# In[62]:


# to plot the feature importances of the forest
plt.rcParams["figure.figsize"] = (12, 6)
plt.title("IPSW - Feature importances")
plt.bar(names,importances, color="y", edgecolor="black")


# #### Advanced CATE estimators

# In[63]:


# to implement Random Forest a cross-validation regression model that automatically chooses the hyperparameters
first_stage = GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid={
                        'max_depth': [3, 4, 5, 6, 7],
                        'n_estimators': (50, 100, 150, 200)
                    }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
mod_y = first_stage.fit(x_train, y_train).best_estimator_


# In[64]:


# to implement Random Forest a cross-validation classification model that automatically chooses the hyperparameters
first_stage0 = GridSearchCV(
                estimator=RandomForestClassifier(),
                param_grid={
                        'max_depth': [3, 4, 5, 6, 7],
                        'n_estimators': (50, 100, 150, 200)
                    }, cv=10, n_jobs=-1, scoring='f1'
                )
mod_y0 = first_stage0.fit(x_train, t_train['t']).best_estimator_


# In[65]:


# to instantiate X learner
X_learner = XLearner(models=mod_y, propensity_model=mod_y0)


# In[66]:


# to train X_learner on the train set
X_learner.fit(y_train, t_train['t'], X=x_train)


# In[67]:


# to estimate treatment effects on test data
x_estimate = X_learner.effect(x_test)


# In[68]:


# to use the pre-defined 'pehe' function
xl_pehe_test = pehe(ite_test,x_estimate)


# In[69]:


results = []
results.append(['XL', xl_pehe_test])

col = ['Method', 'PEHE test']

df = pd.DataFrame(results, columns=col)
df


# #### Results comparison

# In[70]:


# to compare all the results
results = []
results.append(['Simple Learner', pehe_test])
results.append(['IPW', ipsw_pehe_test])
results.append(['XL', xl_pehe_test])

cols = ['Method', 'PEHE test']

df = pd.DataFrame(results, columns=cols)
df

