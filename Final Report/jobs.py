#!/usr/bin/env python
# coding: utf-8

# ## Jobs

# In[3]:


from sklearn.model_selection import train_test_split, cross_val_score # to split the data into train and test datasets 
from sklearn.preprocessing import StandardScaler # use of StandardScaler to standarise the dataset
import numpy as np # library of mathematical operations
import pandas as pd  # for data anlysis and manipulation
import matplotlib.pyplot as plt # to display charts
import seaborn as sns # datisualisation library
from sklearn.model_selection import GridSearchCV # library function for cross-validation
from econml.metalearners import XLearner # advanced CATE estimator
from sklearn.ensemble import RandomForestClassifier # meta estimator for fitting decision trees
from sklearn.tree import DecisionTreeClassifier # tree-structed classifier used for model building

from helper_functions import feat_imp, get_ps_weights, policy_risk # self-defined funtions 


# ### Loading the dataset

# In[4]:


# to load the dataset
jobs = pd.read_csv("https://raw.githubusercontent.com/dmachlanski/CE888_2022/main/project/data/jobs.csv", delimiter=",")
jobs


# ### Exploring the dataset

# In[5]:


# to print information about the dataset
jobs.info()


# In[6]:


jobs['y'].value_counts(1)


# In[7]:


jobs['t'].value_counts(1)


# In[8]:


nrow, ncol = jobs.shape
print(f'There are {nrow} rows and {ncol} columns')


# In[9]:


x = jobs[["x1", 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
         'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17']]


# In[10]:


# reshaping the single dimension vectors into 2D vectors as some methods prefer this representaion of data
T = jobs['t'].values.reshape(-1, 1)
Y = jobs['y'].values.reshape(-1, 1)
e = jobs['e']


# In[11]:


# to plot a boxplot of each feature in the dataset and check if the features vary in scale
plt.figure(figsize=(30,15))
_ = sns.boxplot(data = x)


# In[ ]:


plt.savefig("jobs_boxplot.pdf")


# In[12]:


# to draw histogram and look at the distribution of values of each feature
jobs.hist(bins=50, figsize=(20,20))


# In[ ]:


plt.savefig("jobs_histogram.pdf")


# In[13]:


sns.countplot(x = "y", data = jobs)


# In[ ]:


plt.savefig("jobs_y.pdf")


# In[14]:


sns.countplot(x = "t", data = jobs)


# In[ ]:


plt.savefig("jobs_t.pdf")


# In[15]:


# to calculate the correlations between each pair of variables
corr = x.corr()

# to plot a heatmap of the correlations between pairs of features
sns.set(rc = {'figure.figsize':(20,15)})
sns.heatmap(corr, annot = True)


# In[ ]:


plt.savefig("jobs_heatmap.pdf")


# ### Data Pre-Processing

# In[16]:


# to split the data into train and test datasets
train_set, test_set = train_test_split(jobs, test_size=0.2, random_state=50)  
print(len(train_set), len(test_set))


# In[17]:


x_train = train_set.drop(['t', 'y', 'e'], axis = 1)
x_test = test_set.drop(['t', 'y', 'e'], axis = 1)


# In[18]:


y_train = train_set['y']
y_test = test_set['y']


# In[19]:


t_train = train_set['t']
t_test = test_set['t']


# In[20]:


e_train = train_set['e']
e_test = test_set['e']


# In[21]:


# to standarise the dataset i.e. mean = 0 and s.d. = 1
scaler_x = StandardScaler() 
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)


# In[22]:


xtrain = pd.DataFrame(x_train)
ttrain = pd.DataFrame(t_train)


# In[23]:


xtest = pd.DataFrame(x_test)
ttest = pd.DataFrame(t_test)


# ## Modelling

# #### Simple learners

# In[24]:


# to fit a simple Random Forest classification model
clf = RandomForestClassifier()


# In[25]:


scores = cross_val_score(clf, x_train, y_train, cv=10, scoring = "f1")
print("f1 score: %.2f +/- %.2f" % (scores.mean(), scores.std()))


# In[26]:


param_grid = { 
    'n_estimators': [25,50,100,150],
    'max_depth' : [3,4,5,6,7,8],
}


# In[27]:


# for hyperparameter tuning
CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 10)
CV_clf.fit(x_train, y_train)


# In[28]:


# to find the best suited parameters for the classification model
CV_clf.best_params_


# In[29]:


simple_clf = RandomForestClassifier(n_estimators = 100, max_depth = 4)


# In[30]:


xt_train = np.concatenate([xtrain, ttrain], axis=1)

# to build the model on the traing data
simple_clf.fit(xt_train, y_train)


# In[31]:


scores = cross_val_score(simple_clf, xt_train, y_train, cv = 10, scoring = "f1")
print("f1 score: %.2f +/- %.2f" % (scores.mean(), scores.std()))


# In[32]:


# to merge x and t = 1
xt1_test = np.concatenate([xtest, np.ones_like(ttest)], axis=1)

# to predict y based on features and treatmnent effects
simple_clf_y1_test = simple_clf.predict(xt1_test)


# In[33]:


# to use pre-defined policy risk function
policy_risk1 = policy_risk(simple_clf_y1_test, y_test, t_test, e_test)


# In[34]:


results = []
results.append(['Simple Learner', policy_risk1])

cols = ['Method', 'Policy risk']

df = pd.DataFrame(results, columns=cols)
df


# In[35]:


# to plot feature importance by calling pre-defined function
importances, indices = feat_imp(simple_clf)


# In[36]:


names = jobs.drop(['y', 'e'], axis=1).columns.values


# In[37]:


# to print the feature ranking
print("Feature ranking:")
for f in range(x_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, indices[f],  importances[indices[f]]))


# In[38]:


# to plot the feature importances of the forest
plt.rcParams["figure.figsize"] = (12, 6)
plt.title("Simple Learner - Feature importances")
plt.bar(names,importances, color="m", edgecolor="black")


# #### Propensity score re-weighting

# In[39]:


# to train a classifier to predict propensity scores
prop_clf = DecisionTreeClassifier()


# In[40]:


scores = cross_val_score(prop_clf, x_train, t_train, cv=10, scoring = "f1")
print("f1 score: %.2f" % (scores.mean()))


# In[41]:


param_grid = { 
    'max_depth' : [2, 3, 4, 5, 6],
}


# In[42]:


# for hyperparameter tuning
CV1_model = GridSearchCV(estimator=prop_clf, param_grid=param_grid, cv= 10)
CV1_model.fit(x_train, t_train)


# In[43]:


# to find the best suited parameters for the classification model
CV1_model.best_params_


# In[44]:


prop = DecisionTreeClassifier(max_depth = 2)


# In[45]:


scores = cross_val_score(prop, x_train, t_train, cv=10, scoring = "f1")
print("f1 score: %.2f" % (scores.mean()))


# In[46]:


# to get the sample weights
weights = get_ps_weights(prop, x_train, t_train)


# In[47]:


# to build a Random Forest classification model
model = RandomForestClassifier()


# In[48]:


scores = cross_val_score(model, x_train, y_train, cv=10, scoring = "f1")
print("f1 score: %.2f +/- %.2f" % (scores.mean(), scores.std()))


# In[49]:


param_grid_ = { 
    'n_estimators': [25, 50, 100, 150, 200],
    'max_depth' : [4, 5, 6, 7, 8, 9, 10, 11, 12]
}


# In[50]:


# for hyperparameter tuning
CV_model = GridSearchCV(estimator=model, param_grid=param_grid_, cv= 10)
CV_model.fit(x_train, y_train)


# In[51]:


# to find the best suited parameters for the classification model
CV_model.best_params_


# In[52]:


# to train the classifier based on the propensity model
ipsw = RandomForestClassifier(n_estimators = 50, max_depth = 5)
ipsw.fit(xt_train, y_train, sample_weight= weights)


# In[53]:


scores = cross_val_score(ipsw, x_train, y_train, cv=10, scoring = "f1")
print("f1 score: %.2f +/- %.2f" % (scores.mean(), scores.std()))


# In[54]:


# to make predictions
ipsw_y1_test = ipsw.predict(np.concatenate([xtest, np.ones_like(ttest)], axis=1))


# In[55]:


# to merge x and t = 1
xt1_test = np.concatenate([xtest, np.ones_like(ttest)], axis=1)

# to predict y based on features and treatmnent effects
ipsw_y1_test = ipsw.predict(xt1_test)


# In[56]:


# to use pre-defined policy risk function
policy_risk2 = policy_risk(ipsw_y1_test, y_test, t_test, e_test)


# In[57]:


results = []
results.append(['IPSW', policy_risk2])

cols = ['Method', 'Policy Risk']

df = pd.DataFrame(results, columns=cols)
df


# In[58]:


# to plot feature importance by calling pre-defined function
importances, indices = feat_imp(ipsw)


# In[59]:


# to print the feature ranking
print("Feature ranking:")
for f in range(x_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, indices[f],  importances[indices[f]]))


# In[60]:


# to plot the feature importances of the forest
plt.rcParams["figure.figsize"] = (12, 6)
plt.title("IPSW - Feature importances")
plt.bar(names,importances, color="c", edgecolor="black")


# #### Advanced CATE estimators

# In[61]:


# to implement Random Forest a cross-validation classification model that automatically chooses the hyperparameters
first_stage = GridSearchCV(
                estimator=RandomForestClassifier(),
                param_grid={
                        'max_depth': [3, 4, 5, 6, 7],
                        'n_estimators': (50, 100, 150, 200)
                    }, cv=10, n_jobs=-1, scoring='f1'
                )
mod_y = first_stage.fit(x_train, y_train).best_estimator_


# In[62]:


# to implement Decision Tree Forest a cross-validation classification model that automatically chooses the hyperparameters
first_stage0 = GridSearchCV(
                estimator=DecisionTreeClassifier(),
                param_grid={
                        'max_depth': [2, 3, 4, 5, 6, 7]
                    }, cv=10, n_jobs=-1, scoring='f1'
                )
mod_t = first_stage0.fit(x_train, t_train).best_estimator_


# In[63]:


# to instantiate X learner
X_learner = XLearner(models=mod_y, propensity_model=mod_t)


# In[64]:


# to train X_learner
X_learner.fit(y_train, t_train, X=x_train)


# In[65]:


# to estimate treatment effects on test data
x_estimate = X_learner.effect(x_test)


# In[66]:


# to use pre-defined policy risk function
policy_risk3 = policy_risk(x_estimate, y_test, t_test, e_test)


# In[67]:


results = []
results.append(['XL', policy_risk3])

col = ['Method', 'Policy Risk']

df = pd.DataFrame(results, columns=col)
df


# #### Results comparison

# In[68]:


# to compare all the results
results = []
results.append(['Simple Learner', policy_risk1])
results.append(['IPW', policy_risk2])
results.append(['XL', policy_risk3])

cols = ['Method', 'Policy Risk']

df = pd.DataFrame(results, columns=cols)
df

