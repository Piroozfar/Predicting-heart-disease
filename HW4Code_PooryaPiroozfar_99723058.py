#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

df =  pd.read_csv("F:\\IUST\\ترم 2\\داده کاوی\\HW\\HW4\\processed.cleveland.csv", sep=',', header=None,
                  names=['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak','slope', 'ca', 'thal', 'hd'])

df


# In[2]:


print(df.ca.mode())
print(df.thal.mode())

df['ca'].replace('?','0',inplace=True)
df['thal'].replace('?','3',inplace=True)

df['cp']=df['cp'].apply(str)
df['restecg']=df['restecg'].apply(str)
df['slope']=df['slope'].apply(str)

df


# In[3]:


df.info()


# In[4]:


#split dataset in features and target variable
feature_cols = ['age', 'sex', 'cp' , 'restbp', 'chol', 'fbs' ,'restecg' ,'thalach', 'exang' ,'oldpeak',
'slope', 'ca', 'thal']
X = df[feature_cols] # Features
y = df.hd # Target variable
X


# In[5]:


X=pd.get_dummies(X)
X


# In[6]:


from sklearn.model_selection import train_test_split 
from sklearn import metrics 
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% test
X_train


# In[7]:


from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
# create a scaler object
scaler = preprocessing.MinMaxScaler()
# fit and transform the data
X_train_norm = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns,index=X_train.index)
X_test_norm = pd.DataFrame(scaler.fit_transform(X_test.values), columns=X_test.columns,index=X_test.index)

X_train_norm


# # Build first Tree

# In[8]:


from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

# Create Decision Tree classifer object
clf = DecisionTreeClassifier( random_state = 42)
# Train Decision Tree Classifer
clf = clf.fit(X_train_norm,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test_norm)


# In[9]:


import graphviz
# DOT data

dot_data = tree.export_graphviz(clf, out_file=None,feature_names=X_test_norm.columns,
                               filled=True, rounded=True ,special_characters=True)

# Draw graph

graph = graphviz.Source(dot_data, format="png") 
graph.render("decision_tree_graphivz")
graph


# In[13]:


fig = plt.figure(figsize=(50,50))
_ = tree.plot_tree(clf, filled=True,feature_names=X_test_norm.columns)
plt.savefig('Desktop/tree.png')


# In[10]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # Pruning

# In[11]:


from sklearn.metrics import accuracy_score
import seaborn as sns
path=clf.cost_complexity_pruning_path(X_train_norm , y_train)
alphas=path['ccp_alphas']
print(alphas)
print(len(alphas))

accuracy_train,accuracy_test=[],[]
for i in alphas:
    tree=DecisionTreeClassifier(ccp_alpha=i)
    
    tree.fit(X_train_norm,y_train)
    y_train_pred=tree.predict(X_train_norm)
    y_test_pred=tree.predict(X_test_norm)
    
    accuracy_train.append(accuracy_score(y_train,y_train_pred))
    accuracy_test.append(accuracy_score(y_test,y_test_pred))
    
    
sns.set()    
plt.figure(figsize=(14,7))
sns.lineplot(y=accuracy_train,x=alphas,label="Train Accuracy")
sns.lineplot(y=accuracy_test,x=alphas,label="Test Accuracy")
plt.show()

#از نمودار می توان یافت که آلفای حدود 0.02 حداکثر دقت تست است


# # 5_Fold

# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

mean_accuracy = []
var_accuracy = []
mean = []
var = []
for i in alphas:
    clf = DecisionTreeClassifier(ccp_alpha=i)
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=5, scoring='accuracy')
    mean_accuracy.append((scores.mean()))
    var_accuracy.append((scores.var()))
    mean.append((i,scores.mean()))
    var.append((i,scores.var()))

print(mean)
print(var)


sns.set()    
plt.figure(figsize=(14,7))
sns.lineplot(y=mean_accuracy,x=alphas,label="mean Accuracy")
sns.lineplot(y=var_accuracy,x=alphas,label="var Accuracy")
plt.show()


# In[16]:


def run_cross_validation_on_trees(X, y, alphas, cv=5, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for i in alphas:
        tree_model = DecisionTreeClassifier(ccp_alpha=i)
        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores
  
# function for plotting cross-validation results
def plot_cross_validation_on_trees(D, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(D, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(D, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(D, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('alpha', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(D)
    ax.legend()

sm_tree_alphas = alphas
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, y_train, sm_tree_alphas)

# plotting accuracy
plot_cross_validation_on_trees(sm_tree_alphas, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 
                               'Accuracy per decision tree alpha on training data')


# # Build final Tree

# In[29]:


from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# Create Decision Tree classifer object
clf2 = DecisionTreeClassifier(ccp_alpha=0.023450413223140495, random_state = 42)
# Train Decision Tree Classifer
clf2 = clf2.fit(X_train_norm,y_train)

#Predict the response for test dataset
y_test_pred = clf2.predict(X_test_norm)
print(accuracy_score(y_test,y_test_pred))
print(confusion_matrix(y_test, y_test_pred))  


# In[30]:


import graphviz

dot2_data = tree.export_graphviz(clf2, out_file=None,feature_names=X_test_norm.columns,
                               filled=True, rounded=True ,special_characters=True)

# Draw graph
graph = graphviz.Source(dot2_data, format="png") 
graph


# In[ ]:




