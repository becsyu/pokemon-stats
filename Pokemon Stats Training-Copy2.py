#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# We started with a dataset containing 800 Pokemon game stats and a categorical variable "Type", aiming to visualize the distribution of Pokemon as well as explaining how those of the same "Type" are grouped together. Initial dataframe was treated with a column change - "Type 2" had nearly half the values missing, thus it was replaced with binary 0/1, for data analysis.
# 
# 
# Initial data exploration shows that the distribution of types is binomial, which coincides with the distribution of the overall strength of 800 pokemons, measured by the sum of all stats (attack, defense, sp. atk, sp. def and speed). Each stat by itself shows a right-skewed normal distribution, leaving some outliers that are possibly game specials - "legendary pokemons". Having a secondary type present shows a lift in overall strength by 10.8%.
# 
# 
# To further examine whether, and if so, which attributes might explain the type, we reduced dimensions using PCA and plotted 18 types on a 2D graph. Although the two principal components are hard to intepret, we were able to cover 74.5% of the variance. We further attempted using PCA to find a fit between features and target, by logistic regression model. The result was of low accurancy - only 20%.
# 
# This study shows that game design is sophisticated - when assigning attributes to certain characters, designers must think of whether each attribute fits the distribution, as well as overall strength within each group. Especially with special events, game companies often rewards players with special items and create special levels. These creations are the outliers in game data and must be treated with caution too.

# # Introduction
# In game design we often encouter character/monster stats that are inconsistent across the game - players might experience an extremely difficult starting level, for example. Each character, move and monster has at least 6 attributes (defense, attack, agility etc.) so the question is how do we design such characters whose attributes and level (being strong or weak in the game) are aligned, while all of them come together form a distribution (perhaps a normal distribution) that fits the best for the game play? 
# 
# In this study I will study a dataset of 800 pokemons (out of the 809 most-up-to-date number) to understand how game developers from Nintendo and Game Freak design these infamous creatures and balance their attributes out. For simplicity, I'm using a dataset of pokemons at their beginning level.  

# # Data Overview
# This data set includes 721 Pokemon (with some pokemons of 2 versions, total 800 rows), including their number, name, first and second type, and basic stats: HP, Attack, Defense, Special Attack, Special Defense, and Speed. The dataset was obtained from kaggle, by Alberto Barradas through public domains of pokemon.com, pokemondb and bulbapedia. Link as below to the kaggle kennel:
# https://www.kaggle.com/abcsds/pokemon

# In[2]:


#import important libraries
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np


# In[3]:


#read dataset 
df = pd.read_csv('~/Downloads/Pokemon.csv', index_col=0)


# In[12]:


#check attributes
df.head(10)


# In[5]:


#check types of pokemons
df.groupby('Type 1').size()


# In[6]:


df.groupby('Type 2').size()


# In[9]:


plt.figure(figsize=(12,5))
chart0=sns.countplot(x="Type 1", data=df)


# In[10]:


plt.figure(figsize=(12,5))
sns.countplot(x="Type 2", data=df)


# Pokemons have a primary type stored in 'Type 1' column and a possible secondary type stored in 'Type 2'. Total 18 types.
# We also look at the column 'Total', which represents the total strength of such pokemon - the higher the better. 

# In[6]:


#Distribution of Total
sns.set(color_codes=True)
sns.distplot(df['Total'])


# Total seems to be binomial. 

# In[7]:


#We notice there are NaN in our data. Let's take a look
# checking the percentage of missing values in each variable
df.isnull().sum()/len(df)*100


# Almost half of the Pokemons do not have a Type 2. We will change this column to a binary column. 

# In[8]:


#We also needs to check the variance of the attributes to see if it makes sense to keep all of them.
df.var()/len(df)*100


# Since the column 'Total' is our output, such variance is within our expectation. Major attributes include HP, Attack, Defense, Sp.ATk, Sp.Def, and Speed all within a variance between 80 - 133. We will keep them all. This is expected, given it's Nintendo...

# # Data Prep
# Here we convert 'Type 2' to a boleen dummy column. '0' if this Pokemon does not have a Type 2 attribue and '1' if it does.

# In[9]:


#change NaN to 0
df['Type 2'] = df['Type 2'].fillna(0) 


# In[10]:


#create a new list to change non-NaN values to 1
Type_2 = []
for i in df['Type 2']:
    if i == 0:
        Type_2.append(0)
    else:
        Type_2.append(1)
        
#replace old column 'Type 2' with new binary column        
df['Type 2'] = Type_2


# In[11]:


df.head()


# Much better. Now let's take a look at the distribution of each attribute.

# In[12]:



#Histogram of attribute 'Attack' and 'Defense'
sns.distplot(df['Attack'])
sns.distplot(df['Defense'])


# In[13]:


#A right-skewed normal distribution graph for both attributes.
#Similarly let's look at the distribution of other attributes:


# In[14]:


sns.distplot(df['HP'])


# In[15]:


sns.distplot(df['Speed'])


# In[16]:


sns.distplot(df['Sp. Atk'])
sns.distplot(df['Sp. Def'])


# It appears that all attributes follow a right-skewed pattern. We can further explore the statistical relationships between these attributes in next steps.

# # Preliminary Analysis
# Overviews of categorical plots, statistical estimation and 2D correlations

# In[17]:


#"wide-form" plots of the dataframe
sns.catplot(data=df, kind="box");


# This proves that all attributes are right skewed, so is the outcome 'Total'. The outliers here are likely the 'legendary' kinds - unfortunately not included in this dataset.
# 
# Type 1 is a categorical column that I left it untreated so far. I'm guessing the type of Pokemon has an affect on its total strength too. Let's take a look at 'Total' and 'Type 1'. 

# In[18]:


df.groupby('Type 1', sort=True).mean()


# In[14]:


df['Total'].mean()


# In[19]:


#Table gives the mean of each type but how much variance each type represents? 
plt.figure(figsize=(10,5))
chart1=sns.catplot(x="Type 1", y="Total", kind="bar", data=df)
chart1.set_xticklabels(rotation=60)


# Looks like the developers really favour Dragon-type Pokemons!

# In[20]:


#Now let's breakdown and see what makes up the 'Total'
#A brief overlook of the correlations between each attribute
df.corr()


# Since 'Total' is our outcome and all other attritbues have high correlations with it, this proves our heuristic guess. All other attributes do not have a correlation higher than 0.5 except 'Sp.Def' and 'Sp.Atk'. We could potentially drop one of them, but we will keep them both for now.

# In[21]:


#Let's take a look at the 2D plots of 'Sp.Def' and 'Sp.Atk':
sns.relplot(x="Sp. Atk", y="Sp. Def",data=df);


# In[22]:


#Overall we can see that the higher Sp. Atk a Pokemon has, the higher Sp. Def it has.
#It might make more sense to see if different type would give any more clues.

sns.relplot(x="Sp. Atk", y="Sp. Def",hue="Type 1",data=df);


# We have 18 types... That's kinda crazy to visualize over one scatterplot. Let's take a look at whether having a secondary type would make a difference.

# In[23]:


sns.relplot(x="Sp. Atk", y="Sp. Def",hue="Type 2",data=df);


# In[24]:


#Out of curiosity... Is the strength of pokemon higher when there is a secondary type present?
df.groupby('Type 2', sort=True).mean().sort_values('Total',ascending=False).Total


# In[25]:


(456.6-412)/412*100


# The presence of 'Type 2' has an overall 10.8% lift in overall strength. Perhaps with the help of other attributes we can explain the help of secondary type better.

# In[26]:


#Which pokemon types are more likely to get a secondary type?
chart2=sns.catplot(x="Type 1", kind="count", hue="Type 2", data=df);
chart2.set_xticklabels(rotation=60)


# Bug, Rock and Steel types are way more likely to get a secondary type!
# It is hard to cluster Pokemons based on just any of the two variables. Due to our limited dimensionality plotting, we will consider methods to lower dimensionality by grouping variables together.

# # Data Analysis
# Can we use a model to explain the relationship between total strength and all other attributes?

# In[27]:


#Can we explain everything with our best friend - linear regression?
import statsmodels.api as sm


# In[28]:


#First let's separate the predictors and the target - in this case -- Total.
df1 = df.drop(columns=['Total'])


# In[29]:


df1


# I will use PCA to plot the 6 attribues on a dimensional data to find if they can explain the pattern of types of Pokemon.

# In[164]:


#Lower dimensionality approach using PCA
#import standard scaler package
from sklearn.preprocessing import StandardScaler


# In[262]:


features = ['Total','Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

#separating out the features
x = df.loc[:, features].values
y = df.loc[:,['Type 1']].values
#standardizing the features
x = StandardScaler().fit_transform(x)


# In[254]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[255]:


principalDf


# In[256]:


target=df.iloc[:, 1]
target.index = range(800)
target


# In[257]:


finalDf = pd.concat([principalDf, target], axis=1)
finalDf


# Perfect final PCA table! The two principal components here don't necessarily make any sense except for mapping out the classes and hopefully separating out the classes.

# In[258]:


sns.relplot(x="principal component 1", y="principal component 2",hue="Type 1",data=finalDf);


# In[259]:


#The plot did not seem to separate out types too well. Let's see if accuracy of this model:
pca.explained_variance_ratio_


# Together these two principal components contain 74.5% of the information, better than I thought!
# 
# Let's split data into test and training to test a logistic regression model using PCA.

# In[296]:


#Split dataset
from sklearn.model_selection import train_test_split
dat = df.loc[:, features].values
dat_target = target
x_train, x_test, y_train, y_test = train_test_split(dat, dat_target, test_size=0.2, random_state=0)

#Fit on training set only
scaler.fit(x_train)

#Standardize using scaler
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[297]:


#Make an instance of the model. This means that the minimum number of principal components chosen have 95% of the variance retained.
pca=PCA(.95)


# In[298]:


#Fit PCA on trainig set 
pca.fit(x_train)


# In[299]:


#Now transform the training and the test sets... aka mapping
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)


# In[300]:


#Apply logistic regression to the transformed data
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs') #faster!


# In[301]:


#Train the model on the data
logisticRegr.fit(x_train, y_train)


# In[302]:


#Predict for one observation
logisticRegr.predict(x_test[0].reshape(1,-1))


# Unfortunately a wrong prediction...! Let's see how accurate this model is on test data.

# In[303]:


logisticRegr.score(x_test, y_test)


# Logistic regression is clearly not the answer to our question. Perhaps with the help of other models this could be better explained.

# # Conclusion
# This study has many limitations, starting with the dataset - it had only 800 rows, with 18 unique target types, and 6 features/attributes. Ideally more rows could improve our model fit. The dataset was also a perfectly designed in-game stats. Every statistical estimations at initial exploration showed perfect statistical distribution or scores, meaning that it was not giving any extra information whether any attribute has a higher effect. 
# 
# Future study should start with multivariant cluster methods, to better examine the relations between attributes that form the types. Also it could be fun to look at the learning curve of each pokemon - from level 1 to 100 how do the stats change and do they follow the same pattern as their beginning level.

# # Reference
# https://www.kaggle.com/abcsds/pokemon
# https://scikit-learn.org/stable/index.html
# https://seaborn.pydata.org/index.html
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html

# In[ ]:




