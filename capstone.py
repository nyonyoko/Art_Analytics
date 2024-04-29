# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:02:31 2022

@author: yiyun
"""

#%% import packages
import random
# Seed the random number generator with the number 17105116
random.seed(17105116)
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import expit # this is the logistic sigmoid function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
#%% question 1
# Read in .csv file
df = pd.read_csv("theData.csv")
data = np.genfromtxt("theData.csv", delimiter = ",")

# Select columns of interest
dclassical = df.iloc[:, :35]
dmodern = df.iloc[:, 35:70]


# Get a copy of the data collapsed into one dimension
classicalSample = data[:,:35].flatten()
modernSample = data[:,35:70].flatten()

# Perform t-test
t_statistic, p_value = ttest_ind(classicalSample, modernSample)

# Print results
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print(np.mean(classicalSample))
print(np.mean(modernSample))

print(np.median(classicalSample))
print(np.median(modernSample))

print(pd.Series(classicalSample).mad())
print(pd.Series(modernSample).mad())

#%% question 2

# Select columns of interest
dnonhuman = df.iloc[:, 35:70]

# Get a copy of the data collapsed into one dimension
nonhumanSample = data[:,70:91].flatten()

# Perform t-test
t_statistic, p_value = ttest_ind(modernSample, nonhumanSample)

# Print results
print("t-statistic:", t_statistic)
print("p-value:", p_value)
print(np.mean(nonhumanSample))
print(np.median(nonhumanSample))
print(pd.Series(nonhumanSample).mad())

#%% Question 3

# Select rows of interest
maleList = list(np.where(data[:,216]==1))
maleSample = data[:,0:91]
maleSample = maleSample[maleList]
maleSample = np.asarray(maleSample).flatten()
femaleList = list(np.where(data[:,216]==2))
femaleSample = data[:,0:91]
femaleSample = femaleSample[femaleList]
femaleSample = np.asarray(femaleSample).flatten()

# Perform t-test
t_statistic, p_value = ttest_ind( stats.zscore(femaleSample),  stats.zscore(maleSample))

# Print results
print("t-statistic:", t_statistic)
print("p-value:", p_value)

print(np.mean(femaleSample))
print(np.mean(maleSample))

print(np.median(femaleSample))
print(np.median(maleSample))


#%% Question 4

# Select rows of interest
noEdList = list(np.where(data[:,218]==0))
noEdSample = data[:,0:91]
noEdSample = noEdSample[noEdList]
noEdSample = np.asarray(noEdSample).flatten()
edList = list(np.where(data[:,218]>0))
edSample = data[:,0:91]
edSample = edSample[edList]
edSample = np.asarray(edSample).flatten()




# Perform t-test
t_statistic, p_value = ttest_ind(stats.zscore(edSample), stats.zscore(noEdSample))

# Print results
print("t-statistic:", t_statistic)
print("p-value:", p_value)

print(np.mean(edSample))
print(np.mean(noEdSample))

print(np.median(edSample))
print(np.median(noEdSample))


ed = data[data[:, 218] > 0]
ed = ed[:, 0:91][~np.isnan(ed[:, 0:91]).any(axis=1)]

noed = data[data[:, 218] == 0]
noed = noed[:, 0:91][~np.isnan(noed[:, 0:91]).any(axis=1)]

ed  = np.mean(ed[:, 0])
noed = np.mean(noed[:, 0])

u4,p4 = stats.mannwhitneyu(ed,noed) 
print(u4)
print(p4)

#%% Question 5

#Build a regression model for prediction

#predict art preference ratings from energy ratings only
#use corss-validation to avoid overfitting
#characterize how well the model predicts art preference ratings
#to use cross-validation, we use part of the data as training data to build the model, 
#and the rest of the data as test data to predict 

energy = data[:,91:182][~np.isnan(data[:,91:182]).any(axis=1)]
zScore5 = stats.zscore(energy)
pca5 = PCA().fit(zScore5)
rotatedData5 = pca5.fit_transform(zScore5)*-1
combine5 = np.column_stack([data[:, 0:91],data[:, 91:182]])

#Impute all nan with mode of each column
fullData = SimpleImputer(missing_values=np.nan, strategy = "most_frequent")
fullData.fit(combine5)
combineT5 = fullData.transform(combine5)
y5 = pd.DataFrame(combineT5).iloc[:,:91].to_numpy() #rating


#Test-train split
x_train5, x_test5, y_train5, y_test5 = train_test_split(rotatedData5, y5, test_size = 0.2, random_state=17105116)


#Linear Regression
model5 = LinearRegression().fit(x_train5, y_train5) 
y_pred5 = model5.predict(x_test5)
MSE_5 = metrics.mean_squared_error(y_test5, y_pred5)
print('RMSE of the regression model is ', math.sqrt(MSE_5))
rsq5 = model5.score(x_test5, y_test5)
print(rsq5)

#%% Question 6

data6 = data[:,91:182]
D1 = np.mean(data,axis=0) # take mean of each column
D2 = np.median(data,axis=0) # take median of each column
D3 = np.std(data,axis=0) # take std of each column
D4 = np.corrcoef(data6[:,0],data6[:,1]) # correlate IQ and hours worked
D5 = np.corrcoef(data6.T) #The full correlation matrix 

x6 = np.nan_to_num(data[:,91:182+215:221])
y6 = np.nan_to_num(data[:,0:91])
x_train6, x_test6, y_train6, y_test6 = train_test_split(x6, y6, test_size = 0.2, random_state=17105116)
model6 = LinearRegression().fit(x6,y6)

y66 = model6.predict(x6)
MSE_6 = metrics.mean_squared_error(y6, y66)
print('RMSE of multiple regression model is ', math.sqrt(MSE_6))
rsq6 = model6.score(x6,y6) 
print(rsq6)


#%% Question 7
 
art7 = np.mean(data[:,0:91],axis=0)
energy7 = np.mean(data[:,91:182],axis=0)
data7 = np.column_stack((art7,energy7))

# clustering with PCA
# Z-score the data:
zscoredData7 = stats.zscore(data7)

# Initialize PCA object and fit to our data:
pca7 = PCA().fit(zscoredData7)

# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals7 = pca7.explained_variance_

# Loadings (eigenvectors): Weights per factor in terms of the original data.
loadings7 = pca7.components_*-1

# Rotated Data - simply the transformed data:
origDataNewCoordinates7 = pca7.fit_transform(zscoredData7)*-1
x7 = np.column_stack((origDataNewCoordinates7[:,0],origDataNewCoordinates7[:,1]))
# Init:
numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
sSum = np.empty([numClusters,1])*np.NaN # init container to store sums

# Compute kMeans for each k:
for ii in range(2, numClusters+2): # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters = int(ii),random_state=17105116).fit(x7) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(x7,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = sum(s) # take the sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,250)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot 

plt.show()

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2,numClusters,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()

#%% Question 7

# clustering without PCA
art7 = np.mean(data[:,0:91],axis=0)
energy7 = np.mean(data[:,91:182],axis=0)
data7 = np.column_stack((art7,energy7))
# Compute kMeans for each k:
for ii in range(2, numClusters+2): # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters = int(ii),random_state=17105116).fit(data7) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(data7,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = sum(s) # take the sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,250)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot 

plt.show()

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2,numClusters,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()

#%% Question 8
data8=data[:,205:215] # data
data8=np.nan_to_num(data8)

# Z-score the data:
zscoredData8 = stats.zscore(data8)

# Initialize PCA object and fit to our data:
pca8 = PCA().fit(zscoredData8)

# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals8 = pca8.explained_variance_

# Loadings (eigenvectors): Weights per factor in terms of the original data.
loadings8 = pca8.components_*-1

# Rotated Data - simply the transformed data:
origDataNewCoordinates8 = pca8.fit_transform(zscoredData8)*-1

varExplained8 = eigVals8/sum(eigVals8)*100

# Scree plot:
plt.bar(np.linspace(1,10,10),eigVals8)
plt.plot([0,10],[1,1],color='orange')
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

# Looking at the corrected scree plot, we get 2 meaningful factors, both by 
# Kaiser criterion and Elbow.

#Look at the loadings to figure out meaning:
plt.subplot(1,2,1) # Factor 1: 
plt.bar(np.linspace(1,10,10),loadings8[0,:]) # "Challenges"
plt.title('Challenges')
plt.subplot(1,2,2) # Factor 2:
plt.bar(np.linspace(1,10,10),loadings8[1,:]) # "Support"
plt.title('Support')
plt.show()

plt.plot(origDataNewCoordinates8[:,0],origDataNewCoordinates8[:,1],'o',markersize=1)
plt.xlabel('Challenges')
plt.ylabel('Support')
plt.show() 


kaiserThreshold8 = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals8 > kaiserThreshold8))

threshold = 90 #90% is a commonly used threshold
eigSum8 = np.cumsum(varExplained8) #Cumulative sum of the explained variance 
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum8 < threshold) + 1)

y8 = np.column_stack([data[:, 0:91]])
x_train8, x_test8, y_train8, y_test8 = train_test_split(origDataNewCoordinates8, y8, test_size = 0.2, random_state=17105116)

#Linear Regression
model8 = LinearRegression().fit(x_train8, y_train8) 
y_pred8 = model8.predict(x_test8)
MSE_8 = metrics.mean_squared_error(y_test8, y_pred8)
print('RMSE of multiple regression model is ', math.sqrt(MSE_8))
rsq8 = model8.score(x_test8, y_test8)
print(rsq8)

#%% Question 9
data9=data[:,182:194] # data
data9=np.nan_to_num(data9)

# Z-score the data:
zscoredData9 = stats.zscore(data9)

# Initialize PCA object and fit to our data:
pca9 = PCA().fit(zscoredData9)

# Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals9 = pca9.explained_variance_

# Loadings (eigenvectors): Weights per factor in terms of the original data.
loadings9 = pca9.components_*-1

# Rotated Data - simply the transformed data:
origDataNewCoordinates9 = pca9.fit_transform(zscoredData9)*-1

varExplained9 = eigVals9/sum(eigVals9)*100

# Scree plot:
plt.bar(np.linspace(1,12,12),eigVals9)
plt.plot([0,10],[1,1],color='orange')
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

# Looking at the corrected scree plot, we get 2 meaningful factors, both by 
# Kaiser criterion and Elbow.

#Look at the loadings to figure out meaning:
plt.subplot(1,2,1) # Factor 1: 
plt.bar(np.linspace(1,12,12),loadings9[0,:]) # "Challenges"
plt.title('Challenges')
plt.subplot(1,2,2) # Factor 2:
plt.bar(np.linspace(1,12,12),loadings9[1,:]) # "Support"
plt.title('Support')
plt.show()

plt.plot(origDataNewCoordinates9[:,0],origDataNewCoordinates9[:,1],'o',markersize=1)
plt.xlabel('Challenges')
plt.ylabel('Support')
plt.show() 


kaiserThreshold9 = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals9 > kaiserThreshold9))

eigSum9 = np.cumsum(varExplained9) #Cumulative sum of the explained variance 
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum9 < threshold) + 1)


y9 = np.column_stack([data[:, 0:91]])
x_train9, x_test9, y_train9, y_test9 = train_test_split(origDataNewCoordinates9, y9, test_size = 0.2, random_state=17105116)

#Linear Regression
model9 = LinearRegression().fit(x_train9, y_train9) 
y_pred9 = model9.predict(x_test9)
MSE_9 = metrics.mean_squared_error(y_test9, y_pred9)
print('RMSE of multiple regression model is ', math.sqrt(MSE_9))
rsq9 = model9.score(x_test9, y_test9)
print(rsq9)

#%% Question 10

sophistication = data[:,219]
orientation = data[:,217]
data10 = np.column_stack((sophistication, orientation))

#Inspect the data
print(data10[:10])
print('Total number of users:',len(data10))

# Plot data:
plt.scatter(data10[:,0],data10[:,1],color='black')
plt.xlabel('Sophistication')
plt.ylabel('Left or Right?')
plt.yticks(np.array([0,1,2,3,4,5,6]))
plt.show()


#%% Extra credit
plt.plot(y66,y6,'o',markersize=5) 
plt.xlabel('Prediction from model') 
plt.ylabel('Actual rating')  
plt.title('R^2 = {:.3f}'.format(model6.score(x6,y6)))


plt.plot(x6,y6,'o',markersize=5) 
plt.xlabel('Prediction from model') 
plt.ylabel('Actual rating')  
plt.title('R^2 = {:.3f}'.format(model6.score(x6,y6)))
