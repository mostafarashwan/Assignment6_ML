import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm

#Importing dataset
mydataset= pd.read_csv('house_prices_data_training_data.csv')
mydataset.dropna(inplace=True)
x=mydataset.iloc[:,3:21]
x=x.values
#Extracting Training Examples from the dataset
x_training=x[0:10798,:]
#Exctracting Cross validation Examples from dataset
x_cv=x[10799:14398,:]
#Extracting Testing Examples from the dataset
x_testing=x[14399:17998,:]
#Getting Means and STDs of training data
Mu=x_training.mean(axis=0)
Sigma=x_training.std(axis=0)
#Applying Anomaly Algorithm on Cross validation data
probability_cv=np.zeros((x_cv.shape[0],x_cv.shape[1]))
probability_product_cv=np.zeros((x_cv.shape[0],1))
Decision_cv=np.zeros((x_cv.shape[0],1))
AnomalyCount_cv=0
for example in range(x_cv.shape[0]):
    p_cv=1
    for feature in range(x_cv.shape[1]):
        probability_cv[example,feature]=norm.cdf(x_cv[example,feature],loc = Mu[feature],scale = Sigma[feature])
        p_cv=p_cv*norm.cdf(x_cv[example,feature],loc = Mu[feature],scale = Sigma[feature])
    probability_product_cv[example,0]=p_cv
#Calculating quartiles for decision threshold
first_quartile,second_quartile,third_quartile=np.percentile(probability_product_cv, [25, 50, 75])
interquartile_range=third_quartile-first_quartile
firstquartile1=probability_product_cv[probability_product_cv<=first_quartile]
firstquartile=np.sort(firstquartile1)
thirdquartile1=probability_product_cv[probability_product_cv>=third_quartile]
thirdquartile=np.sort(thirdquartile1)

anomaly_decision1_cv=np.ones((firstquartile1.shape[0],1))
anomaly_decision2_cv=np.ones((thirdquartile1.shape[0],1))
a1=int(0.08*firstquartile1.shape[0])
a2=int(0.92*thirdquartile1.shape[0])
 
anomaly_percent1=firstquartile[a1]
anomaly_percent2=thirdquartile[a2]
AnomalyCount_cv=0


decision_cv=np.zeros((probability_product_cv.shape[0],1))
for v in range((probability_product_cv.shape[0])):
    if probability_product_cv[v,0]<anomaly_percent1 or probability_product_cv[v,0]>anomaly_percent2:
        AnomalyCount_cv=AnomalyCount_cv+1
        decision_cv[v,0]=1
     
###################################################################################################################
#Applying Anomaly Algorithm on Testing data
probability=np.zeros((x_testing.shape[0],x_testing.shape[1]))
probability_product=np.zeros((x_testing.shape[0],1))
Decision=np.zeros((x_testing.shape[0],1))
AnomalyCount=0

for example in range(x_testing.shape[0]):
    p=1
    for feature in range(x_testing.shape[1]):
        probability[example,feature]=norm.cdf(x_testing[example,feature],loc = Mu[feature],scale = Sigma[feature])
        p=p*norm.cdf(x_testing[example,feature],loc = Mu[feature],scale = Sigma[feature])
    probability_product[example,0]=p

AnomalyCount=0

decision=np.zeros((probability_product.shape[0],1))
for v in range((probability_product.shape[0])):
    if probability_product[v,0]<anomaly_percent1 or probability_product[v,0]>anomaly_percent2:
        AnomalyCount=AnomalyCount+1
        decision[v,0]=1






















#Decide epsilon    
#parameters = norm.fit(x)
#xx=np.linspace(-3,9,200)
#fitted_pdf = norm.pdf(x_testing[0,0],loc = Mu[0],scale = Sigma[0])        