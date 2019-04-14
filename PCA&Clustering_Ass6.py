import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import matplotlib.pyplot as plt
#Importing dataset
mydataset= pd.read_csv('house_prices_data_training_data.csv')
mydataset.dropna(inplace=True)

#Extracting x from the dataset
x=mydataset.iloc[:,3:21]
price=mydataset.iloc[:,2]
price=price.values

#Number of examples
m=x.shape[0]

#Removing the average
x=x-x.mean()
price=price-price.mean()

#Normalization
x=x/x.std()
price=price/price.std()

#Correlation of x
x_corr=x.corr()

#Covariance of x
x_cov=x.cov()

#Extracting Unit vectors U and V and Diagonal vector S 
U, sigma, V = np.linalg.svd(x_cov)

#Extracting Eigen values
eigen_values= np.square(sigma)

#Choosing K
k=0
alpha=1
while alpha>0.001:
    k=k+1
    sum_k=eigen_values[0:k].sum()
    sum_m=eigen_values.sum()
    alpha=1-(sum_k/sum_m)

number_of_dimensions=k-1

##Reduced Dataset
U_reduced=np.transpose(U[:,0:k])
x_values=x.values
x4reduced_data=x_values.T
z=np.dot(U_reduced,x4reduced_data) #Obtaining the new transformed data

##Obtaining the Original dimensions
x_approx=np.dot(np.transpose(U_reduced),z)
x_approx=np.transpose(x_approx)

##Error estimation
error=np.sum(x_approx-x_values)/m

#House prices estimation using Linear regression

model=LinearRegression()
model.fit(x_approx,price)
estimation=model.predict(x_approx)

mse=np.mean((price-estimation)**2)


###################################

#Clustering
def clustering(data,k_ranger):
    m=data.shape[0]
    columns=data.shape[1]
    ks=np.zeros((k_ranger-1,1))
    cost_k=np.zeros((k_ranger-1,1))
    
    for k in range(1,k_ranger):
        Means_initial=np.zeros((k,columns)) #Means for clusters
        Means_new=np.zeros((k,columns))
        c=np.zeros((m,1)) #Cluster decision
        diff=np.zeros((k,1)) #difference between X and Mean
        cost=0
        cost_prev=0
        Means_Examples=np.zeros((m,columns))
        iteration=0
        ### K-mean Clustering Algorithm with 100 trials so not to get stuck at local minima
        #for trials in range(100):
        
        #Initiating the Mean

        for row in range(0,k):
            for col in range(0,columns):
                Means_index=random.randint(0,m-1)
                Means_initial[row,col]=data[Means_index,col]
        while True:
            iteration=iteration+1
            print(iteration)
            #Deciding Clusters for all examples 
            for i in range(m):
                for j in range(k):
                    diff[j,:]=np.sum((data[i,:]-Means_initial[j,:])**2)
                c[i,:]=np.argmin(diff)
            Means_new=np.zeros((k,columns))  
            #Getting the new Mean   
            for cluster in range(k):
                count=0
                for example in range(m):
                    c_index=c[example,:]
                    if c_index==cluster:
                        Means_new[cluster,:]=Means_new[cluster,:]+data[example,:]
                        #print(Means_new[cluster,:])
                        count=count+1
                #print(count)
                Means_new[cluster,:]=Means_new[cluster,:]/(count)

            Means_initial=Means_new
            #Cost Function
            for u in range(m):
                for g in range(k):
                    c_index=c[u,:]
                    if c_index==g:
                        Means_Examples[u,:]=Means_new[g,:]
            cost=np.sum((data-Means_Examples)**2)/m
            print(cost)
            if iteration==1:
                cost_prev=cost
            else:
                cost_difference=np.absolute(cost_prev-cost)
                if cost_difference<0.0001:
                    break
                else:
                    cost_prev=cost
        cost_k[k-1,0]=cost
        ks[k-1,0]=k

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(ks, cost_k)
    return Means_new    


#Calling the K-mean Clustering method
Means_Original_Dataset=clustering(x_values,11)
Means_approximated_Dataset=clustering(x_approx,11)    