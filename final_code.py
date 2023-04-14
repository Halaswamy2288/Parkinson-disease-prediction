#Importing the Libraries
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm  #SVM:Support Vector Machine
from sklearn.metrics import accuracy_score

#Loading the dataset
parkinson_data=pd.read_csv('Parkinsson disease.csv')
print(parkinson_data)

#Data pre-processing
x=parkinson_data.drop(columns=['name','status'],axis=1)
y=parkinson_data['status']              
print(x)
print(y)

#Spliting dataset to training data and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
print(x.shape,x_train.shape,x_test.shape)   

#xgboost
from xgboost import XGBClassifier   
model = XGBClassifier()
model.fit(x_train, y_train,)
print(model)

#Data Standardization
scaler=StandardScaler()         
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
print(x_train)

#Support Vector Machine Model(SVM)
model=svm.SVC(kernel='linear')   

#Training the svm model with training data
model.fit(x_train,y_train)

#Accuracy Score
#Accuracy Score of training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(y_train,x_train_prediction)
print("Accuracy score of training data: ",training_data_accuracy)

#Accuracy Score of testing data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(y_test,x_test_prediction)
print("Accuracy score of test data: ",test_data_accuracy)

#Predicting
input_data=(116.879,131.897,108.153,0.00788,0.00007,0.00334,00.00493,0.01003,0.02645,0.265,0.01394,0.01625,0.02137,0.04183,0.00786,22.603,0.540049,0.813432,-4.476755,0.262633,1.827012,0.326197)
#changing input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)
#reshape the numpy array
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
#standarize the data
std_data= scaler.transform(input_data_reshaped)

prediction= model.predict(std_data)
print(prediction)

if(prediction[0]==0):
    print("Person is not having Parkinson Disease")
else:
    print("Person is having Parkinson Disease")

   




