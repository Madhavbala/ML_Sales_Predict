#import libraries
import pandas as pd    
import numpy as np 

#load the dataset 
file = pd.read_csv("Salary_Data.csv")
#to find how many null values on the data sets
print(" Num of null values : ",file.isnull().sum())
print(file.head())

#to find the rows and columns
print("File Shape : ",file.shape)

#split the dataset's one x is input y is output
x = file[["Age","Salary"]]
y = file["status"]

#split 2 test cases and 2 train cases
from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train = train_test_split(x,y,test_size = 0.25 , random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)      # fit_transform only use train data to find  mean , variance 
x_test = sc.transform(x_test)            # transform it take full data to find mean , variance 
print(x_test)

#import logistic regression algorithm library and applied into train cases
from sklearn.linear_model import LogisticRegression    
algorithm = LogisticRegression()
algorithm.fit(x_train,y_train)

y_pred = algorithm.predict(x_test)
print(y_pred)
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix ,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix : " , cm)
print("Accurancy of the matrix {0} % ".format(accuracy_score(y_test,y_pred*100)))


age = input("Enter the Customer Age        : ")
salary = input("Entr the coustomer salary  : ")
newcustomer = [[age,salary]]
result = algorithm.predict(sc.transform(newcustomer))
if result == 0:
    print("Customer Don't Buy This")
else:
    print("Custome Definently Buy ")

