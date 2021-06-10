# Nama: Dea Claresta
# Nim: 2301863736
# Kelas: LB08


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

data= pd.read_csv('titanic.txt')
# print(data.describe())
# No.1a
Survived=data['Survived']
PassengerId=data['PassengerId']
Pclass=data['Pclass']
Age=data['Age']
SibSp=data['SibSp']
Parch=data['Parch']
del data['Name']
del data['Sex']
del data['Ticket']
del data['Fare']
del data['Cabin']
del data['Embarked']


# No.1b
# untuk periksa missing value
print("Sebelum diisi :")
print(data.isnull().sum())   
# untuk mengisi missing value dengan nilai rata-rata   
data=data.fillna(data.mean())
# untuk mengecek semua missing value sudah terisi atau belum
print("Setelah diisi :")
print(data.isnull().sum())

# No.1c
plt.scatter(PassengerId,Survived,color='Red')
plt.show()
plt.scatter(Pclass,Survived,color='Blue')
plt.show()
plt.scatter(Age,Survived,color='Orange')
plt.show()
plt.scatter(SibSp,Survived,color='Green')
plt.show()
plt.scatter(Parch,Survived,color='Black')
plt.show()

# No.1d
print("Correlation dari independent variable dengan dependent variable:")
correlation=data.corr()
print(correlation)


# No.2
# untuk memisahkan kolom independent(x) dengan kolom dependent(y)
y=data.Survived
x=data.drop('Survived',axis=1)
# untuk membagi data train set(70%) dengan test set(30%)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
# untuk print training set
print(x_train)
print(y_train)

# No.3a
# untuk melakukan pembelajaran
Belajar=LinearRegression().fit(x_train,y_train)
# untuk menyimpan hasil prediksi
hasilPrediksi=Belajar.predict(x_test)

# No.3b
# untuk print hasil prediksi
print(hasilPrediksi)
# untuk print hasil aktual atau sebenarnya
print(y_test)

# No.4a
print('Evaluasi Coefficient:')
print(Belajar.coef_)

# No.4b
print('Evaluasi Metrics:')
print("MAE: " , metrics.mean_absolute_error(y_test,hasilPrediksi))
print("MSE: " , metrics.mean_squared_error(y_test,hasilPrediksi))
print("RMSE: " , np.sqrt(metrics.mean_squared_error(y_test,hasilPrediksi)))

# No.5
# untuk menghasilkan Plot Predicted Value vs Measure
plt.scatter(y_test,hasilPrediksi,color='Pink')
plt.show()
