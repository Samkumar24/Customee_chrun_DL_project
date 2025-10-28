import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tenflow.keras.models import load_model

df = pd.read_csv("C:\\Users\\sam\\Downloads\\bank_churn_data.csv")

#print(df.columns)

X = df.drop(['CustomerId','Surname','Exited'], axis=1)
y = df['Exited']

print(X.columns,  y)


