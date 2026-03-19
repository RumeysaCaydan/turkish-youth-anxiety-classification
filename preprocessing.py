import pandas as pnd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

data=pnd.read_excel("dataset.xlsx", sheet_name ="Sheet1")

#print(data.head())
print(data.columns)


le_thema = LabelEncoder()
data["thema1"] = le_thema.fit_transform(data["thema"])


le_source = LabelEncoder()
data["source1"] = le_source.fit_transform(data["source"])
print(data.head())

X=data["contents"] #independent value
y=data["thema1"] #dependent value(numerical)

X_train,X_test, y_train, y_test= train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state=42,
    stratify= y
)
