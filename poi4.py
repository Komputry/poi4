import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv('combined_file.csv')

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)

onehot_encoder = OneHotEncoder(sparse_output=False)
y_int = y_int.reshape(len(y_int), 1)
y_onehot = onehot_encoder.fit_transform(y_int)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3)

model = Sequential()
model.add(Dense(10, activation='sigmoid', input_dim=8))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)

y_pred = model.predict(X_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)
