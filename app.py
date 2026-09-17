import pandas as pd

df = pd.read_csv("/content/Iris.csv")

df

df.head()

df.tail()

df.columns

df.info()

df['Species'].value_counts()

df.duplicated().sum()

df.describe()

x = df.drop(columns=['Id','Species'])

x

y = df['Species']

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.8,random_state=56)

x_train

x_test

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

model = knn.fit(x_train,y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(y_test,y_pred)

cr = classification_report(y_test,y_pred)

import joblib

!pip install streamlit
import streamlit as st

#deployment
import streamlit as st

#load model
model = joblib.dump(model,"iris_model.pkl")

#page title
st.title('machine learning on iris dataset')

#input labels
sepal_lemgth = st.number_input('sepal_length')
sepal_width = st.number_input('sepal_width')
sepal_lemgth = st.number_input('petal _length')
sepal_width = st.number_input('petal_width')

#prediction
if st.button('predict'):
  input = np.array([[sepal_length,
                     sepal_width,
                     petal_length,
                     petal_width]]).astype(np.float64)
  prediction = model.predict(input)
  st.success(prediction)
