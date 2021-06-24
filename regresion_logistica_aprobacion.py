# -*- coding:utf-8 -*-
# Se importan las librerias#
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

##### Se cargan los datos ####
dataframe = pd.DataFrame({'Alumno':['Gladis', 'Laura','Griselda','Adriana','Monica', 'Humberto'], 'NotaPrimerParcial': ['10','10','10','10', '10','0'], 'NotaSegundoParcial': ['8','9','10','10','9','0'],'NotaTrabajoFinal': ['8','9','10','10','9','0'],'Resultado': ['1','1','1','1','1','0']})
print (dataframe.head(5))
##### Se muestra la informacion de la matriz ####
print ('La dimension de la matriz de datos es: ', dataframe.shape)
print ('Info de la matriz de datos')
print (dataframe.info)
##### Se grafica los resultados ####
valores_x = dataframe['Resultado'].unique()
valores_y = dataframe['Resultado'].value_counts().tolist()
plt.bar(valores_x, valores_y)
plt.show()
plt.close('all')
##### Se describe la matriz de datos sin los nombres de alumnos ####
dataframe1=dataframe.drop("Alumno",axis=1)
print (dataframe1.describe())
#### Creamos el modelo ####
X = np.array(dataframe.drop(['Alumno'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
modelo = linear_model.LogisticRegression()
modelo.fit(X,y)
predicciones = modelo.predict(X)
print(predicciones)
print(modelo.score(X,y))
### Validacion del modelo ####
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
predicciones = modelo.predict(X_validation)
print('Precision del algoritmo: ',accuracy_score(Y_validation, predicciones))
#### Reporte de resultados ####
print(confusion_matrix(Y_validation, predicciones))
print(classification_report(Y_validation, predicciones))
#### Clasificacion de nuevos registros  ####
X_new = pd.DataFrame({'NotaPrimerParcial': ['10'], 'NotaSegundoParcial': ['8'],'NotaTrabajoFinal': ['10'],'Resultado': ['1']})
modelo.predict(X_new)
