# -*- coding:utf-8 -*-
# Se importan las librerias#
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


##### Se cargan los datos ####
dataframe = pd.DataFrame({'Alumno':['ALBAMONTE MONICA ANDREA','ALEGRE MARIA ELENA','ALTAMIRANO MARGARITA GRACIELA','ALVAREZ MIGUEL ANGEL','BENITEZ MIGUEL ANTONIO','BENITEZ YESICA ROMINA','BILLORDO EVA JULIA','BRITEZ NATALIA MABEL','BRITEZ SONIA ALICIA','CANDIYU LUIS HECTOR EDUARDO','COLLI ESTHELA SANDRA','DELGADO GUADALUPE ROSA','FERNANDEZ MIRTA ELISABETH','GARCIA CARLOS ALBERTO','GAUNA MARIA BEATRIZ','GUTIERREZ MARIANA MABEL','HERRERA ARIELA BEATRIZ','LEZCANO ANA CRISTINA','LOPEZ DIEGO HORACIO','LUNA GUILLERMO ARIEL','MAFFEI GABRIELA ROSANNA','MARTINEZ MIRIAN DIONISIA','MENDOZA FELIX HORACIO','NEGRI CARLOS ADRIAN','PALACIOS MIRIAN EMILCE','PORCEL DE PERALT5A SANDRA','PROS SERGIO AMILCAR','QUEBEDO JAVIER LUIS','QUEBEDO LUISA MAGDALENA','RAMIREZ LAURA ANDREA','RODRIGUEZ ALEGRE ANALIA','SANCHEZ PEDRO ANTONIO','SEGOVIA JUANA BEATRIZ','SOTO ARIEL LUIS','VARELA ANTONIA','VARELA RAMONA JULIETA','VERA RUIZ DIAZ ESCOLASTICA','ACOSTA GLADIS ELIZABETH NOEMI','ACUÑA LAURA','ALCARAZ GRISELDA LILIANA','ALFONSO ADRIANA ISABEL','ALMIRON  MONICA ELIZABETH','ARREGIN HUBERTO RAMON','BORDA SIXTA','CABRAL SILVIA MERCEDES','CARDOZO DELIA DAMIANA','CENA DIONISIA CONCEPCION','CODUTTI NOELIA BEATRIZ','CORONEL SILVIA MARINA','ESCOBAR ANDREA SILVANA','FRANCO IRENE','GALARZA DE LA ROSA MIRTHA','GIMENEZ PEDEMONTE DINA','GOMEZ ANGEL','LATOR TERESA','LOPEZ BIBIANA CEFERINA','LOPEZ CAROLINA INÉS','MACHUCA TEODORA PILAR','MONTIEL MARTIN RODRIGO','MORALES MARCELO LEONCIO','OJEDA LORENA','OJEDA NORMA','PORTILLO JOSE DOLORES','RIOS ANTONIA','RIOS NANCY MALVINA','RIVERO NAVARRO ELIDA','RODRIGUEZ NILDA ELIZABETH','ROMERO SERGIO','SARZA MIRIAN ESTELA','TROLSON JULIAN','SENA MIRTA','SILVA ESTEBAN MILAGROS','VALENZUELA LUCIA','VILLAVA JOSE EMMANUEL','ZACARIAS GRISELDA EMILCE','ZACARIAS HUGO ANTONIO','ALEGRE BEATRIZ IRIS','BARRIOS CLAUDIA MARILINA','FERNANDEZ SUSANA BEATRIZ','FRANCO MONICA','FRIAS NANCY BEATRIZ','GALARZA OLGA','GALEANO KARINA','GAUNA ELIANA GRACIELA','GIMENEZ MARCELINA','GOMEZ EMILIANA','JARA YOLANDA MARISOL','MORALES MARCELO','PEREYRA CAROLINA GRISELDA','PINTOS MERCEDES','RAMIREZ JOSE LUIS','ROJAS ARGENTINA ISABEL'],'Sexo':['0','0','0','1','1','0','0','0','0','1','0','0','0','1','0','0','0','0','1','1','0','0','1','1','0','0','1','1','0','0','0','1','0','1','0','0','0','0','0','0','0','0','1','0','0','0','0','0','0','0','0','0','0','1','0','0','0','0','1','1','0','0','1','0','0','0','0','1','0','1','0','1','0','1','0','1','0','0','0','0','0','0','0','0','0','0','0','1','0','0','1','0'],'TrabajoPractico1': ['10','6.5','10','0','9','9','10','10','5','9','9','10','9','10','8.5','0','10','8','7','0','10','0','4','10','0','10','8','10','9','0','8','9','10','0','0','0','7','10','10','9','10','10','0','10','10','0','0','10','10','10','0','10','8.5','0','10','10','0','10','10','9','10','4','0','0','0','0','10','8','0','0','8','0','10','10','9','0','8','7.5','7','0','10','6.5','0','7','10','9.5','0','0','9','8.5','9.5','8'], 'TrabajoPractico2': ['8.5','10','9','0','8','9.5','6.5','9','9','10','10','9.5','9','9.5','4','0','9','9','7.5','0','10','0','5','8.5','0','8.5','9.5','9.5','4','0','9','6.5','10','0','0','0','9','10','10','10','10','9','4','9','6','0','0','9.5','9.5','9.5','8','9.5','9.5','0','8.5','10','0','10','10','4','10','4','0','0','0','0','9.5','6','0','0','10','0','8','9.5','8','0','9.5','9.5','9','0','10','4.5','7','7','6','9','0','0','9.5','6.5','10','9'], 'TrabajoPractico3': ['9','10','10','0','10','10','10','10','10','10','9','10','10','10','10','0','10','10','10','0','10','0','4','10','0','9','0','10','10','0','10','10','10','0','0','0','10','10','9','10','9','9','0','10','4','0','0','10','10','10','10','10','9','0','10','9','0','10','10','0','9','4','0','0','0','0','10','10','0','0','10','0','10','10','8','0','9','8','8','0','10','9','0','10','10','10','0','0','10','10','9','9'],'TrabajoPractico4': ['10','10','10','0','10','10','10','10','10','10','10','10','10','10','10','0','10','10','10','0','10','0','10','10','0','10','10','10','10','0','10','10','10','0','0','0','10','10','10','10','10','10','10','10','10','0','0','10','10','10','10','10','10','0','10','10','0','10','10','10','10','10','0','0','0','0','10','10','0','0','10','0','10','10','10','0','5','0','10','0','5','7','0','0','10','0','0','0','10','10','5','4'],'TrabajoPractico5': ['3','10','10','0','10','7','7','9.5','3','10','4','10','10','10','9','0','10','4','10','0','10','0','4','10','0','8','9.5','10','9.5','0','9','4','9','0','0','0','9','10','10','10','10','10','0','5','4','0','0','9','10','10','10','10','10','0','4','10','0','4','7','0','10','4','0','0','0','0','10','0','0','0','10','0','3','10','0','0','9','8','8','0','10','7','0','10','10','8','0','0','10','10','10','9'],'TrabajoPractico6': ['4','10','10','0','10','10','10','10','10','10','10','10','10','10','10','0','10','10','10','0','10','0','10','10','0','10','10','10','10','0','10','10','10','0','0','0','10','10','10','9','10','10','0','10','10','0','0','10','10','10','0','10','9','0','10','4','0','10','4','0','10','4','0','0','0','0','10','0','0','0','10','0','10','10','0','0','10','10','10','0','10','10','0','0','10','10','0','0','0','10','10','10'],'TrabajoPractico7': ['9','10','10','0','10','10','10','10','10','10','9','10','10','10','10','0','10','0','10','0','10','0','10','10','0','10','10','10','10','0','9','4','10','0','0','0','10','10','10','10','10','10','0','10','4','0','0','10','4','10','4','10','10','0','10','10','0','10','10','0','10','4','0','0','0','0','10','0','0','0','10','0','10','10','4','0','10','8','10','0','8','8','0','0','10','9','0','0','0','10','10','10'],'TrabajoPractico8': ['10','10','10','0','10','10','10','10','10','10','10','4','10','10','10','0','10','10','4','0','10','0','10','10','0','9','10','10','10','0','10','10','10','0','0','0','9','10','10','10','10','10','0','9','4','0','0','10','9','10','4','10','4','0','10','9','0','10','10','0','10','4','0','0','0','0','10','0','0','0','10','0','10','10','0','0','10','5','10','0','10','10','0','0','10','10','0','0','0','10','0','10'],'TrabajoPractico9': ['9','9','9','0','10','9','9','10','10','10','4','9','10','9','8','0','10','10','8','0','9','0','10','10','0','8','10','9','10','0','9','10','8','0','0','0','10','10','10','10','10','10','0','9','4','0','0','10','9','10','4','10','4','0','10','9','0','10','10','0','10','4','0','0','0','0','10','0','0','0','10','0','10','10','0','0','10','10','10','0','9','9','0','0','10','10','0','0','0','10','0','6'],'TrabajoPractico10': ['9','9','10','0','10','10','10','9','10','9.5','10','10','10','10','10','0','10','0','4','0','10','0','5.5','10','0','10','10','9','10','0','10','10','10','0','0','0','10','8','9','10','10','10','4','10','2','0','0','4','8','9','10','0','4','3.5','0','10','10','0','10','4','10','4','0','0','0','0','10','0','0','0','10','0','10','9','7.5','0','10','6.5','8','0','10','2.5','0','0','2.5','8','0','0','0','9.5','7','4.5'], 'NotaPrimerParcial':  ['7','10','10','0','10','10','10','10','9','10','10','10','10','10','10','0','10','10','9','0','9','0','7','9','0','10','7','10','10','0','10','10','10','0','0','0','10','10','10','10','10','10','0','10','9','0','0','10','9','10','6.5','10','10','0','10','10','0','10','9','7','10','4','0','0','0','0','9','0','0','0','10','0','10','10','10','0','6','9.5','9','0','10','8','0','8.5','10','6','0','0','10','10','9','9'], 'NotaSegundoParcial':  ['7','7.5','8','9','9','10','9','10','10','10','9','9.5','9.5','9','7.5','0','9','0','10','0','8','0','7.5','9','0','8','10','10','10','0','10','9','10','0','0','0','9','8','9','10','10','9','0','8','9','0','0','9','0','10','0','10','9','0','10','8','0','9','10','0','10','10','0','0','0','0','10','0','0','0','10','0','10','9','0','0','10','7','8.5','0','10','10','0','0','8','9.5','0','0','0','9','7','8'],'NotaTrabajoFinal':  ['7','10','10','0','8','10','7','10','10','10','7','10','10','10','9','0','10','0','10','0','10','0','7','10','0','10','10','10','10','0','10','9','7','0','0','0','10','10','10','10','10','7','0','10','0','0','0','10','0','10','3','10','10','0','7','10','0','10','7','0','10','10','0','0','0','0','10','0','0','0','10','0','6','7','6','0','7','7','10','0','10','7','0','0','7','9','0','0','0','10','4','7'], 'Resultado':  ['1','1','1','0','1','1','1','1','1','1','1','1','1','1','1','0','1','0','1','0','1','0','1','1','0','1','1','1','1','0','1','1','1','0','0','0','1','1','1','1','1','1','0','1','0','0','0','1','0','1','0','1','1','0','1','1','0','1','1','0','1','1','0','0','0','0','1','0','0','0','1','0','1','1','0','0','1','1','1','0','1','1','0','0','1','1','0','0','0','1','1','1']})

##### Se muestra la informacion de la matriz ####
print ('La dimension de la matriz de datos es: ', dataframe.shape)
print ('Info de la matriz de datos')
print (dataframe.info)

##### Se grafica los resultados estudiantiles ####
valores_x = dataframe['Resultado'].unique()
valores_y = dataframe['Resultado'].value_counts().tolist()
plt.title("Cantidad de alumnos aprobados y desaprobados")
plt.bar(valores_x, valores_y, color=['green','red'])
plt.xlabel("Aprobados(1) - Desaprobados(0)")
plt.ylabel("Cantidad")
plt.show()
plt.close('all')

##### Se describe la matriz de datos sin los nombres de alumnos ni resultados  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    # Se definen modelos para penalizaciones l1, l2, elasticnet
    modelo_n = linear_model.LogisticRegression(C=c, penalty='none')
    modelo_l2 = linear_model.LogisticRegression(C=c, penalty='l2')
    modelo_n.fit(X,y)
    modelo_l2.fit(X,y)
    predicciones_n = modelo_n.predict(X)
    predicciones_l2 = modelo_l2.predict(X)
    coeficientes_n = modelo_n.coef_
    coeficientes_l2 = modelo_l2.coef_
    termino_independiente_n = modelo_n.intercept_
    termino_independiente_l2 = modelo_l2.intercept_
    # Se visualizan los resultados del entrenamiento
    print('Predicciones para No penalizado: \n', predicciones_n)
    print('Predicciones para L2: \n', predicciones_l2)
    # Se visualizan los coeficientes obtenidos.
    print('Coeficientes para No penalizado: \n', coeficientes_n)
    print('Coeficientes para L2: \n', coeficientes_l2)
    # Se visualizan los valores donde cortan al eje Y (en X=0)
    print('Termino independiente para No penalizado: \n', termino_independiente_n)
    print('Termino independiente para L2: \n', termino_independiente_l2)
    ### Se valida el modelo ####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones_n = modelo_n.predict(X_validation)
    predicciones_l2 = modelo_l2.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio para No penaizado: %.2f" % mean_squared_error(Y_validation, predicciones_n))
    print("El error cuadratico medio para L2: %.2f" % mean_squared_error(Y_validation, predicciones_l2))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza para No penalizado: %.2f' % r2_score(Y_validation, predicciones_n))
    print('Puntaje de varianza para L2: %.2f' % r2_score(Y_validation, predicciones_l2))
    # Reporte de resultados #
    print('La matriz de confusion para No penalizado es: \n',confusion_matrix(Y_validation, predicciones_n))
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('La matriz de confusion para L2 es: \n',confusion_matrix(Y_validation, predicciones_l2))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8'],'TrabajoPractico5': ['9'],'TrabajoPractico6': ['10'],'TrabajoPractico7': ['10'],'TrabajoPractico8': ['10'],'TrabajoPractico9': ['0'],'TrabajoPractico10': ['0'], 'NotaPrimerParcial':  ['10'], 'NotaSegundoParcial':  ['8.5'],'NotaTrabajoFinal':  ['7']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
