# -*- coding:utf-8 -*-
# Se importan las librerias#
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
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
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8'],'TrabajoPractico5': ['9'],'TrabajoPractico6': ['10'],'TrabajoPractico7': ['10'],'TrabajoPractico8': ['10'],'TrabajoPractico9': ['0'],'TrabajoPractico10': ['0'], 'NotaPrimerParcial':  ['10'], 'NotaSegundoParcial':  ['8.5'],'NotaTrabajoFinal':  ['7']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')
# Estructura del arbol
# 
# El clasificador de decision tiene un atributo llamado tree_ que permite el acceso a atributos
# de bajo nivel, como node_countel numero total de nodos y max_depth la profundidad máxima del arbol. 
# También almacena toda la estructura del arbol binario, representada como una serie de matrices 
# paralelas. El elemento i-ésimo de cada matriz contiene información sobre el nodo i. 
# El nodo 0 es la raíz del árbol. Algunas de las matrices solo se aplican a hojas o nodos divididos. 
# En este caso, los valores de los nodos del otro tipo son arbitrarios. Por ejemplo, las matrices 
# feature y threshold solo se aplican a los nodos divididos. Por tanto, los valores de los nodos 
# hoja en estas matrices son arbitrarios.
#
# Entre estos array, tenemos:
#
#        children_left[i]: id del hijo izquierdo del nodo io -1 si el nodo hoja
#        children_right[i]: id del hijo derecho del nodo io -1 si el nodo hoja
#        feature[i]: característica utilizada para dividir el nodo i
#        threshold[i]: valor umbral en el nodo i
#        n_node_samples[i]: el número de muestras de entrenamiento que llegan al nodo i
#        impurity[i]: la impureza en el nodo i
#
# Usando las matrices, podemos atravesar la estructura del árbol para calcular varias propiedades. 
# A continuacion, calcularemos la profundidad de cada nodo y si es una hoja o no.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable NotaTrabajoFinal para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8'],'TrabajoPractico5': ['9'],'TrabajoPractico6': ['10'],'TrabajoPractico7': ['10'],'TrabajoPractico8': ['10'],'TrabajoPractico9': ['0'],'TrabajoPractico10': ['0'], 'NotaPrimerParcial':  ['10'], 'NotaSegundoParcial':  ['8.5']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()

print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable NotaSegundoParcial para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8'],'TrabajoPractico5': ['9'],'TrabajoPractico6': ['10'],'TrabajoPractico7': ['10'],'TrabajoPractico8': ['10'],'TrabajoPractico9': ['0'],'TrabajoPractico10': ['0'], 'NotaPrimerParcial':  ['10']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()

print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable NotaPrimerParcial para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8'],'TrabajoPractico5': ['9'],'TrabajoPractico6': ['10'],'TrabajoPractico7': ['10'],'TrabajoPractico8': ['10'],'TrabajoPractico9': ['0'],'TrabajoPractico10': ['0']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()


print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable TrabajoPractico10 para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de TP10, 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8'],'TrabajoPractico5': ['9'],'TrabajoPractico6': ['10'],'TrabajoPractico7': ['10'],'TrabajoPractico8': ['10'],'TrabajoPractico9': ['0']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()

print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable TrabajoPractico9 para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de TP9, TP10, 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8'],'TrabajoPractico5': ['9'],'TrabajoPractico6': ['10'],'TrabajoPractico7': ['10'],'TrabajoPractico8': ['10']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()

print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable TrabajoPractico8 para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de TP8, TP9, TP10, 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8'],'TrabajoPractico5': ['9'],'TrabajoPractico6': ['10'],'TrabajoPractico7': ['10']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()

print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable TrabajoPractico7 para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de TP7, TP8, TP9, TP10, 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8'],'TrabajoPractico5': ['9'],'TrabajoPractico6': ['10']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()

print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable TrabajoPractico6 para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de TP6, TP7, TP8, TP9, TP10, 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8'],'TrabajoPractico5': ['9']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()


print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable TrabajoPractico5 para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de TP5, TP6, TP7, TP8, TP9, TP10, 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico5', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico5', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8'],'TrabajoPractico4': ['8']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()


print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable TrabajoPractico4 para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de TP4, TP5, TP6, TP7, TP8, TP9, TP10, 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico4', 'TrabajoPractico5', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico4', 'TrabajoPractico5', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9'],'TrabajoPractico3': ['8']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()

print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable TrabajoPractico3 para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico3', 'TrabajoPractico4', 'TrabajoPractico5', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico3', 'TrabajoPractico4', 'TrabajoPractico5', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5'],'TrabajoPractico2': ['9']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()

print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable TrabajoPractico2 para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico2', 'TrabajoPractico3', 'TrabajoPractico4', 'TrabajoPractico5', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico2', 'TrabajoPractico3', 'TrabajoPractico4', 'TrabajoPractico5', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1'],'TrabajoPractico1': ['9.5']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()

print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('+ Eliminación de la variable TrabajoPractico1 para la prediccion de etiquetas ')
print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print ('')
##### Se describe la matriz de datos sin las variables alumno, nota de TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9, TP10, 1er, 2do parcial, trabajo final y resultado  ####
print ('Descripcion de la matriz X')
dataframe1=dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico1', 'TrabajoPractico2', 'TrabajoPractico3', 'TrabajoPractico4', 'TrabajoPractico5', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],axis=1)
print (dataframe1.describe())

#### Se crea y entrenan los modelos de regresion logistica ####
X = np.array(dataframe.drop(['Alumno', 'NotaTrabajoFinal', 'TrabajoPractico1', 'TrabajoPractico2', 'TrabajoPractico3', 'TrabajoPractico4', 'TrabajoPractico5', 'TrabajoPractico6', 'TrabajoPractico7', 'TrabajoPractico8', 'TrabajoPractico9', 'TrabajoPractico10', 'NotaPrimerParcial', 'NotaSegundoParcial', 'Resultado'],1))
y = np.array(dataframe['Resultado'])
print ('La dimension de la matriz X es: ', X.shape)
print('------------------------------------------------------------')
print('Algoritmo de Regresion Logistica')
print('------------------------------------------------------------')
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
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
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
    print('El reporte de clasificacion para No penalizado es: \n',classification_report(Y_validation, predicciones_n))
    print('El reporte de clasificacion para L2 es: \n',classification_report(Y_validation, predicciones_l2))
    #### Clasificacion de nuevos registros  ####
    X_new = pd.DataFrame({'Sexo':['1']})
    print('La nueva clasificacion para No penalizado es: ', modelo_n.predict(X_new))
    print('La nueva clasificacion para L2 es: ', modelo_l2.predict(X_new))
    print(' ')
#### Se crea y entrenan los modelos de SVM #### 
print('------------------------------------------------------------')
print('Algoritmo SVM generando matrices de confusion')
print('------------------------------------------------------------')
# Se modifica iterativamente el hiperparametro C 
for i, c in enumerate([1, 0.1, 0.01]):
    print('------------------------------------------------------------')
    print('Modelos de diferentes penalizaciones con hiperparametro C = ', c)
    print('------------------------------------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Ejecutando el clasificador con distintos valores de C
    classifier = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Se grafican las matrices de cifusion
    titles_options = [("Matriz de confusion, sin normalizacion", None), ("Matriz de confusion normalizada", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Greens, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()
    ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = classifier.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
#### Se crea y entrenan los modelos de Arboles de Decision #### 
print('------------------------------------------------------------')
print('Algoritmo de arboles de decision')
print('------------------------------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i, maximo_hojas in enumerate([2, 3, 4]):
    print('------------------------------------------------------------')
    print('Modelos de arbol con hiperparametro numero maximo de hojas = ', maximo_hojas)
    print('------------------------------------------------------------')
    clf = DecisionTreeClassifier(max_leaf_nodes=maximo_hojas, random_state=0)
    clf.fit(X_train, y_train)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # Se comienza con la identificación del nodo raíz (0) y su profundidad (0) 
    while len(stack) > 0:
        # `pop` asegura que cada nodo solo se visite una vez 
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Si el hijo izquierdo y derecho de un nodo no es el mismo, tenemos un nodo dividido 
        is_split_node = children_left[node_id] != children_right[node_id]
        # Si es un nodo dividido, agregue los elementos secundarios izquierdo y derecho y 
        # la profundidad a la "pila" para que podamos recorrerlos.
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("La estructura de arbol binaria tiene {n} nodos y tiene la siguiente estructura de arbol:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}nodo={node} es nodo hoja.".format( space=node_depth[i] * "\t", node=i))
        else:
            print("{space}nodo={node} is un nodo divido: va al nodo {left} si X[:, {feature}] <= {threshold} "
                  "sino va al nodo {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i]))
     # Se grafican los resultados a modo de comparacion.
     ### Se valida la generalizacion del modelo con particion de muestra por la claridad de resultados que aporta####
    validation_size = 0.20
    seed = 7
    X_trains, X_validation, Y_trains, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
    predicciones = clf.predict(X_validation)
    # Se visualiza el Error Cuadratico Medio
    print("El error cuadratico medio es: %.2f" % mean_squared_error(Y_validation, predicciones))
    # Se visualiza el Puntaje de Varianza. El mejor puntaje es un 1.0
    print('Puntaje de varianza es: %.2f' % r2_score(Y_validation, predicciones))
    # Reporte de resultados #
    print('El reporte de clasificacion es: \n',classification_report(Y_validation, predicciones))
    tree.plot_tree(clf)
    plt.show()