# Regresión logística que predice la aprobación de un alumno

Autores: Fernandez, Laura. Sandoval, Cristhian

El objetivo del presente trabajo es el desarrollo de un modelo de Regresión Logística que intente predecir las etiquetas de Aprobado (1) / Desaprobado (0) de un alumno en un contexto universitario, en base a tres variables continuas que consisten en la calificación del primer parcial, la calificación del segundo parcial y la nota de un trabajo final integrador. Siendo el rango numérico de estos tres valores la escala numérica de [0, 10].

## Uso de Python

Python es un **lenguaje de programación interpretado**, de **código abierto**, **multiplataforma**, de **tipado dinámico** y **multiparadigma**, ya que soporta programación orientación a objetos, programación imperativa y, en menor medida, programación funcional. 

Python sigue los principios de legibilidad y transparencia, haciendo hincapié en una sintaxis que favorezca un **código legible**. Los lenguajes que siguen estos principios se dice que son "pythonicos", en contraposición con aquellos cuyo código es opaco u ofuscado, llamados "no pythonicos".

Por todas estas razones, para poder visualizar el proyecto y probar se requiere instalar **Python**.

## Instalación de Python

Python se descarga como un instalador desde la **página oficial de Python** y su instalación consiste en dar doble click, ir siguiendo los pasos que muestra el programa y no olvidarse de elegir que configure el **PATH o ruta de ejecución** automáticamente. Hacer clic en [Instalar Python 3.9](https://www.python.org/ftp/python/3.9.5/python-3.9.5-amd64.exe)
Python está incluido en el repositorio de paquetes de las ditribuciones de Linux, por lo que su instalación es sumamente sencilla.

La secuencia de comandos es la siguiente:

```	
# Actualiza la base de datos local de paquetes disponibles para la distribucion Linux
sudo apt-get update

# Instala el paquete Python
sudo apt-get install python3.9

# Instala el administrador de paquetes de Python
sudo apt-get install pip3

```

Una vez instalado Python podemos ver la versión exacta ejecutando el siguiente comando:
```
python3.9 -V
```

## Instalación de las librerias necesarias

Python utiliza una serie de liberías que extienden su funcionalidad en una amplia gama de aplicaciones. Para el presente aprendizaje automàtico particularmente, utiliza tres librerías Pandas, NumPy, Matplotlib y Scikit Learn, que permitirán calcular, visualizar los gráficos y realizar el aprendizaje, respectivamente.

```
pip3 install numpy
pip3 install scikit-learn
pip3 install matplotlib
pip3 install pandas
o
apt-get install python3-pandas
```


## Instalación de git para descargar el proyecto

Git fue desarrollado haciendo foco en la **eficiencia**, la **confiabilidad** y **compatibilidad** del mantenimiento de versiones de aplicaciones cuando estas tienen un gran número de archivos de **código fuente**. Su propósito es llevar registro de los cambios en archivos de computadora incluyendo coordinar el trabajo que varias personas realizan sobre archivos compartidos en un repositorio de código.

Para instalarlo se debe hacer clic en [Git 2.32.0](https://github.com/git-for-windows/git/releases/download/v2.32.0.windows.1/Git-2.32.0-64-bit.exe) si se usa Windows.

O realizar el siguiente comando si se usa Windows.

```
sudo apt-get install git
```

## Ejecutar el proyecto

Para ejecutar el proyecto hay que posicionarse sobre una carpeta y abrir una consola de comandos **Windows** se realiza **Mantener Shift y clic derecho** seleccionando la opción **Abrir consola de ccomandos aquí** y en linux **Clic derecho. Acciones. Abrir terminal aquí** seguiemdo la secuencia de comandos siguiente:

```
git clone https://github.com/cristhiansandoval/regresion_logistica_aprobacion.git
cd regresion_logistica_aprobacion
python3.9 regresion_logistica_aprobacion.py
```

## Referencias y más información
- [Python.org](https://www.python.org/)
- [Documentación de Python3](https://docs.python.org/3/)
- [Python (wikipedia)](https://es.wikipedia.org/wiki/Python)
