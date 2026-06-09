Clasificación de Tumores Mamarios mediante Perceptrón Multicapa (MLP)
Este repositorio contiene la resolución práctica de un problema de clasificación binaria en el ámbito de la salud médica (HealthTech). El objetivo es entrenar una red neuronal densa para determinar si un tumor mamario es benigno o maligno a partir de características morfológicas celulares.
________________________________________
📝 Descripción de la Problemática
El cáncer de mama es una de las principales causas de muerte en la población femenina a nivel mundial. Un diagnóstico temprano y preciso es crucial para aumentar las tasas de supervivencia.
El método estándar para evaluar una masa mamaria sospechosa es la Aspiración con Aguja Fina (FNA). A partir de las imágenes digitalizadas de este procedimiento, se extraen características geométricas de los núcleos celulares presentes en la muestra.
El desafío de este proyecto consiste en automatizar este análisis: utilizando las métricas geométricas extraídas, se construye un clasificador basado en un Perceptrón Multicapa (MLP) capaz de predecir el diagnóstico con alta precisión, reduciendo el margen de error humano y agilizando la toma de decisiones médicas.
________________________________________
📊 Conjunto de Datos: Wisconsin Breast Cancer
El proyecto utiliza el célebre dataset Breast Cancer Wisconsin (Diagnostic), el cual está disponible directamente de forma nativa en la librería Scikit-learn.
¿Cómo obtener el dataset en Python?
Para cargar y explorar los datos dentro del código, se utiliza el módulo datasets de la siguiente manera:
Python
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Cargar el conjunto de datos
cancer_data = load_breast_cancer()

# Convertir a DataFrame de Pandas para análisis exploratorio
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target  # 0 = Maligno, 1 = Benigno
Características del Dataset:
•	Total de Muestras: 569 instancias (pacientes).
•	Características ($X$): 30 atributos numéricos continuos que describen propiedades de los núcleos celulares, tales como:
o	Radio, Textura, Perímetro, Área y Suavidad.
o	Compacidad, Concavidad y Puntos cóncavos.
o	Simetría y Dimensión fractal.
•	Variable Objetivo ($Y$): Binaria.
o	0: Maligno (Requiere intervención médica urgente).
o	1: Benigno (No cancerígeno).
________________________________________
🤖 Arquitectura de la Red: Perceptrón Multicapa (MLP)
Al tratarse de datos tabulares con características ya extraídas (y no de imágenes en bruto), una Red Neuronal Densa (FNN / MLP) es la arquitectura óptima.
El flujo de procesamiento diseñado para este ejercicio consta de las siguientes etapas:
1.	Preprocesamiento Crítico: Dado que las características tienen escalas muy diferentes (por ejemplo, el área maneja valores en miles y la suavidad valores menores a 1), es obligatorio aplicar una estandarización (StandardScaler) antes de alimentar a la red.
2.	Capas Ocultas: Bloques de capas densas (Dense) que utilizan funciones de activación ReLU para aprender las combinaciones no lineales de la morfología celular.
3.	Capa de Salida: Una única neurona con activación Sigmoid que entrega la probabilidad de que el tumor pertenezca a la clase positiva.
________________________________________

