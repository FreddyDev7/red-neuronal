from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score

# Cargar los datos
datos = load_breast_cancer()
C= datos.data     #Caracteristicas del tumor
E= datos.target   #Etiquetas [1-benigno, 0-maligno]

# Datos dividos en entrenamiento y prueba
C_entrenamiento, C_prueba, E_entrenamiento, E_prueba = train_test_split(C, E, test_size=0.2 , random_state=12)

# Normalizar los datos
escalador = StandardScaler()
C_entrenamiento = escalador.fit_transform(C_entrenamiento)
C_prueba= escalador.transform(C_prueba)

# Creando y entrenando la Red Neuronal MLP( Perceptron Multicapa)
red_neuronal = MLPClassifier(hidden_layer_sizes=(30,8), activation='relu', max_iter=1000, random_state=12)
red_neuronal.fit(C_entrenamiento, E_entrenamiento)

# Evaluar 
predicciones = red_neuronal.predict(C_prueba)
precision = accuracy_score(E_prueba, predicciones)
print(f"Precision del modelo : {precision * 100 :.2f}%")

# Prediciendo por datos del usuario
def predecir_tumor():
   print("\n*** Predictor del tipo de tumor (benigno o maligno) ***")
   while True:
    try: 
       dato_usuario = input("Ingresa los 30 valores de un tumor(separados por coma): ")
       valores = list(map(float, dato_usuario.split(',')))
       if len (valores)!= 30: 
         print ("Error, son 30 valores a ingresar")
         continue 
       tumor = escalador.transform([valores])
       pred = red_neuronal.predict(tumor)
       resultado ="Benigno" if pred[0] == 1 else "Maligno"
       print(f"\n Resultado : El tumor es {resultado} ")
       break 
    except ValueError:
      print("Error : Ingresa solo numeros separados por comas")
    except Exception as e: 
      print("Error inesperado: {e}")

predecir_tumor()
