# PAQUETERIA 

#import pandas as pd

from matplotlib import pyplot as plt
import statistics as st
import imageio.v2 as imageio

import pickle
import os

import numpy as np

from math import cos, sin, radians, atan2, degrees, floor, sqrt, ceil, exp

from random import random, randint
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import GlorotNormal

import networkx as nx

np.random.seed(2024)

# Función para guardar las listas de una generación específica
def guardar_listas_generacion(generacion, vendedores, compradores, evoluciondelmercado, resumengenD, resumengenP, folder_path):
    with open(os.path.join(folder_path, f"evoluciondelmercado_{generacion}.pkl"), "wb") as f:
        pickle.dump(evoluciondelmercado, f)
    
    with open(os.path.join(folder_path, f"resumengenD_{generacion}.pkl"), "wb") as f:
        pickle.dump(resumengenD, f)
    
    with open(os.path.join(folder_path, f"resumengenP_{generacion}.pkl"), "wb") as f:
        pickle.dump(resumengenP, f)

    with open(os.path.join(folder_path, f"vendedores_{generacion}.pkl"), "wb") as f:
        pickle.dump(vendedores, f)
    
    with open(os.path.join(folder_path, f"compradores_{generacion}.pkl"), "wb") as f:
        pickle.dump(compradores, f)

# Función para cargar las listas de una generación específica
def cargar_listas_generacion(generacion, folder_path):
    with open(os.path.join(folder_path, f"evoluciondelmercado_{generacion}.pkl"), "rb") as f:
        evoluciondelmercado = pickle.load(f)
    
    with open(os.path.join(folder_path, f"resumengenD_{generacion}.pkl"), "rb") as f:
        resumengenD = pickle.load(f)
    
    with open(os.path.join(folder_path, f"resumengenP_{generacion}.pkl"), "rb") as f:
        resumengenP = pickle.load(f)
        
    with open(os.path.join(folder_path, f"vendedores_{generacion}.pkl"), "rb") as f:
        vendedores = pickle.load(f)
    
    with open(os.path.join(folder_path, f"compradores_{generacion}.pkl"), "rb") as f:
        compradores = pickle.load(f)    
    
    return vendedores, compradores, evoluciondelmercado, resumengenD, resumengenP

# Determinar la última generación completada
def obtener_ultima_generacion(folder_path):
    ultima_generacion = 0
    for filename in os.listdir(folder_path):
        if filename.startswith("evoluciondelmercado_") and filename.endswith(".pkl"):
            generacion = int(filename.split("_")[1].split(".")[0])
            ultima_generacion = max(ultima_generacion, generacion)
    return ultima_generacion

def generacion0 (numeroP):# PARA COMENZAR, EN LA GENERACION 0, 
    # poblar el entorno de redes neuronales de decision de la clase comprador
    compradores=[]
    for P in range(numeroP): 
       # las compradores heredan los atributos y los grados de libertad de los padres
        atributos = [] # valor de los inputs: los no seleccionados a 0
        atributosP = []  # inputs a considerar en la red neuronal: numatritutos valores entre 16
        exitostrainP = []  # lista de valores de angulo y distancia que en los juegos han resultado buenas estrategias
        exitostestP = [] # lista de angulos y distancias que en la huida han funcionado
        exitostestRP = []
        tiemposP = [] # tiempo de la red neuronal en tomar una decision, un valor por cada decision de compraventa
        comprador=[P,
               np.random.uniform(x_min,x_max),# posicion aleatoria de cada comprador
               np.random.uniform(y_min,y_max),
               np.random.uniform(0,drmaxP),# angulo
               np.random.uniform(0,velocidadP),# velocidad
               [], # 5, resultados de los conjuntos de juegos entre amigotes
               [], # 6, contador de eventos compraventa
               0, # 7, contador de encuentros reproductivos           
               1, # 8, fitness 
               0, # 9, antiguedad
               exitostestP, # 10, velocidad y angulo de exitos de huida en la compraventa          
               exitostestRP, # 11, exitos reproductivos (las compradores siempre que se encuentran, ligan)
               exitostrainP, # 12, tabla de angulos y velocidades con exito en la huida (entrenamiento) 
               atributosP, # 13, seleccion de los inputs 
               [], # 14, entrenamientos
               0, # 15, contador de conexiones
               # tupla de maximo 16 que son los atributos de cada comprador
               atributos, # 16, valores de los inputs, se comportará según un input de longitud len(atributosP)
               numprocesadores,
               [], # tabla de valores de atributos utilizada en cada juego          
               [], # Inicializacion entre la entrada y la capa oculta           
               [], # Inicialización de los pesos de las conexiones dentro de la capa oculta 
               [], # Inicializacion de los pesos entre la capa oculta y el output 
               0, # tiempo de proceso para tomar la decision
               1.176 * exp( -0.0862 * (28 - 23)), # 23, conectividad 5G, wifi, bluetooth,
               1.176 * exp( -0.0862 * (23 - 23)), # 24, duracion de la bateria                  
               1.176 * exp( -0.0862 * (34 - 23)), # 25, funciones inteligencia artificial
               1.176 * exp( -0.0862 * (33 - 23)), # 26, ecocertificacion, sostenibilidad
               1.176 * exp( -0.0862 * (30 - 23)), # 27, seguridad y privacidad
               1.176 * exp( -0.0862 * (24 - 23)), # 28, camara y software de video
               1.176 * exp( -0.0862 * (25 - 23)), # 29, memoria
               1.176 * exp( -0.0862 * (29 - 23)), # 30, resolución y tamaño de pantalla
               1.176 * exp( -0.0862 * (35 - 23)), # 31, realidad aumentada y realidad virtual
               1.176 * exp( -0.0862 * (32 - 23)), # 32, integracion con otros sensores y dispositivos
               1.176 * exp( -0.0862 * (26 - 23)), # 33, diseño y estética
               1.176 * exp( -0.0862 * (36 - 23)), # 34, sensores integrados IoT 
               1.176 * exp( -0.0862 * (31 - 23)), # 35, usabilidad y ergonomía
               1.176 * exp( -0.0862 * (37 - 23)), # 36, sensores y funcionalidades de bienestar y salud
               1.176 * exp( -0.0862 * (27 - 23)), # 37, funcionalidades adicionales
               1.176 * exp( -0.0862 * (38 - 23)), # 38, accesorios disponibles 
               tiemposP,
               None
               ]

        # Elegir el numatributos inicial
        comprador[13] = np.random.choice(np.arange(23,39), numatributos, replace=False).tolist()
        for i in range(23,39):  # Campos entre 23 y 38 (inclusive)
            if i not in comprador[13]:
                comprador[i] = 0
        comprador[16] = [comprador[i] for i in range(23, 39)]
        
        compradores.append(comprador)
    
    # poblar el entorno de vendedores con redes neuronales de pesos aleatorios
    vendedores=[] # lista de listas vendedor
    for D in range(numeroD):
        # VALORES INICIALES: generacion 0, tiempo 0
        # la escala de valor y la estructura de la red son heredable, no los pesos
        
        atributosD = [] # lista vacía con un numero variable de entradas por su posicion
        atributos = [] # lista de valores de las entradas
        exitostrainD = [] # lista vacia de dos valores con un numero variable de entradas
        exitostestD = []
        exitostestRD = []
        tiemposD = [] # lista vacia de un valor con un numero variable de entradas
        vendedor = [D,# cada vendedor tiene su lista de valores iniciales
                      np.random.uniform(x_min,x_max),
                      np.random.uniform(y_min,y_max),# posicion aleatoria de cada vendedor
                      np.random.uniform(0,drmaxD),# angulo
                      np.random.uniform(0,velocidadD),# velocidad
                      [], # 5, conjunto de juegos entre dos amigos que comparten intereses: experiencias y fracasos
                      [], # 6, contador de intentos de compraventa
                      0, # 7, contador de eventos reproductivos                  
                      ratioCV, # 8, fitness o gordura                  
                      0, # 9, antiguedad 
                      exitostestD, # 10, tabla de angulos y velocidades con exito de compraventa                   
                      exitostestRD, # 11, tabla de angulos y velocidades con exitos reproductivos
                      exitostrainD, # 12, tabla de angulos y velocidades con exito (entrenamiento)
                      atributosD, # 13, inputs que se van a considerar != 0
                      [], # 14, contador de entrenamientos
                      0, # 15, contador de conexiones
                      atributos, # 16, valores de los inputs elejidos entre 23 a 38 
                                  # (los no seleccionados en atributosD, son =0)
                      numprocesadores, # 17, dos hidden layers iguales que simulan una recurrente
                      [], # 18, lista de inputs para el aprendizaje
                      [], # Inicializacion entre la entrada y la capa oculta  
                      [], # Inicializacion dentro de la capa oculta
                      [], # Inicialización de los pesos de las conexiones entre la capa oculta y la capa de salida
                      0, # 22, tiempo de proceso para tomar la decision
                      1.176 * exp( -0.0862 * (30 - 23)), # 23, conectividad 5G, wifi, bluetooth,
                      1.176 * exp( -0.0862 * (28 - 23)), # 24, duracion de la bateria                  
                      1.176 * exp( -0.0862 * (31 - 23)), # 25, funciones inteligencia artificial
                      1.176 * exp( -0.0862 * (27 - 23)), # 26, ecocertificacion, sostenibilidad
                      1.176 * exp( -0.0862 * (32 - 23)), # 27, seguridad y privacidad
                      1.176 * exp( -0.0862 * (23 - 23)), # 28, camara y software de video
                      1.176 * exp( -0.0862 * (24 - 23)), # 29, memoria
                      1.176 * exp( -0.0862 * (25 - 23)), # 30, resolución y tamaño de pantalla
                      1.176 * exp( -0.0862 * (33 - 23)), # 31, realidad aumentada y realidad virtual
                      1.176 * exp( -0.0862 * (29 - 23)), # 32, integracion con otros sensores y dispositivos
                      1.176 * exp( -0.0862 * (26 - 23)), # 33, diseño y estética
                      1.176 * exp( -0.0862 * (34 - 23)), # 34, sensores integrados IoT 
                      1.176 * exp( -0.0862 * (35 - 23)), # 35, usabilidad y ergonomía
                      1.176 * exp( -0.0862 * (36 - 23)), # 36, sensores y funcionalidades de bienestar y salud
                      1.176 * exp( -0.0862 * (37 - 23)), # 37, funcionalidades adicionales
                      1.176 * exp( -0.0862 * (38 - 23)), # 38, accesorios disponibles 
                      tiemposD, # 39, tabla de tiempos de cada decision neuronal
                      None
                      ] #generacion y paso de tiempo 
        
        # Elegir el numatributos inicial
        vendedor[13] = np.random.choice(np.arange(23,39), numatributos, replace=False).tolist()
        for i in range(23,39):  # Campos entre 23 y 38 (inclusive)
            if i not in vendedor[13]:
                vendedor[i] = 0

        vendedor[16] = [vendedor[i] for i in range(23, 39)]
        
        vendedores.append(vendedor)
        
    return (vendedores,compradores)

# Función para normalizar los datos
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def prune(model, threshold=0.1):
    pruned_weights = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights = layer.get_weights()
            if len(weights) > 0:
                pruned_weights.append(np.where(np.abs(weights[0]) >= threshold, weights[0], 0.0))
                pruned_weights.append(np.where(np.abs(weights[1]) >= threshold, weights[1], 0.0))
    
    return pruned_weights

def neuralD(vendedor):
    # si el vendedor viene de una generacion anterior, no aprende mas
    if vendedor[40] == None: # si ya tiene experiencia anterior
        # tambien puede haber pollos que no hayan acumulado experiencia pero no se hayan descartado para evitar demasiadas bajas
        # estos quedan como cachorros para otra oportunidad en la siguiente generacion
        if len(vendedor[18]) != 0:
            # Obtener datos de entrada y salida para el entrenamiento
            train_inputs = np.array(vendedor[18])
            train_outputs = np.array(vendedor[12])
            # Convierte tus listas de Python a tensores
            train_inputs = tf.convert_to_tensor(train_inputs, dtype=tf.float32)
            train_outputs = tf.convert_to_tensor(train_outputs, dtype=tf.float32)
       
            # Crear un modelo secuencial
            modeloD = Sequential()
            # Capa de entrada con el número de atributosD
            modeloD.add(Input(shape=(len(vendedor[18][0],)), name="input_layer", dtype='float32'))
    
            # Capa hidden numprocesadoresD inicialización Xavier solo la primera vez 
            pesosinput = np.array(vendedor[19], dtype=np.float32)
            if pesosinput.size == 0:  # Verificar si los pesos están vacíos
                modeloD.add(Dense(vendedor[17], activation='relu', kernel_initializer=GlorotNormal(), name="hidden_layer"))
            # cuando ya tiene pesos, la inicializacion es con la experiencia anterior
            else:
                modeloD.add(Dense(vendedor[17], activation='relu', name="hidden_layer"))
                modeloD.layers[-1].set_weights([vendedor[19], modeloD.layers[-1].get_weights()[1]])  # incializa con pesos y sesgos previos
        
            # Segunda capa oculta para simular una capa recurrente, todos con todos
            pesoshidden = np.array(vendedor[20], dtype=np.float32)
            if pesoshidden.size == 0:  # Verificar si los pesos están vacíos
                modeloD.add(Dense(vendedor[17], activation='relu', kernel_initializer=GlorotNormal(), name="recurrent_layer"))
            else:
                modeloD.add(Dense(vendedor[17], activation='relu', name="recurrent_layer"))
                modeloD.layers[-1].set_weights([vendedor[20],modeloD.layers[-1].get_weights()[1]])  # inicializa con pesos y sesgos previos
        
            # Capa output con dos nodos: angulo y velocidad
            pesosoutput = np.array(vendedor[21], dtype=np.float32)
            if pesosoutput.size == 0:  # Verificar si los pesos están vacíos
                modeloD.add(Dense(2, activation='linear', kernel_initializer=GlorotNormal(), name="output_layer"))
            else:
                modeloD.add(Dense(2, activation='linear', name="output_layer"))
                modeloD.layers[-1].set_weights([vendedor[21], modeloD.layers[-1].get_weights()[1]])  # inicializa con pesos y sesgos previos
        
            # Compilar el modelo
            modeloD.compile(loss = 'mean_squared_error', optimizer = 'adam')    
            
            # Configurar EarlyStopping
            early_stopping_monitor = EarlyStopping(
                monitor = 'val_loss',  # Monitorear la pérdida de validación
                patience = 5,         # Detener el entrenamiento después de 10 épocas sin mejora
                restore_best_weights = True,  # Restaurar los pesos del modelo a la mejor época
                mode = 'min',
                verbose = 0
            )
            
            batch_size = 10
            
            # Medir el tiempo de inicio
            start_time = time.process_time()
            
            numjuegos=len(train_inputs)
            # Entrenamiento con EarlyStopping
            history = modeloD.fit(train_inputs, train_outputs, validation_split=0.2, epochs=100, batch_size = batch_size, callbacks=[early_stopping_monitor], verbose=0)
            
            # Medir el tiempo de finalización
            end_time = time.process_time()
            # Calcular la duración del entrenamiento
            tiemporeaccionD = (end_time - start_time)/numjuegos
            
            utilizadas = len(history.history['loss'])
            if 'val_loss' in history.history:
                utilizadasD = min(utilizadas, len(history.history['val_loss']))
            
            #Plot del historial de pérdida
            #plt.plot(history.history['loss'], label='train_loss')
            #plt.plot(history.history['val_loss'], label='val_loss')
            #plt.xlabel('Épocas')
            #plt.ylabel('Loss')
            #plt.legend()
            #plt.title('Training Loss y Validation Loss' + str(vendedor[0]))
            #plt.show()
            
            # Podar pesos y sesgos 
            pruned_weights = prune(modeloD, threshold=0.1)
            vendedor[19] = pruned_weights[0]
            vendedor[20] = pruned_weights[2]
            vendedor[21] = pruned_weights[4]     
    
            vendedor[40] = modeloD
            
            # Calcular la modularidad entre las capas ocultas
            if pesoshidden.size != 0:
                # Crear un grafo dirigido a partir de la matriz de pesos específicos entre las capas ocultas
                G = nx.DiGraph(pesoshidden)
                # Calcular las comunidades utilizando un algoritmo de detección de comunidades
                comunidades = nx.algorithms.community.greedy_modularity_communities(G)
                # Calcular la modularidad
                modularidad = nx.a0lgorithms.community.quality.modularity(G, comunidades)
                # Guardar el valor de modularidad en comprador[15]
                vendedor[15] = modularidad
            else:
                # No se pueden calcular los pesos específicos entre las capas ocultas
                vendedor[15] = None

    return vendedor, tiemporeaccionD, utilizadasD

def neuralP(comprador):  
    if comprador[40] == None:  
        if len(comprador[18]) != 0:
            train_inputs = np.array(comprador[18])
            train_outputs = np.array(comprador[12])
            train_inputs = tf.convert_to_tensor(train_inputs, dtype=tf.float32)
            train_outputs = tf.convert_to_tensor(train_outputs, dtype=tf.float32)
           
            modeloP = Sequential()
            modeloP.add(Input(shape=(len(comprador[18][0],)), name="input_layer", dtype='float32'))
    
            pesosinput = np.array(comprador[19], dtype=np.float32)
            if pesosinput.size == 0:  
                modeloP.add(Dense(comprador[17], activation='relu', kernel_initializer=GlorotNormal(), name="hidden_layer"))
            else:
                modeloP.add(Dense(comprador[17], activation='relu', name="hidden_layer"))
                modeloP.layers[-1].set_weights([comprador[19], modeloP.layers[-1].get_weights()[1]])  
        
            pesoshidden = np.array(comprador[20], dtype=np.float32)
            if pesoshidden.size == 0:  
                modeloP.add(Dense(comprador[17], activation='relu', kernel_initializer=GlorotNormal(), name="recurrent_layer"))
            else:
                modeloP.add(Dense(comprador[17], activation='relu', name="recurrent_layer"))
                modeloP.layers[-1].set_weights([comprador[20], modeloP.layers[-1].get_weights()[1]])  
        
            pesosoutput = np.array(comprador[21], dtype=np.float32)
            if pesosoutput.size == 0:  
                modeloP.add(Dense(2, activation='linear', kernel_initializer=GlorotNormal(), name="output_layer"))
            else:
                modeloP.add(Dense(2, activation='linear', name="output_layer"))
                modeloP.layers[-1].set_weights([comprador[21], modeloP.layers[-1].get_weights()[1]])  
        
            modeloP.compile(loss='mean_squared_error', optimizer='adam')
            batch_size=10
            
            # Definir EarlyStopping (supondremos que las crias de herviboros necesitan menos juegos para aprender, paciencia la mitad que los vendedores)
            early_stopping_monitor = EarlyStopping(patience=10, monitor='val_loss', mode='min', restore_best_weights=True, verbose=0)
            start_time = time.process_time()
            # Entrenar con EarlyStopping
            history = modeloP.fit(train_inputs, train_outputs, validation_split=0.2, epochs=100, batch_size = batch_size, callbacks=[early_stopping_monitor], verbose=0)
    
            end_time = time.process_time()
            tiemporeaccionP = (end_time - start_time) / len(train_inputs)
            
            # Poda con OBC
            pruned_weights = prune(modeloP, threshold=0.1)
            comprador[19] = pruned_weights[0]
            comprador[20] = pruned_weights[2]
            comprador[21] = pruned_weights[4]
            
            # Obtener el número de experiencias utilizadas hasta EarlyStopping
            utilizadas = len(history.history['loss'])
            if 'val_loss' in history.history:
                utilizadasP = min(utilizadas, len(history.history['val_loss']))
            
            comprador[40] = modeloP
            
            # Calcular la modularidad entre las capas ocultas
            if pesoshidden.size != 0:
                # Crear un grafo dirigido a partir de la matriz de pesos específicos entre las capas ocultas
                G = nx.DiGraph(pesoshidden)
                # Calcular las comunidades utilizando un algoritmo de detección de comunidades
                comunidades = nx.algorithms.community.greedy_modularity_communities(G)
                # Calcular la modularidad
                modularidad = nx.algorithms.community.quality.modularity(G, comunidades)
                # Guardar el valor de modularidad en comprador[15]
                comprador[15] = modularidad
            else:
                # No se pueden calcular los pesos específicos entre las capas ocultas
                comprador[15] = None

    return comprador, tiemporeaccionP, utilizadasP

def compraventa(vendedor, comprador):
    # por si acaso se colara el 39 como atributo en alguna reproducción
    comprador[13] = [valor for valor in comprador[13] if 23 <= valor <= 38]
    vendedor[13] = [valor for valor in vendedor[13] if 23 <= valor <= 38]
    # atributos por ambos compartidos
    intcomun = list(np.intersect1d(vendedor[13], comprador[13]))
    exito = None  # inicializamos
    # se puede hacer todo lo complicado que se quiera, por eso necesitamos una funcion aparte
    # incorporar atributos puede hacer mas lenta la decision, asi que para que merezca la pena, debe haber una presion en contra
    # solo aumentaran atributos si el tiempo de decision lo admite, esto a la vez presiona hacia una mejor eficiencia
    if vendedor[22] <= comprador[22]: # si el vendedor atiende al comprador es posible que haya operacion
        # si el comprador es mas lento en decidir, se penaliza al vendedor solamente en coste de oportunidad
        contadorD = 0
        contadorP = 0
        if intcomun:
            for i in intcomun:  # Campos no 0 entre 23 y 38 (inclusive) 
                if exito is not None:  # Salir del bucle si ya ha habido compraventa
                    break
                
                if comprador[i] != 0 and vendedor[i] != 0:
                    if vendedor[i] > comprador[i]: # el vendedor asigna un precio a la memoria mayor al que el comprador aprecia como valor
                        # al hacerlo secuencial damos mas importancia al primer atributo común que al segundo, y sucesivamente
                        # Obtenemos la posición del atributo en el vector 13 del vendedor
                        #posD = vendedor[13].index(i)
                        #if posD != 0: # salvo que este el primero, el atributo promociona en prioridad
                        #    vendedor[13][posD], vendedor[13][posD-1] = vendedor[13][posD-1], vendedor[13][posD]
                        contadorD += (vendedor[i]-comprador[i]) * (1.176 * exp( -0.0862 * (i - 23))) #tener mas atributos sube el contador ante el que tenga menos, aunque sean de mas valor                 
                        comprador = refuerzoP (comprador, i) # el comprador se adapta algo a la tendencia del mercado

                    else: # el comprador da mayor valor tener 5G de lo que el precio de tener 5G propone el comprador
                        contadorP += (comprador[i]-vendedor[i]) * (1.176 * exp( -0.0862 * (i - 23)))
                        vendedor = refuerzoD (vendedor, i) # el vendedor debe ajustar su precio a la tendencia del mercado                       
                    
            if contadorD > contadorP: # al tener atributos de mayor prioridad mayor, el comprador opina que es una mala oferta
                # el vendedor cree que su producto tiene mayor valor que lo que cree el comprador, y este no compra 
                exito = False # no compraventa
                # promociona el atributo con el que logrado escapar
                #posD = vendedor[13].index(i)
                #posP = comprador[13].index(i)
                #if posP != 0:
                #    comprador[13][posP],comprador[13][posP-1] = comprador[13][posP-1],comprador[13][posP]
                
                # la comprador busca otra tienda aleatoria
                comprador[1] = np.random.uniform(x_min,x_max)
                comprador[2] = np.random.uniform(y_min,y_max)

            else:
                exito = True # el comprador decide comprar    

        else: # si no comparten ningun atributo comun, el vendedor no entiende al comprador
            exito = False
            if vendedor[8] > comprador[8]:
                vendedor[8] -= comprador[8]*ratioCV  / pasostiempo # hay un coste por no estar en el mercado
 
            else: # pero el comprador tambien ha perdido el tiempo
                comprador[8] -= vendedor[8] / pasostiempo  
                
                comprador[1] = np.random.uniform(x_min,x_max)
                comprador[2] = np.random.uniform(y_min,y_max) 
            # se va de la tienda porque no ha encontrado lo que busca
            
    else: # si la comprador es mas rapido, se cansa de esperar y se larga
        exito = False
        comprador[8] += - vendedor[8]*ratioCV / pasostiempo  # consume eneria
        vendedor[8] += - comprador[8] / pasostiempo # consume energía
        
    return vendedor, comprador, exito

def refuerzoD(vendedor, i):
    # Según el producto, se cambian estas reglas, que pueden ser incluso no lineales
    
    # Cobertura 5G, wifi y bluetooth 
    if i == 23 and 1 > vendedor[23] > 0:    
        vendedor[23] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Cuanta más duración de la batería
    if i == 24 and 1 > vendedor[24] > 0:    
        vendedor[24] += 1 - ((1 / 16) *(i - 23)) / pasostiempo     
    # Incorporación de licencias chatGPT, Bing, Dall-E, voice chatbot,...
    if i == 25 and 1 > vendedor[25] > 0:    
        vendedor[25] += 1 - ((1 / 16) *(i - 23)) / pasostiempo   
    # Ecocertificación, variable que empeora con la rapidez de decisiones
    if i == 26 and 1 > vendedor[26] > 0:    
        vendedor[26] += 1 - ((1 / 16) *(i - 23)) / pasostiempo 
    # Seguridad y privacidad
    if i == 27 and 1 > vendedor[27] > 0:    
        vendedor[27] += 1 - ((1 / 16) *(i - 23)) / pasostiempo    
    # Resolución de la cámara y software de enfoque, edición de fotos y videos
    if i == 28 and 1 > vendedor[28] > 0:    
        vendedor[28] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Memoria y almacenamiento
    if i == 29 and 1 > vendedor[29] > 0:    
        vendedor[29] += 1 - ((1 / 16) *(i - 23)) / pasostiempo    
    # Tamaño, resolución de la pantalla
    if i == 30 and 1 > vendedor[30] > 0:    
        vendedor[30] += 1 - ((1 / 16) *(i - 23)) / pasostiempo    
    # Realidad Aumentada y gafas de realidad virtual
    if i == 31 and 1 > vendedor[31] > 0:    
        vendedor[31] += 1 - ((1 / 16) *(i - 23)) / pasostiempo   
    # Integración con otros sensores y dispositivos 
    if i == 32 and 1 > vendedor[32] > 0:    
        vendedor[32] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Diseño, ergonomía y estética
    if i == 33 and 1 > vendedor[33] > 0:    
        vendedor[33] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Sensores, integración y funcionalidades IoT
    if i == 34 and 1 > vendedor[34] > 0:    
        vendedor[34] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Usabilidad y accesibilidad
    if i == 35 and 1 > vendedor[35] > 0:    
        vendedor[35] += 1 - ((1 / 16) *(i - 23)) / pasostiempo    
    # Sensores, integración y funcionalidades de salud y bienestar
    if i == 36 and 1 > vendedor[36] > 0:    
        vendedor[36] += 1 - ((1 / 16) *(i - 23)) / pasostiempo   
    # Funcionalidades extra
    if i == 37 and 1 > vendedor[37] > 0:    
        vendedor[37] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Accesorios disponibles
    if i == 38 and 1 > vendedor[38] > 0:    
        vendedor[38] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    
    # Para evitar valores negativos o excesivos
    for j in range(23, 39): 
        if vendedor[j] < 0:
            vendedor[j] = 0.01
        if vendedor[j] > 1:
            vendedor[j] = 0.99
    
    # Actualización del vector de valores de los atributos en 16,
    print("vendedor[13]:", vendedor[13])
    print("vendedor[16]:", vendedor[16])
    
    # Asegurarse de que los valores en vendedor[16] sean accesibles
    for k in vendedor[13]:
        if 23 <= k < 39:  # Asegúrate de que k está en el rango [23, 38]
            index = k - 23
            if 0 <= index < len(vendedor[16]):
                print(f"Actualizando vendedor[16][{index}] con vendedor[{k}] = {vendedor[k]}")
                vendedor[16][index] = vendedor[k]
            else:
                print(f"Índice fuera de rango para vendedor[16]: {index}")
        else:
            print(f"Índice fuera de rango en vendedor[13]: {k}")
    
    # Imprimir los resultados finales para verificar
    print("vendedor[16] actualizado:", vendedor[16])

    return vendedor

def refuerzoP (comprador,i):
    # segun el producto se cambian estas reglas, que pueden ser incluso no lineales    
    # Cobertura 5G, wifi y bluetooth 
    if i == 23 and 1 > comprador[23] > 0:    
        comprador[23] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Cuanta más duración de la batería
    if i == 24 and 1 > comprador[24] > 0:    
        comprador[24] += 1 - ((1 / 16) *(i - 23)) / pasostiempo     
    # Incorporación de licencias chatGPT, Bing, Dall-E, voice chatbot,...
    if i == 25 and 1 > comprador[25] > 0:    
        comprador[25] += 1 - ((1 / 16) *(i - 23)) / pasostiempo   
    # Ecocertificación, variable que empeora con la rapidez de decisiones
    if i == 26 and 1 > comprador[26] > 0:    
        comprador[26] += 1 - ((1 / 16) *(i - 23)) / pasostiempo 
    # Seguridad y privacidad
    if i == 27 and 1 > comprador[27] > 0:    
        comprador[27] += 1 - ((1 / 16) *(i - 23)) / pasostiempo    
    # Resolución de la cámara y software de enfoque, edición de fotos y videos
    if i == 28 and 1 > comprador[28] > 0:    
        comprador[28] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Memoria y almacenamiento
    if i == 29 and 1 > comprador[29] > 0:    
        comprador[29] += 1 - ((1 / 16) *(i - 23)) / pasostiempo    
    # Tamaño, resolución de la pantalla
    if i == 30 and 1 > comprador[30] > 0:    
        comprador[30] += 1 - ((1 / 16) *(i - 23)) / pasostiempo    
    # Realidad Aumentada y gafas de realidad virtual
    if i == 31 and 1 > comprador[31] > 0:    
        comprador[31] += 1 - ((1 / 16) *(i - 23)) / pasostiempo   
    # Integración con otros sensores y dispositivos 
    if i == 32 and 1 > comprador[32] > 0:    
        comprador[32] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Diseño, ergonomía y estética
    if i == 33 and 1 > comprador[33] > 0:    
        comprador[33] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Sensores, integración y funcionalidades IoT
    if i == 34 and 1 > comprador[34] > 0:    
        comprador[34] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Usabilidad y accesibilidad
    if i == 35 and 1 > comprador[35] > 0:    
        comprador[35] += 1 - ((1 / 16) *(i - 23)) / pasostiempo    
    # Sensores, integración y funcionalidades de salud y bienestar
    if i == 36 and 1 > comprador[36] > 0:    
        comprador[36] += 1 - ((1 / 16) *(i - 23)) / pasostiempo   
    # Funcionalidades extra
    if i == 37 and 1 > comprador[37] > 0:    
        comprador[37] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    # Accesorios disponibles
    if i == 38 and 1 > comprador[38] > 0:    
        comprador[38] += 1 - ((1 / 16) *(i - 23)) / pasostiempo
    
    # Para evitar valores negativos o excesivos
    for j in range(23, 39): 
        if comprador[j] < 0:
            comprador[j] = 0.01
        if comprador[j] > 1:
            comprador[j] = 0.99
    
    # Actualización del vector de valores de los atributos en 16,
    print("comprador[13]:", comprador[13])
    print("comprador[16]:", comprador[16])
    
    # Asegurarse de que los valores en comprador[16] sean accesibles
    for k in comprador[13]:
        if 23 <= k < 39:  # Asegúrate de que k está en el rango [23, 38]
            index = k - 23
            if 0 <= index < len(comprador[16]):
                print(f"Actualizando comprador[16][{index}] con comprador[{k}] = {comprador[k]}")
                comprador[16][index] = comprador[k]
            else:
                print(f"Índice fuera de rango para comprador[16]: {index}")
        else:
            print(f"Índice fuera de rango en comprador[13]: {k}")
    
    # Imprimir los resultados finales para verificar
    print("comprador[16] actualizado:", comprador[16])

    return comprador

def decisionD(vendedor):
    # mediremos el tiempo que tarda la red neuronal en tomar una decision
    inicio_tiempo = time.process_time()
    
    datos_prediccion = np.array([vendedor[16]])  # atributos almacenados en el vendedor
    predicciones = vendedor[40].predict(datos_prediccion)

    tiempoD = time.process_time() - inicio_tiempo

    # Desnormalizar las predicciones si es necesario
    alphaD = predicciones[0,0]  
    carreraD = predicciones[0,1]  

    return alphaD, carreraD, tiempoD

def decisionP(comprador): # viene a ser el entorno de test 
    # Hacer predicciones y medir el tiempo
    inicio_tiempo = time.process_time()
    datos_prediccion = np.array([comprador[16]])  # atributosP almacenados en comprador
    predicciones = comprador[40].predict(datos_prediccion)
    
    tiempoP = time.process_time() - inicio_tiempo

    # Desnormalizar las predicciones si es necesario
    alphaP = predicciones[0,0]  
    carreraP = predicciones[0,1]  

    return alphaP, carreraP, tiempoP

# para que no se salgan fuera del ecosistema cuadrado
def controlfrontera(vendedor,comprador):
    # para que no se salga del ecosistema
    if vendedor[1] > x_max + x_max*0.25:
        vendedor[1]=vendedor[1]-(x_max+x_max*0.25)+(x_min+x_min*0.25) 
    if vendedor[2] > y_max+y_max*0.25:
        vendedor[2]=vendedor[2]-(y_max+y_max*0.25)+(y_min+y_min*0.25)
    if vendedor[1] < x_min+x_min*0.25:
        vendedor[1]=vendedor[1]+(x_max+x_max*0.25)-(x_min+x_min*0.25)
    if vendedor[2] < y_min+y_min*0.25:
        vendedor[2]=vendedor[2]+(y_max+y_max*0.25)-(y_min+y_min*0.25)
        
    if comprador[1] > x_max+x_max*0.25:
        comprador[1]=comprador[1]-(x_max+x_max*0.25)+(x_min+x_min*0.25) 
    if comprador[2] > y_max+y_max*0.25:
        comprador[2]=comprador[2]-(y_max+y_max*0.25)+(y_min+y_min*0.25)
    if comprador[1] < x_min+x_min*0.25:
        comprador[1]=comprador[1]+(x_max+x_max*0.25)-(x_min+x_min*0.25)
    if comprador[2] < y_min+y_min*0.25:
        comprador[2]=comprador[2]+(y_max+y_max*0.25)-(y_min+y_min*0.25)

def contarno0D(vendedores):
    # Sustituimos los valores de 0.01 por 0 y los valores de 0.99 por 1
    for vendedor in vendedores:
        for i in range(23, 39):
            if 0.009 < vendedor[i] < 0.011:
                vendedor[i] = 0
            elif 0.989 < vendedor[i] < 0.991:
                vendedor[i] = 1
    
    # Inicializamos
    cantidadno0D = [0] * 16   
    sumasno0D = [0] * 16 
    mediasno0D = [0] * 16

    # Recorremos las posiciones de la 23 a la 38
    for i in range(23, 39):
        # Calculamos la suma de valores no cero en la posición i
        sumasno0D[i - 23] = sum(vendedor[i] for vendedor in vendedores if isinstance(vendedor, list) and vendedor[i] != 0)
        
        # Contamos el número de valores no cero en la posición i
        cantidadno0D[i - 23] = sum(1 for vendedor in vendedores if isinstance(vendedor, list) and vendedor[i] != 0)

    # Calculamos las medias de valores no cero en cada posición
    for i in range(16):
        if cantidadno0D[i] != 0:  # Verificamos que haya al menos un valor no cero en la posición i
            mediasno0D[i] = sumasno0D[i] / cantidadno0D[i]
    
    return cantidadno0D, mediasno0D

def contarno0P(compradores):
    # Sustituimos los valores de 0.01 por 0 y los valores de 0.99 por 1
    for comprador in compradores:
        for i in range(23,39):
            if 0.009 < comprador[i] < 0.011:
                comprador[i] = 0
            elif 0.989 < comprador[i] < 0.991:
                comprador[i] = 1
    # Inicializamos
    cantidadno0P = [0] * 16   
    sumasno0P = [0] * 16  
    mediasno0P = [0] * 16

    # Recorremos los 16 elementos de los vectores
    for i in range(23,39):
        # Calculamos la suma de valores no cero en la posición i
        sumasno0P[i-23] = sum(compradores[j][i] for j in range(len(compradores)) if isinstance(compradores[j], list) and isinstance(compradores[j][i], int) and compradores[j][i] != 0)
        # Contamos el número de valores no cero en la posición i
        cantidadno0P[i-23] = sum(1 for j in range(len(compradores)) if isinstance(compradores[j], list) and isinstance(compradores[j][i], int) and compradores[j][i] != 0)

    # Calculamos las medias de valores no cero en cada posición
    for i in range(16):
        if cantidadno0P[i] != 0:  # Verificamos que haya al menos un valor no cero en la posición i
            mediasno0P[i] = sumasno0P[i] / cantidadno0P[i]
    
    return cantidadno0P, mediasno0P

def hijo(atributos_padre1, atributos_padre2):
    # Calcular la media de la cantidad de atributos de ambos padres
    media = (len(atributos_padre1) + len(atributos_padre2)) / 2
    
    # Elegir aleatoriamente entre el entero menor o mayor más cercano a la media
    num_atributos_hijo = int(media)
    if randint(0,1) == 1:
        num_atributos_hijo += 1
    
    # Inicializar los índices para recorrer los atributos de los padres
    index_padre1 = 0
    index_padre2 = 0
    
    # Lista para almacenar los atributos del hijo
    atributos_hijo = []
    
    # Comenzar a construir los atributos del hijo considerando las prioridades de los padres
    for i in range(num_atributos_hijo):
        # Alternar entre los padres para seleccionar los atributos
        if i % 2 == 0:
            # Si hay atributos disponibles en el padre1 y no están en el hijo, añadirlos
            while index_padre1 < len(atributos_padre1) and atributos_padre1[index_padre1] in atributos_hijo:
                index_padre1 += 1
            if index_padre1 < len(atributos_padre1):
                atributos_hijo.append(atributos_padre1[index_padre1])
                index_padre1 += 1
        else:
            # Si hay atributos disponibles en el padre2 y no están en el hijo, añadirlos
            while index_padre2 < len(atributos_padre2) and atributos_padre2[index_padre2] in atributos_hijo:
                index_padre2 += 1
            if index_padre2 < len(atributos_padre2):
                atributos_hijo.append(atributos_padre2[index_padre2])
                index_padre2 += 1
    
    return atributos_hijo

def mutacion (atributos, numprocesadores):
    # la probabilidad debe ser simetrica en mas o menos, y sera la presion selectiva la que promueva mas uno u otro sentido              
    mutacion = random() # la mutacion es simetrica sumando o restando procesador
    if mutacion <= 3*tasamutacion:               
        # añadir o eliminar un nodo procesador en ambas capas hidden (para mantener la simulacion de recurrencia)
        if randint(0, 1) == 1:# elige al azar 0, 1
            numprocesadores += 1
        else:
            if numprocesadores > 5: # por debajo de un minimo es ser tonto del pueblo
                numprocesadores -= 1
                    
    mutacion=random()
    if mutacion <= tasamutacion:   
        # añadir o eliminar uno de los atributos, no hay preferencia
        numatributos=len(list(filter(lambda x: x != 0, atributos)))
        caracruz = randint(0, 1) # moneda al aire: 0 o 1
        if caracruz == 1:
            numatributos += 1
            atributonuevo = randint(23,39) # del 23 al 38, ambos incluidos
            while atributonuevo in atributos: # se cambia hasta encontrar uno no repetido
                atributonuevo = randint(23,39) # para evitar un atributo repetido
            atributos.append(atributonuevo)
        else:
            if numatributos > 1:
                numatributos -= 1
                atributos.pop(randint(0, numatributos)) #se quita un atributo al azar 
            else:
                numatributos = 1

    return atributos, numprocesadores

# preparar datos para las graficas
def lista_a_diccionario(datos):
    
    etiquetas = ['jornada', 'identificador', 'transacciones', 'beneficios', 'nuevos agentes', 'antiguedad', 'atributos', 'entrenamientos', 'modularidad', 'procesadores', 'tiempo de proceso', 'conectividad 5G, wifi, bluetooth, radiofrecuencia','duracion de la bateria','funciones inteligencia artificial','ecocertificacion, sostenibilidad','seguridad y privacidad','camara y software de video','memoria','resolución y tamaño de pantalla','realidad aumentada y realidad virtual','integracion con otros sensores y dispositivos','diseño y estética','sensores integrados IoT','usabilidad y ergonomia','sensores y funcionalidades de bienestar y salud','otras funcionalidades adicionales','accesorios disponibles']
    diccionario = {}

    for i, etiqueta in enumerate(etiquetas):
        diccionario[etiqueta] = datos[i]

    return diccionario

# CREAR GRAFICOS PARA CADA PASO DE TIEMPO EN CADA GENERACION
def dibujarcompraventa(vendedores,compradores,generacion,tiempo):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(9.6, 5.4)

    plt.xlim([x_min+x_min* 0.25,x_max+x_max* 0.25])
    plt.ylim([y_min+y_min* 0.25,y_max+y_max* 0.25])
    # vendedores    
    maxfitnessD = max([vendedor[8] for vendedor in vendedores])
   
    for D, vendedor in enumerate(vendedores):
        etiqueta = vendedor[3] # angulo decidido
        tamano = 100*vendedor[8]/maxfitnessD # tamaño proporcional a su exito en comer
        color = (tamano, 0, 0)  # Componente rojo proporcional a la intensidad, verde en 0 y azul en 0
        
        plt.scatter(vendedor[1], vendedor[2],
                    s=tamano, c=[color], alpha=0.5, label=etiqueta)
    # compradores
    maxfitnessP = max([comprador[8] for comprador in compradores])
    
    for P,comprador in enumerate(compradores):
        etiqueta = comprador[3] # angulo decidido
        tamano = 33*comprador[8]/maxfitnessP # tamaño proporcional a su exito en comer
        color = (0,tamano, 0)  # Componente verde proporcional a la intensidad, verde en 0 y azul en 0
        
        plt.scatter(comprador[1], comprador[2],
                    s=tamano, c=[color], marker='*', alpha=0.5)
    # ENTORNO    
    ax.set_aspect('equal')
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    # Informacion    
    plt.figtext(0.025, 0.95, r'COMIENDO')
    plt.figtext(0.025, 0.90, r'JORNADA: ' + str(generacion) )
    plt.figtext(0.025, 0.85, r'TIEMPO: ' + str(tiempo))
    plt.figtext(0.025, 0.80, r'vendedores: ' + str(len(vendedores)))
    plt.figtext(0.025, 0.75, r'compradores: ' + str(len(compradores)))
    historia = str(generacion)  + '-' + str(tiempo) + '.png'
    plt.legend()
    plt.savefig(historia , dpi=100)    
    # Cerrar la figura después de guardarla para ahorrar memoria
    plt.close(fig)
    
def dibujarreproduccion(vendedores,vendedoreshijos,compradores,compradoreshijos,generacion,tiempo,pasostiempo):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(9.6, 5.4)

    plt.xlim([x_min+x_min* 0.25,x_max+x_max* 0.25])
    plt.ylim([y_min+y_min* 0.25,y_max+y_max* 0.25])
    # vendedores    
    maxfitnessD = max([vendedor[8] for vendedor in vendedores])
    for D, vendedor in enumerate(vendedores):
        tamano = 100*vendedor[8]/maxfitnessD # tamaño proporcional a su exito en comer
        color = (tamano, 0, 0)  # Componente rojo proporcional a la intensidad, verde en 0 y azul en 0
        
        plt.scatter(vendedor[1], vendedor[2],
                    s=tamano, c=[color], alpha=0.5)
    # CACHORROS DE LOS vendedores    
    for D, vendedor_hijo in enumerate(vendedoreshijos):
        tamano = 50 
       
        plt.scatter(vendedor[1], vendedor[2],
                    s=tamano, c='r', alpha=0.5)
    # compradores
    maxfitnessP = max([comprador[8] for comprador in compradores])  
    for P,comprador in enumerate(compradores):
        tamano = 33*comprador[8]/maxfitnessP # tamaño proporcional a su exito en comer
        color = (0,tamano, 0)  # Componente verde proporcional a la intensidad, verde en 0 y azul en 0
        
        plt.scatter(comprador[1], comprador[2],
                    s=tamano, c=[color], marker='*', alpha=0.5)
    # CACHORROS DE compradores
    for P,comprador in enumerate(compradoreshijos):
        tamano = 33
        
        plt.scatter(comprador[1], comprador[2],
                    s=tamano, c='g', marker='^', alpha=0.5)
    # ENTORNO    
    ax.set_aspect('equal')
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    # GRAFICO   
    plt.figtext(0.025, 0.95, r'LIGANDO')
    plt.figtext(0.025, 0.90, r'JORNADA: ' + str(generacion) )
    plt.figtext(0.025, 0.85, r'TIEMPO: ' + str(pasostiempo+tiempo))
    plt.figtext(0.025, 0.80, r'vendedores: ' + str(len(vendedores)))
    plt.figtext(0.025, 0.75, r'compradores: ' + str(len(compradores)))
    plt.figtext(0.025, 0.70, r'nuevos vendedores: ' + str(len(vendedoreshijos)))
    plt.figtext(0.025, 0.65, r'nuevos compradores: ' + str(len(compradoreshijos)))
    historia = str(generacion)  + '-' + str(pasostiempo+tiempo) + '.png'
    plt.legend()
    plt.savefig(historia , dpi=100)    

    # Cerrar la figura después de guardarla para ahorrar memoria    
    plt.close(fig)
    
def dibujar_grafica(evoluciondelmercado):
    
    generaciones = [poblacion[0] for poblacion in evoluciondelmercado]
    compradores = [poblacion[1] for poblacion in evoluciondelmercado]
    #compradores_fallecidas = [poblacion[2] for poblacion in evoluciondelmercado]
    compradores_nacidas = [poblacion[3] for poblacion in evoluciondelmercado]
    vendedores = [poblacion[4] for poblacion in evoluciondelmercado]
    #vendedores_fallecidos = [poblacion[5] for poblacion in evoluciondelmercado]
    vendedores_nacidos = [poblacion[6] for poblacion in evoluciondelmercado]

    colores_compradores = ['green', 'limegreen', 'mediumseagreen']
    colores_vendedores = ['red', 'darkred', 'firebrick']

    plt.plot(generaciones, compradores, label='compradores', color=colores_compradores[0])
    #plt.plot(generaciones, compradores_fallecidas, label='compradores Fallecidas', color=colores_compradores[1])
    plt.plot(generaciones, compradores_nacidas, label='nuevos compradores', color=colores_compradores[2])
    plt.plot(generaciones, vendedores, label='vendedores', color=colores_vendedores[0])
    #plt.plot(generaciones, vendedores_fallecidos, label='vendedores Fallecidos', color=colores_vendedores[1])
    plt.plot(generaciones, vendedores_nacidos, label='nuevos vendedores', color=colores_vendedores[2])
    plt.xlabel('Jornadas')
    plt.ylabel('Cantidad')
    # Ajustar la posición y tamaño de la leyenda fuera de la gráfica
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='x-small')
    plt.show()

def graficarevolucion(resumengenD, resumengenP, etiqueta, limite):

    # vendedores
    keneracionesD = []
    depre = {}
    for D, vendedor in enumerate(resumengenD):
        keneracionD = vendedor['jornada']  # generacion
        identif = vendedor['identificador']  # identificador
        valor = vendedor[etiqueta]  # valor del parametro ordenada
        keneracionesD.append(keneracionD)  # guardo esos 3 valores en keneraciones
        # en depre se guardaran todos los pares generacion-valordeordenada que tengan el mismo identificador
        if identif not in depre:
            depre[identif] = {'jornadas': [], 'valores': []}
        depre[identif]['jornadas'].append(keneracionD)
        depre[identif]['valores'].append(valor)

    # compradors
    keneracionesP = []
    pres = {}
    for P, comprador in enumerate(resumengenP):
        keneracionP = comprador['jornada']  # generacion
        identif = comprador['identificador']  # identificador
        valor = comprador[etiqueta]  # valor del parametro ordenada
        keneracionesP.append(keneracionP)  # guardo esos 3 valores en keneraciones
        # en depre se guardaran todos los pares generacion-valordeordenada que tengan el mismo identificador
        if identif not in pres:
            pres[identif] = {'jornadas': [], 'valores': []}
        pres[identif]['jornadas'].append(keneracionP)
        pres[identif]['valores'].append(valor)

    # Configurar colores y marcas
    vendedor_color = iter(plt.cm.Reds(np.linspace(0, 1, len(depre))))
    comprador_color = iter(plt.cm.Greens(np.linspace(0, 1, len(pres))))
    markers = ['o', 'x']
    
    if limite == 1:
        plt.ylim(0, 1)
    
    if limite == 0.1:
        plt.ylim(0, 0.1)
    
    # para cada vendedor
    for D, (depr, data) in enumerate(depre.items()):
        plt.plot(data['jornadas'], data['valores'], color=next(vendedor_color), marker=markers[D % len(markers)], linestyle='-')

    # para cada millon de compradores
    for P, (pre, data) in enumerate(pres.items()):
        plt.plot(data['jornadas'], data['valores'], color=next(comprador_color), marker=markers[P % len(markers)], linestyle='solid')

    plt.xlabel('jornada')
    plt.ylabel(etiqueta.capitalize())
    plt.grid(True)
    plt.show()

def graficar_generacion(generaciones, datos1, datos2, titulo, limite):
    # Calcular los límites del rango intercuartílico para datos1
    q1_datos1 = np.percentile(datos1, 25)
    q3_datos1 = np.percentile(datos1, 75)
    iqr_datos1 = q3_datos1 - q1_datos1
    lower_bound_datos1 = q1_datos1 - 1.5 * iqr_datos1
    upper_bound_datos1 = q3_datos1 + 1.5 * iqr_datos1

    # Calcular los límites del rango intercuartílico para datos2
    q1_datos2 = np.percentile(datos2, 25)
    q3_datos2 = np.percentile(datos2, 75)
    iqr_datos2 = q3_datos2 - q1_datos2
    lower_bound_datos2 = q1_datos2 - 1.5 * iqr_datos2
    upper_bound_datos2 = q3_datos2 + 1.5 * iqr_datos2

    # Filtrar datos1 y datos2 eliminando los valores atípicos
    datos1_filtrados = [d if lower_bound_datos1 <= d <= upper_bound_datos1 else np.nan for d in datos1]
    datos2_filtrados = [d if lower_bound_datos2 <= d <= upper_bound_datos2 else np.nan for d in datos2]

    plt.figure(figsize=(10, 6))  # Ajustar el tamaño de la figura
    plt.plot(generaciones, datos1_filtrados, marker='o', color='green')
    plt.plot(generaciones, datos2_filtrados, marker='x', color='brown')
    
    
    if limite == 1: # Configurar la escala del eje y entre 0 y 1
        plt.ylim(0, 1)
    
    plt.xlabel('Jornadas')
    plt.ylabel('Valor')
    plt.title(titulo)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def animacion():
    # lista de archivos PNG en la carpeta actual  
    archivos_png = [archivo for archivo in os.listdir() if archivo.endswith('.png')]
    archivos_png.sort()  # Ordenar los archivos en orden alfanumérico

    imagenes = []
    for archivo in archivos_png:
        imagen = imageio.imread(archivo)
        imagenes.append(imagen)

    # Guardar la animación en formato GIF    
    imageio.mimsave('animacion.gif', imagenes, duration=0.2)
    # imageio.imread('animacion.gif',imagenes,duration=0.2)
    # return 'animacion.gif' que se guarda en la carpeta

# VALORES INICIALES
    
numeroD = 50   # NUMERO DE vendedores
ratioCV = 1  # regla trofica: num compradores para beneficios suficientes de un vendedor
generaciones = 365 # NUMERO DE GENERACIONES

# PADRES REPRODUCTORES, compradores QUE CALIFICAN LIKES,...
padresD = 0.2         # PORCIÓN MINIMA DE padresD
antiguedadD = 20

# PADRES compradores, PRODUCTOS (LOS QUE SE REPONEN), VOTOS
padresP = 0.1              # PORCIÓN MINIMA DE padresP
antiguedadP = 10

tasamutacion = 0.1        # TASA DE MUTACIONES

amistad = 5

# GENERACION (gen_tiempo = gen_vendedores = ratioDP * gen_compradores)

ratioDP = 1         # generaciones de compradores por cada gen vendedores
gen_tiempo = 1     # VIDA DE LA GENERACION         
dt = 0.01           # paso de tiempo, año, mes o como sea que se mida (dt)

# ATRIBUTOS DINAMICOS DE LOS vendedores, PARTIDOS, EMcompradores,...

drmaxD = np.pi        # ROTACION      (radianes por segundo) siempre ataca
velocidadD = 3         # VELOCIDAD     (unidades por segundo)

# ATRIBUTOS DINAMICOS DE LAS compradores, VOTOS, START-UPs

drmaxP = np.pi * 2        # ROTACION      (radianes por pasotiempo o segundo o dia o lo que sea)
velocidadP = 5        # VELOCIDAD    (unidades por segundo)

# ENTORNO, ECOSISTEMA, ESTANTERIA, URNA,...

x_min = -1.0        # borde izquierdo
x_max = 1.0         # borde derecho
y_min = -1.0        # borde inferior
y_max = 1.0         # borde superior

# RED NEURONAL DE LOS vendedores, compradores,... 

numatributos = 5        # NODOS SENSORES
numacciones = 2          # NODOS ACTORES
numprocesadores = ceil(sqrt(numatributos*numacciones))     # NODOS PROCESADORES

learnrate = 0.1   # velocidad de aprendizaje

# medir tiempos
tiempoinicial = time.process_time()

# hay que controlar que las poblaciones sean sostenibles
    
numpadresD = int(floor(padresD*numeroD)) # num min padres para que sigan las generaciones
numhijosD = numeroD - numpadresD # num de vendedores que deberian reproducirse en la siguiente generacion
numeroP = numeroD*ratioCV # num de compradores   
numpadresP = int(floor(padresP*numeroP))
numhijosP = numeroP - numpadresP
compradoreshijos = [] # solo para inicializar, tras el primer bucle ya no es necesario
vendedoreshijos = []

# vida de la generacion / medida en dias del paso de tiempo
pasostiempo=int(gen_tiempo/dt)

# GRAFICO (llama a la funcion dibujar) SITUACION INICIAL GEN=0, T=0 
#dibujarcompraventa(vendedores,compradores,generacion=0,tiempo=0)    # devuelve grafico png 

evoluciondelmercado = []
resumengenD = [] # vendedores en cada generacion
resumengenP = [] # compradores en cada generacion

folder_path = "mercado365decreciente"
   # aqui se guardaran las generaciones
ultima_generacion_completada = obtener_ultima_generacion(folder_path) # el numero de generacion mayor guardado
if ultima_generacion_completada > 0:
    vendedores, compradores, evoluciondelmercado, resumengenD, resumengenP = cargar_listas_generacion(ultima_generacion_completada, folder_path)
    print(f"Se reanudará desde la generación {ultima_generacion_completada + 1}")
else:
    vendedores , compradores = generacion0 (numeroP)
   
for generacion in range(ultima_generacion_completada + 1, generaciones):
# en realidad la vida de la generacion es x3, pues 1/3 a juegos, 1/3 a compraventa y 1/3 a reproduccion
    # inicializamos las variables:
    falsosconsumidores = 0   # compradores que no saben lo que quieren
    compradoresfuerademercado = 0   # no tienen capacidad de gasto 
    compradoresquecompranP = []  # compradores que han decidido comprar 
    falsosvendedores = 0 # vendedores que no entienden el mercado
    dep = [] # vendedores en quiebra, sin financiacion
    vendedoresfuerademercado = 0
    vendedoresquevendenD = []
    # la diferencia entre vendedores dep y compradores sin pasta, es que unos quiebran y los otros tienen otra oportunidad cuando tengan presupuesto
    utilizadasD = 50 # juegos que el earlystopping definira como los necesarios para aprender
    utilizadasP = 50  # aqui solo se inicializan
    MbeneficioD = ratioCV
    MbeneficioP = 1
    MreproduccionD = 5
    MreproduccionP = 5
    MexitosD = 1
    MexitosP = 1
    MtiempoprocesoD = 1
    MtiempoprocesoP = 1
    M23P = 0.5
    M23D = 0.5
    M24P = 0.5
    M24D = 0.5
    M25P = 0.5
    M25D = 0.5
    M26P = 0.5
    M26D = 0.5
    M27P = 0.5
    M27D = 0.5
    M28P = 0.5
    M28D = 0.5
    M29P = 0.5
    M29D = 0.5
    M30P = 0.5
    M30D = 0.5
    M31P = 0.5
    M31D = 0.5
    M32P = 0.5
    M32D = 0.5
    M33P = 0.5
    M33D = 0.5
    M34P = 0.5
    M34D = 0.5
    M35P = 0.5
    M35D = 0.5
    M36P = 0.5
    M36D = 0.5
    M37P = 0.5
    M37D = 0.5
    M38P = 0.5
    M38D = 0.5
    # SON JOVENES, JUEGAN PERO NO SE COMEN NI MATAN UNOS A OTROS, ENSAYAN Y ENTRENAN
    # ACUMULANDO EXPERIENCIA SOBRE LAS ESTRATEGIAS (ANGULO Y VELOCIDAD) QUE FUNCIONAN
    # ESTOS DATOS SERAN PARA ENTRENAR CADA UNO A SU RED NEURONAL: APRENDIZAJE
    # el juego define el conjunto de train: valores de los inputs (16) y outputs (2)
    for tiempo in range(1, pasostiempo, 1): 
        amistadD = amistad
        # COMIENZA LA INVESTIGACION DE MERCADO               
        for fabricanteD,vendedor in enumerate(vendedores):
            if vendedores[fabricanteD][40] == None: # pregunta a otros si no tiene experiencia
                distanciaDP = []
            
                for fabricanteP,vendedor in enumerate(vendedores):
                    # solo si ambos no tienen experiencia, hacen pruebas, los demás no estan interesados en compartir informacion: ya tienen su set de train
                    if vendedores[fabricanteP][40] == None:
                        if fabricanteP == fabricanteD:
                            DDP = 5 # si no hacemos esta ficcion, se pierde el indice de fabricanteP
                        else:
                        # el cachorro que juega de vendedor mide la distancia al cachorro vendedor mas proximo para que juegue a comprador    
                            DDP = sqrt((vendedores[fabricanteD][1]-vendedores[fabricanteP][1])**2 + (vendedores[fabricanteD][2]-vendedores[fabricanteP][2])**2)
                        distanciaDP.append(DDP)
                    
                if distanciaDP:
                    # entre los que puede jugar fabricanteD, elegimos el cachorro P más próximo
                    DPmin = np.argmin(np.array(distanciaDP)) # el cachorro mas proximo es vendedores[DPmin]

                    if distanciaDP[DPmin] >= 0.1: # para evitar superpuestos (por casualidad o DDP=5)   
                        # prueban valores 
                        # AD es un vector con los valores compartidos por los roles de comprador y vendedor
                        AD = list(np.intersect1d(vendedores[fabricanteD][13], vendedores[DPmin][13]))
                        lenad = len(AD) # si solo tiene un valor sera AD[0], si tiene mas, el prioritario sera AD[0]
                        lenD = len(vendedores[fabricanteD][13]) # numatributos en 13
                        lenP = len(vendedores[DPmin][13]) # numatributos en 13
                        triunfos = 0
                        fracasos = 0 # se refiere solo a las estrategias en estas consultas
                        NDP = distanciaDP[DPmin] # inicializa para el primer bucle: la primera experiencia supone que la anterior ha sido satisfactoria
                    
                        # angulo de ataque
                        alfaD = radians(atan2((vendedores[DPmin][2]-vendedores[fabricanteD][2]), (vendedores[DPmin][1]-vendedores[fabricanteD][1]))) - (vendedores[DPmin][3] - vendedores[fabricanteD][3])
                        
                        if amistadD < 1:    # regula la cantidad de juegos para que sea un numero que permita aprender multiplicando o limitando los juegos entre dos amigos
                            amistadD = 1
                            
                        # son mas amigos cuantos mas intereses en comun tengan
                        # con pocos atributos es poco probable que se hagan amigos y menos que sean muy amigos, pero segun crecen los atributos tendran mas experiencias train
                        if lenad != 0: # si ambos no van del mismo rollo en algo, no hay experiencia util y no juegan
                            # es mas facil tener exito por haber aumentado la distancia que acercarse lo suficiente, por lo que los vendedores deberan jugar mas que las compradores
                            for _ in range(1,amistadD*lenad): # ahora que se han hecho amigos pues comparten intereses, jugaran un numero de juegos = amistad
                                if len(vendedores[fabricanteD][18]) < utilizadasD: # a partir de un numero "utilizados" juegos, maduran y ya se consideran experimentados         
                                    ADusado=[]
                                    for i in range(23,39):                                    
                                        # si ambos tienen uno o mas intereses en comun, i.e. uno ofrece certificado de trazabilidad de su producto,
                                        # y el otro busca productos con trazabilidad certificada 
                                        if i in AD and i not in ADusado:                               
                                            # los atributos compartidos AD, tendran prioridad para el vendedor distinta de la prioridad de la comprador:
                                            # la dispersion de su decision sera menor si es mas prioritario (exploran menos si lo tienen mas claro)
                                            # o cuanto menos prioritario, mas abierto a opciones
                                            # definiremos una lista en 18 de atributos en los que uno, i, va a variar segun criterio anterior
                                            # los otros atributos de AD conservan los valores guardados en 16
                                            posicion_AD = vendedores[fabricanteD][13].index(i)
                                            prioridadD = (posicion_AD + 1) / lenad + 1
                                            posicion_AP = vendedores[DPmin][13].index(i)
                                            prioridadP = (posicion_AP + 1) / lenad + 1
                                            
                                            # para dar mas importancia por ejemplo al precio que al tamaño de la pantalla, 
                                            # introducimos la variabilidad segun su prioridad en los atributos herantiguedados
                                            VD = np.random.normal(1,prioridadD)
                                            VAD = vendedores[fabricanteD][i] * VD # valor del atributo que se oferta
                                            # para controlar que no nos salgamos del intervalo (0,1), de los valores normalizados del atributo
                                            if VAD <= 0:
                                                VAD = 0.01
                                            elif VAD >= 1:
                                                VAD = 0.99
                        
                                            # si fabricanteD juega a precio y DPmin a calidad, cada uno tendrá una variabilidad segun su prioridad
                                            # se encontraran con menos probabilidad que si ambos juegan al mismo rollo
                                            # lo mismo que si uno juega a velocidad y el otro juega a angulo (bajar precio o cambiar orientacion del producto)
                                            VP = np.random.normal(1,prioridadP)
                                            VAP = vendedores[DPmin][i] * VP # valor del atributo que se demanda
                                            if VAP <= 0:
                                                VAP = 0.01
                                            elif VAP >= 1:
                                                VAP = 0.99
                                                
                                            ADusado.append(i)  # en el siguiente bucle no se repetira este i                                  
                                            # cada uno tiene asi un valor en el atributo comun, precio, color ofrecido-disponible, o dientes-cuernos    
                                            # supondremos que cuanto mejor dotado esta el atributo, mejor se acercara al objetivo
                                            # mas claro lo tiene y con menos dudas, mas directo y veloz va
                                            # el paso de tiempo es 1, así que la distancia de la carrera es la velocidadD=vendedor[4]
                                            
                                            # cuanto mas valor tiene un atributo, mas intensidad en su incursion en el mercado, mas publicidad, mas inversion, mas velocidad en la carrera
                                            # pero el atributo VAD tiene valor<1
                                            carreraD = vendedores[fabricanteD][4]*VAD + np.random.normal(0,vendedores[fabricanteD][4]*prioridadD)
                                            if carreraD < 0: carreraD = 0
                                            # ambos tienen intenciones distintas (angulo de ataque con cierta variabilidad, que es su estrategia)
                                            # supondremos que cuanto mas importante es la variable esta en orden correlativo en 13
                                            # atributos mas importantes y de mayor valor, estaran mas focalizados (menor variabilidad en el giro)
                                            alphaD = alfaD + np.random.normal(0,(1-VAD)*prioridadD)
                                            while alphaD > np.pi*2:
                                                alphaD = alphaD - np.pi*2 
                                            while alphaD < -np.pi*2:
                                                alphaD = alphaD + np.pi*2
                                            
                                            # al compartir valores, el que juega a comprador, toma la misma decision de acercarse
                                            # pero si su prioridad es menor, su incertidumbre sera mayor
                                            carreraP = vendedores[DPmin][4]*VAP + np.random.normal(0,vendedores[DPmin][4]*prioridadP) 
                                            if carreraP < 0: carreraP = 0
                                            # si huye lo hara en la misma direccion que el que ataca  
                                            alphaP = alfaD + np.random.normal(0,(1-VAP)*prioridadP)
                                            while alphaP > np.pi*2:
                                                alphaP = alphaP - np.pi*2 
                                            while alphaP < -np.pi*2:
                                                alphaP = alphaP + np.pi*2
                                            
                                            # enfilada la proxima comprador, va a por ella con un margen estocástico
                                            vendedores[fabricanteD][1] = vendedores[fabricanteD][1] + carreraD*cos(alphaD)
                                            vendedores[fabricanteD][2] = vendedores[fabricanteD][2] + carreraD*sin(alphaD)
                
                                            # pero la comprador embestira (se gira) al vendedor o huira
                                            if VAD > VAP: # huye 
                                                vendedores[DPmin][1] = vendedores[DPmin][1] + carreraP*cos(alphaP)
                                                vendedores[DPmin][2] = vendedores[DPmin][2] + carreraP*sin(alphaP)
                                            else: # embiste porque se gira
                                                vendedores[DPmin][1] = vendedores[DPmin][1] - carreraP*cos(alphaP)
                                                vendedores[DPmin][2] = vendedores[DPmin][2] - carreraP*sin(alphaP)
                                            
                                            # volvemos a calcular la distancia tras la experiencia de ambos
                                            NDPf = sqrt((vendedores[fabricanteD][1]-vendedores[DPmin][1])**2 + (vendedores[fabricanteD][2]-vendedores[DPmin][2])**2)
                                            
                                            if abs(NDP - NDPf) <= 0.1: # se han acercado
                                                if VAD >= VAP: # precio o calidad ofrecida mejor de la demandada
                                                    # train input
                                                    vectorVAD = vendedores[fabricanteD][16]
                                                    vectorVAD[i-23] = VAD
                                                    # desconozco el motivo, pero para evitar un error
                                                    vectorVAD = [0 if val == 0.0 or val == [0.0] else val for val in vectorVAD]                        
                                                    vendedores[fabricanteD][18].append(vectorVAD)
                                                    # train output
                                                    vendedores[fabricanteD][12].append([carreraD,alphaD])
                                            else:
                                                fracasos += 1
                                            # ahora tienen una nueva distancia entre ellos, sea mayor o menor    
                                            NDP = NDPf
                                
                            vendedores[fabricanteD][5].append([triunfos,fracasos])
                            
                            # alejamos al amigo para dar oportunidad a otros de jugar con el fabricanteD
                            vendedores[DPmin][1] = np.random.uniform(x_min,x_max)
                            vendedores[DPmin][2] = np.random.uniform(y_min,y_max)
                            # en 16 estan los valores de los atributos, en 18 estan los valores de exploracion
                            # en 18 tienen cierta variabilidad segun las prioridades y son para explorar y adquirir experiencia
                
                controlfrontera(vendedores[fabricanteD],vendedores[DPmin]) # para controlar que no se escapen fuera del cuadrado
                            
        # ahora las compradores hacen lo mismo
        for consumidorP,comprador in enumerate(compradores):
            if compradores[consumidorP][40] == None:
                distanciaPD=[]     
                for consumidorD,comprador in enumerate(compradores):                     
                    if compradores[consumidorD][40] == None: # ambos deben ser consumidores sin experiencia
                        if consumidorD == consumidorP:# si se elimina la distancia consigo misma el indice corre una posicion
                            DPD=5
                        # la cria que juega de vendedor busca a la cria comprador mas proxima y le cita   
                        else:
                            DPD= sqrt((compradores[consumidorP][1]-compradores[consumidorD][1])**2 + (compradores[consumidorP][2]-compradores[consumidorD][2])**2)
                        distanciaPD.append(DPD)

                if distanciaPD:
                    PDmin = np.argmin(np.array(distanciaPD)) # indice de la cria mas proxima a cada consumidorP           

                    if distanciaPD[PDmin] >= 0.1:    # evitar experiencias de paseo entre superpuestos               
                        AP =list(np.intersect1d(compradores[consumidorP][13], compradores[PDmin][13]))
                        lenap = len(AP) # si solo tiene un valor sera AP[0], si tiene mas, el prioritario sera AD[0]
                        lenP = len(compradores[consumidorP][13]) # numatributos en 13
                        lenD = len(compradores[PDmin][13]) # numatributos en 13
                        triunfos = 0  # de la consumidorP
                        fracasos = 0  # de la consumidorP
                        NPD = distanciaPD[PDmin] # solamente sirve para inicializar
                        # angulo de huida 
                        alfaP = radians(atan2((compradores[consumidorP][2]-compradores[PDmin][2]), (compradores[consumidorP][1]-compradores[PDmin][1]))) - (compradores[consumidorP][3] - compradores[PDmin][3])
                        # las crias pueden ser mas y para compensar que aprendan mas que los vendedores... 
                        amistadP = int(amistad/ratioCV)
                        if amistadP < 1: amistadP = 1
                        if lenap != 0: # si ambos no van del mismo rollo, no hay experiencia util y no juegan                        
                            for _ in range(1,amistadP * lenap): # pero si van de lo mismo, se hacen amiguetes y juegan muchas veces
                                if len(compradores[consumidorP][18]) < utilizadasP:  # limitamos los juegos a un valor suficiente para aprender
                                    APusado=[]
                                    for i in range(23,39):
                                        if i in AP and i not in APusado:
                                            posicion_AP = compradores[consumidorP][13].index(i)
                                            prioridadP = (posicion_AP + 1) / lenap + 1
                                            posicion_AD = compradores[PDmin][13].index(i)
                                            prioridadD = (posicion_AD + 1) / lenap + 1
                                            
                                            # para dar mas importancia por ejemplo al precio que al tamaño de la pantalla, 
                                            # introducimos la variabilidad segun su prioridad en los atributos herantiguedados
                                            VP = np.random.normal(1,prioridadP)
                                            VAP = compradores[consumidorP][i] * VP # valor del atributo que se oferta
                                            # para controlar que no nos salgamos del intervalo (0,1), de los valores normalizados del atributo
                                            if VAP <= 0:
                                                VAP = 0.01
                                            elif VAP >= 1:
                                                VAP = 0.99
                        
                                            # si consumidorP juega a precio y PDmin a calidad, cada uno tendrá una variabilidad segun su prioridad
                                            # se encontraran con menos probabilidad que si ambos juegan al mismo rollo
                                            VD = np.random.normal(1,prioridadD)
                                            VAD = compradores[PDmin][i] * VD # valor del atributo que se demanda
                                            if VAD <= 0:
                                                VAD = 0.01
                                            elif VAD >= 1:
                                                VAD = 0.99
                                                
                                            # consumidorP juega a huir, lo que ya esta considerado en alfaP
                                            carreraP = compradores[consumidorP][4]*VAP + np.random.normal(0,compradores[consumidorP][4]*prioridadP)
                                            if carreraP < 0: carreraP = 0
                                            alphaP = alfaP + np.random.normal(0,(1-VAP)*prioridadP)
                                            while alphaP > np.pi * 2:
                                                alphaP = alphaP - np.pi * 2 
                                            while alphaP < -np.pi * 2:
                                                alphaP = alphaP + np.pi * 2
                                            
                                            carreraD = compradores[PDmin][4]*VAD + np.random.normal(0,compradores[PDmin][4]*prioridadD)
                                            if carreraD < 0: carreraD = 0
                                            alphaD = alfaP + np.random.normal(0,(1-VAD)*prioridadD)
                                            while alphaD > np.pi * 2:
                                                alphaD = alphaD - np.pi * 2 
                                            while alphaD < -np.pi * 2:
                                                alphaD = alphaD + np.pi * 2
                                            
                                            compradores[consumidorP][1] = compradores[consumidorP][1] + carreraD * cos(radians(alphaP))
                                            compradores[consumidorP][2] = compradores[consumidorP][2] + carreraD * sin(radians(alphaP))
                                            
                                            if VAP > VAD: # huir  
                                                compradores[PDmin][1]= compradores[PDmin][1] + carreraP * cos(alphaD)
                                                compradores[PDmin][2]= compradores[PDmin][2] + carreraP * sin(alphaD)
                                            else: # embestir
                                                compradores[PDmin][1]= compradores[PDmin][1] - carreraP * cos(alphaD)
                                                compradores[PDmin][2]= compradores[PDmin][2] - carreraP * sin(alphaD)
        
                                            # volvemos a calcular la distancia tras la experiencia de ambos
                                            NPDf = sqrt((compradores[consumidorP][1]-compradores[PDmin][1])**2 + (compradores[consumidorP][2]-compradores[PDmin][2])**2)
                                           
                                            if abs(NPD - NPDf) >= 0.1: # si ha conseguido huir, sea cual sea el tamaño de sus cuernos o 
                                                if VAP >= VAD: # precio o calidad ofrecida mejor de la demandada
                                                    # train input
                                                    vectorVAP = compradores[consumidorP][16]
                                                    vectorVAP[i-23] = VAP
                                                    vectorVAP = [0 if val == 0.0 or val == [0.0] else val for val in vectorVAP]
                                                    compradores[consumidorP][18].append(vectorVAP)
                                                    # train output
                                                    compradores[consumidorP][12].append([carreraP,alphaP])
                                                    triunfos += 1
                                                else:
                                                    fracasos += 1
                                            else: fracasos +=1
                                            
                                            NPD = NPDf 
                                        
                            compradores[consumidorP][5].append([triunfos,fracasos])
                            # alejamos al amigo para dar oportunidad a otros de jugar con el fabricanteD
                            compradores[PDmin][1] = np.random.uniform(x_min,x_max)
                            compradores[PDmin][2] = np.random.uniform(y_min,y_max)
                
                controlfrontera(compradores[consumidorP], compradores[PDmin])
                            
    # la mortalidad infantil es necesaria para el programa (individuos que tienen una configuracion de atributos
    # tal que no podran aprender), pero tambien en la realidad se descartan los pollitos que menos pian o en el futbol quien no va al entreno
    if generacion < 10: juegosD = generacion
    else: juegosD = 10 

    #  Depurar la lista que contiene solo los vendedores que han acumulado experiencia (viejos y nuevos jugones)
    apaticosD= [vendedor for vendedor in vendedores if len(vendedor[18]) <= juegosD]
     
    # ahora los ordenamos para que se eliminen primero los de la generacion anterior que pudieran quedar
    apaticosD = sorted(apaticosD, key=lambda vendedor: vendedor[0])

    for vendedor in vendedores[:]:
        if vendedor in apaticosD:
            falsosvendedores += 1
            vendedores.remove(vendedor)
            # precaucion por si hay demasiados cachorros sin experiencia
            if len(vendedores) <= numpadresD*2: break
            # los cachorros sin experiencia no eliminados tendran una segunda oportunidad de adquirir train en la proxima generacion
        else: # ya que estamos depuro un error extraño que asigna [0.0] a algunos sitios
            vendedor[16] = [0 if val == 0.0 or val == [0.0] else val for val in vendedor[16]]
            vendedor[18] = [0 if val == 0.0 or val == [0.0] else val for val in vendedor[18]]
    print('quedan', len(vendedores),'vendedores para iniciar venta, de los que', len(apaticosD), 'no entienden la demanda')
            
    valoresD = [len(sublistaD[18]) for sublistaD in vendedores if sublistaD[18]]
    if valoresD:
        mediaD = sum(valoresD) / len(valoresD)
        #print("La media de las experiencias en vendedores:", mediaD)
    else:
        print("No hay valores para calcular la media de experiencias para los nuevos vendedores")
    
    # Construir apaticosP dentro del bucle donde se eliminan las compradores inútiles
    if generacion < 10: juegosP = generacion
    else: juegosP = 10 
    
    # Construir apaticosP dentro del bucle donde se eliminan las presas inútiles
    apaticosP = [comprador for comprador in compradores if len(comprador[18]) <= juegosP]
    # en el caso en que haya demasiados inutiles, no nos los podemos cargar a todos 
    apaticosP = sorted(apaticosP, key=lambda comprador: comprador[0])
    
    for comprador in compradores[:]:
        if comprador in apaticosP:
            falsosconsumidores += 1
            compradores.remove(comprador)
            if len(apaticosP) <= numpadresP*2: break
        else: # ya que estamos depuro un error extraño que asigna [0.0] a algunos sitios
            comprador[16] = [0 if val == 0.0 or val == [0.0] else val for val in comprador[16]]
            comprador[18] = [0 if val == 0.0 or val == [0.0] else val for val in comprador[18]]
    print('quedan', len(compradores), 'compradores, de los que', len(apaticosP), 'no saben lo que quieren')
    
    valoresP = [len(sublistaP[18]) for sublistaP in compradores if sublistaP[18]]
    if valoresP:
        mediaP = sum(valoresP) / len(valoresP)
        #print("La media de las experiencias en compradores:", mediaP)
    else:
        print("No hay valores para calcular la media de experiencias para los consumidores")
    
    # LOS VENDEDORES ACTIVOS DESARROLLAN SUS REDES NEURONALES CON LAS EXPERIENCIAS DEL ANALISIS DEL MERCADO       
    # deben prever cuales de las variabilidades sobre angulos y distancias son mas exitosas
    for vendedor in vendedores[:]:
        if vendedor[40] == None and len(vendedor[18]) > 10: # esto selecciona a los nuevos con experiencia
            vendedor, tiemporeaccionD, utilizadasD = neuralD(vendedor)
            vendedor[14].append(utilizadasD)
            # tiempo para realizar el aprendizaje en cada encuesta y analisis
            vendedor[39].append(tiemporeaccionD)
            #print('cachorro', vendedor[0] ,'aprende de', len(vendedor[18]) ,'experiencias')
        else: # reubicamos a todos los que van a seguir en el ecosistema
            vendedor[1] = np.random.uniform(x_min,x_max)
            vendedor[2] = np.random.uniform(y_min,y_max) 
            
    # LOS CONSUMIDORES INTERESADOS DESARROLLAN SUS REDES NEURONALES CON LAS EXPERIENCIAS DE LOS AMIGOS
    # la distribucion de estrategias ha sido gaussiana, pero su seleccion de exitos no
    # segun los atributos, numprocesadores y experiencias, cada una tendrá una red distinta
    for comprador in compradores[:]:
        if comprador[40] == None and len(comprador[18]) > 10: # solo calculamos la red para los jovenes que juegan
            comprador, tiemporeaccionP, utilizadasP = neuralP(comprador)
            comprador[14].append(utilizadasP)
            # tiempo para realizar el aprendizaje
            comprador[39].append(tiemporeaccionP)
            #print('cria', comprador[0] ,' aprende de', len(comprador[18]),'experiencias')
        else: # controlaremos que no se hayan escapado a tiendas fuera del mercado
            comprador[1] = np.random.uniform(x_min,x_max)
            comprador[2] = np.random.uniform(y_min,y_max)        
    
    # COMIENZA LA compraventa DONDE SE SELECCIONARAN LAS MEJORES DECISIONES (conjunto de test)  
    # los bichos ya son redes neuronales individuales y distintas a partir de similares experiencias
    # pero los distintos atributos, configuraciones de la red y eficiencia en tiempo, las hace diversas en opciones de supervivencia      
    print('salen de compras', len(compradores), 'y hay', len(vendedores), 'propuestas en tiendas o paginas web')
    # recuento de las experiencias de todos los vendedores y compradores
    #segundamano = 0
    comprasventas = 0 # de compradores y vendedores
    nocompraventa = 0 # de compradores y vendedores
    paseo = 0
    # inicializacion solo por si en los primeros momentos o por falta de buenos vendedores o compradores, no se pasara el bucle de compraventa
    #DPsegundamano = 0 # recuento por cada vendedor
    DPcomprasventas = 0 # solo de compradores
    DPnocompraventa = 0 # solo de compradores
    DPpaseo = 0
    for tiempo in range(1, pasostiempo, 1):
        # en cada paso de tiempo las compradores pasean y van cobrando sus salarios y pensiones
        for P, comprador in enumerate(compradores):
            comprador[8] += ratioCV / pasostiempo
            
        if len(compradores) >= numpadresP + 2:# Verificamos si hay suficientes compradores para continuar con el proceso de compraventa
            for P, comprador in enumerate(compradores):            
                DPcomprasventas = 0 # quien dice tiendas, dice paginas web
                DPnocompraventa = 0 # recuento de las visitas a tiendas por cada comprador
                DPpaseo = 0 # busca tiendas o navega por internet
                if comprador[40] != None: # siempre que sea un comprador activo, con criterio, oportuno e interesado
                    # si es falsocomprador, sin criterio, no va de tiendas, pero corre el riesgo de ser engañado por algun vendedor si se lo encuentra
                    distanciaDP = []  # Lista para las distancias de cada comprador a todas las tiendas
                    for D, vendedor in enumerate(vendedores):
                        DPP = sqrt((vendedor[1] - comprador[1])**2 + (vendedor[2] - comprador[2])**2)
                        distanciaDP.append(DPP)
                    
                    if distanciaDP: # para controlar generaciones sin intentos de compraventa
                        DPmin = np.argmin(np.array(distanciaDP))  # Índice de la tienda mas cercana
                            
                        if distanciaDP[DPmin] <= 0.1:  # SI NO, con acciones de marketing PUEDE HABERLA ALCANZADO
                            # si el comprador valora mas algun atributo que el vendedor, lo considera una propuesta interesante
                            # por ejemplo la comprador quiere pantalla grande y el vendedor ofrece su modelo al mismo precio que pantalla pequeña
                            # esto se puede detallar mucho, pero para el objetivo, no importa

                            # inicializamos tiempos unitarios en 39 con datos del 22 (en compras ficticias, es decir, consultas a los demas incluidos publicidad y medios)
                            # con la experiencia en comprar o no ficticiamente, se van tomando decisiones
                            # si uno de los dos duda y tarda más en decidir, no hay compraventa
                            if comprador[39]:
                                comprador[22] = st.mean(comprador[39])
                            if vendedores[DPmin][39]:
                                vendedores[DPmin][22] = st.mean(vendedores[DPmin][39])
                            
                            vendedores[DPmin] , comprador , exito = compraventa(vendedores[DPmin] , comprador)
                            
                            if exito == True:
                                comprasventas += 1
                                DPcomprasventas += 1
                                vendedores[DPmin][6].append(1)
                                comprador[6].append(1)
                                compradoresquecompranP.append(comprador)
                                print('se añade el comprador',comprador[0])
                                vendedoresquevendenD.append(vendedores[DPmin])
                                print('se añade el vendedor',vendedores[DPmin])

                            else: # se va de la tienda
                                nocompraventa += 1
                                DPnocompraventa += 1
                                vendedores[DPmin][6].append(0)
                                comprador[6].append(0)                 
                                
                                comprador[1] = np.random.uniform(x_min,x_max)
                                comprador[2] = np.random.uniform(y_min,y_max)
                                                        
                        # equivale al set de test tanto para compradores como para vendedores:
                        else:    # comprador busca una tienda, los vendedores hacen publicidad, es decir, ambos se buscan
                            if vendedores[DPmin][40] != None:
                                paseo += 1                             
                                DPpaseo += 1 # el comprador pasea en escaparates y reseñas, pero el vendedor "pasea" con marketing
        
                                # Llamar a la función decisionD con la información del vendedor
                                # el vendedor tiene una necesidad (velocidadD) y una oferta mas o menos clara de lo que el comprador quiere (angulo)
                                alphaD, carreraD, tiempoD = decisionD(vendedores[DPmin])
                                
                                vendedores[DPmin][39].append(tiempoD) # guardamos la eficiencia de la decision
                                vendedores[DPmin][10].append([carreraD,alphaD])
                                # enfilada la proxima comprador, va a por ella con un margen estocástico
                                vendedores[DPmin][1] = vendedores[DPmin][1] + carreraD * cos(radians(alphaD))
                                vendedores[DPmin][2] = vendedores[DPmin][2] + carreraD * sin(radians(alphaD))
                                # la comprador escapa pero va mas lenta
                                # corre en sentido contrario también con variabilidad (se le suma cuando los 180º)
                                
                                # Llamar a la función decisionP con la información de la comprador
                                alphaP, carreraP, tiempoP = decisionP(comprador)
                                
                                comprador[39].append(tiempoP) # guardamos la eficiencia de la decision 
                                comprador[10].append([carreraP,alphaP])
        
                                if vendedores[DPmin][8] > comprador[8]: # el vendedor cree en su valor e invierte en marketing 
                                    vendedores[DPmin][1]= vendedores[DPmin][1] + carreraD * cos(alphaD)
                                    vendedores[DPmin][2]= vendedores[DPmin][2] + carreraD * sin(alphaD)
                                    
                                    comprador[8] += ratioCV / pasostiempo # el comprador recibe la publicidad
                                    vendedores[DPmin][8] -= ratioCV / pasostiempo # el vendedor invierte en el comprador
                                
                                else:# el vendedor no tiene buenas perspectivas de venta ante ese comprador, por lo que no invierte
                                    comprador[8] -= ratioCV / pasostiempo
                                    vendedores[DPmin][8] += ratioCV / pasostiempo
                            
                                    comprador[1]= comprador[1] - carreraP * cos(alphaP)
                                    comprador[2]= comprador[2] - carreraP * sin(alphaP)
                            else:         # el vendedor ha abierto tienda sin encomendarse a nadie                        
                                comprador[8] -= ratioCV / pasostiempo
                                vendedores[DPmin][8] += ratioCV / pasostiempo
            
                                comprador[1]= comprador[1] - carreraP * cos(alphaP)
                                comprador[2]= comprador[2] - carreraP * sin(alphaP)
                                    
                    controlfrontera(vendedores[DPmin],comprador) # si se ha largado de la tienda, ha perdido su oportunidad
                # si tiene criterio, puede haberse quedado sin pasta
                else: # no tiene criterio
                    if len(compradores) >= numhijosP + 2: # +2 para evitar que los gemelos se lo salten
                        if comprador[8] <= 0:                         
                            compradores.remove(comprador) # sin criterio pero con pasta, pasa a la siguiente generacion, pero sin pasta, no
                    else: break   # nunca habra menos de numpadresP
         
    print('comprobamos que quedan',len(compradoresquecompranP), 'compradores y', len(vendedoresquevendenD), 'vendedores')
    # certificamos la muerte de los no convencidos por algun vendedor   
    compradores = [comprador for comprador in compradores if comprador in compradoresquecompranP]
    # certificamos la defuncion de los que no compraventan lo suficiente o han enfermado
    vendedores = [vendedor for vendedor in vendedores if vendedor in vendedoresquevendenD]    
    print('tambien comprobamos que quedan',len(compradores), 'compradores y', len(vendedores), 'vendedores')    
    # GRAFICO (llama a la funcion dibujar) PARA CADA PASO DE CADA GENERACION 
    #dibujarcompraventa(vendedores,compradores,generacion,tiempo)    # devuelve grafico png

    print('compraventa', comprasventas, 'se van a otra tienda', nocompraventa, 'paseos', paseo)
    print('antes de evaluarlos, quedan', len(vendedores), 'vendedores y', len(compradores), 'compradores')
    
    # TOCA REPRODUCIRSE Y RESTITUIR EL NUMERO DE falsosconsumidores + compradoresquecompranP con numhijosP, 
    # LOS MEJORES REPRODUCTORES HAN SOBREVIVIDO, COMO MIN numpadresD y numpadresP   
    # puede suceder que en algunas generaciones no haya renovacion suficiente,
    # el programa sugiere la necesidad de que los malos vendedores y clientes salgan del mercado
    # si no podriamos estancarnos en mesetas poblacionales 
    while len(vendedores) >= numpadresD + 2:
        valor_minimoD = min(vendedor[8] for vendedor in vendedores)
        if valor_minimoD <= 1: # hasta suficientemente financiados
            vendedormin = next(vendedor for vendedor in vendedores if vendedor[8] == valor_minimoD)
            dep.append(vendedormin[0])
            vendedoresfuerademercado += 1
            vendedores.remove(vendedormin) # cierra la tienda y quiebra
            #print('vendedor', vendedormin[0] ,'se muere de hambre')
        else: break
    
    while len(compradores) >= numpadresP + (2*ratioCV):
        compradoresactivos = [comprador for comprador in compradores if comprador[40] is not None]  # Filtrar las compradores activas
        valor_minimoP = min(comprador[8] for comprador in compradoresactivos)   
        if valor_minimoP <= ratioCV: # con un minimo de presupuesto
            # Filtrar la comprador que no tiene capacidad de gasto, y pasan a ser inactivos
            compradormin = next(comprador for comprador in compradoresactivos if comprador[8] == valor_minimoP)
            compradormin[40] = None # a diferencia del vendedor en quiebra, el comprador solo se queda sin criterio y espera a la siguiente generacion
            compradoresfuerademercado += 1
            #print('comprador', compradormin[0] ,'sale del mercado')
        else: break
            
    print('han salido del mercado', vendedoresfuerademercado, 'vendedores y', compradoresfuerademercado, 'compradores')
    
    beneficioP = [comprador[8] for comprador in compradores]
    MbeneficioP = st.mean(beneficioP) 
    beneficioD = [vendedor[8] for vendedor in vendedores]
    MbeneficioD = st.mean(beneficioD)
    print('beneficio de los vendedores', MbeneficioD, 'y gasto de los compradores', MbeneficioP)

    exitosP = [len(comprador[6]) for comprador in compradores]
    MexitosP = st.mean(exitosP)
    exitosD = [len(vendedor[6]) for vendedor in vendedores]
    MexitosD = st.mean(exitosD)
    print('ventas de los vendedores', MexitosD, 'compras de las compradores', MexitosP)
    
    tiempoprocesoP = [comprador[22] for comprador in compradores]
    MtiempoprocesoP = st.mean(tiempoprocesoP)
    tiempoprocesoD = [vendedor[22] for vendedor in vendedores]
    MtiempoprocesoD = st.mean(tiempoprocesoD)
    print('tiempo de decision de los vendedores', MtiempoprocesoD, 'tiempo de decision de los compradores', MtiempoprocesoP)
    
    P23 = [comprador[23] for comprador in compradores]
    M23P = st.mean(P23)
    D23 = [vendedor[23] for vendedor in vendedores]
    M23D = st.mean(D23)
    
    P24 = [comprador[24] for comprador in compradores]
    M24P = st.mean(P24)
    D24 = [vendedor[24] for vendedor in vendedores]
    M24D = st.mean(D24)
    
    P25 = [comprador[25] for comprador in compradores]
    M25P = st.mean(P25)
    D25 = [vendedor[25] for vendedor in vendedores]
    M25D = st.mean(D25)
    
    P26 = [comprador[26] for comprador in compradores]
    M26P = st.mean(P26)
    D26 = [vendedor[26] for vendedor in vendedores]
    M26D = st.mean(D26)
    
    P27 = [comprador[27] for comprador in compradores]
    M27P = st.mean(P27)
    D27 = [vendedor[27] for vendedor in vendedores]
    M27D = st.mean(D27)
    
    P28 = [comprador[28] for comprador in compradores]
    M28P = st.mean(P28)
    D28 = [vendedor[28] for vendedor in vendedores]
    M28D = st.mean(D28)
    
    P29 = [comprador[29] for comprador in compradores]
    M29P = st.mean(P29)
    D29 = [vendedor[29] for vendedor in vendedores]
    M29D = st.mean(D29)
    
    P30 = [comprador[30] for comprador in compradores]
    M30P = st.mean(P30)
    D30 = [vendedor[30] for vendedor in vendedores]
    M30D = st.mean(D30)
    
    P31 = [comprador[31] for comprador in compradores]
    M31P = st.mean(P31)
    D31 = [vendedor[31] for vendedor in vendedores]
    M31D = st.mean(D31)
    
    P32 = [comprador[32] for comprador in compradores]
    M32P = st.mean(P32)
    D32 = [vendedor[32] for vendedor in vendedores]
    M32D = st.mean(D32)
    
    P33 = [comprador[33] for comprador in compradores]
    M33P = st.mean(P33)
    D33 = [vendedor[33] for vendedor in vendedores]
    M33D = st.mean(D33)
    
    P34 = [comprador[34] for comprador in compradores]
    M34P = st.mean(P34)
    D34 = [vendedor[34] for vendedor in vendedores]
    M34D = st.mean(D34)
    
    P35 = [comprador[35] for comprador in compradores]
    M35P = st.mean(P35)
    D35 = [vendedor[35] for vendedor in vendedores]
    M35D = st.mean(D35)
    
    P36 = [comprador[36] for comprador in compradores]
    M36P = st.mean(P36)
    D36 = [vendedor[36] for vendedor in vendedores]
    M36D = st.mean(D36)
    
    P37 = [comprador[37] for comprador in compradores]
    M37P = st.mean(P37)
    D37 = [vendedor[37] for vendedor in vendedores]
    M37D = st.mean(D37)
    
    P38 = [comprador[38] for comprador in compradores]
    M38P = st.mean(P38)
    D38 = [vendedor[38] for vendedor in vendedores]
    M38D = st.mean(D38)
    
    # compradores CORTEJAN CON compradores Y vendedores CON vendedores
    #print('comienza la fase reproductiva')
    # comprador BUSCA comprador (seguimos en la misma generacion, pero entran en celo)
    SVP = len(compradores)   # compradores supervivientes, mayor o igual que numpadresP
    compradoreshijos = []           
    numcompradoreshijos = 0
    identificadorD = numeroD*(generacion+1)
    
    # vendedor BUSCA vendedor
    SVD = len(vendedores)   # vendedores supervivientes, mayor o igual que numpadresP
    vendedoreshijos = []
    numvendedoreshijos = 0
    identificadorP = numeroP*(generacion+1)    # para numerar a las compradores
    
    # puede que haya mas necesidad de reproduccion que la que es capaz de suministrar en el tiempo de la generacion
    # por ello pondremos un adaptador de los ciclos reproductivos que dependera de las crias y cachorros que caben
    promiscuidad = int((numeroP+numeroD)/(SVP+SVD))
    for tiempo in range(pasostiempo*promiscuidad):
        # bucle hasta que se haya restablecido el numeroP
        if numcompradoreshijos + SVP <= numeroP:# SVP mayor o igual que numpadresP
            # pasan de comer y buscan novios
            for padreP, comprador in enumerate(compradores):
                distanciaPP=[]
                for madreP, comprador in enumerate(compradores):
                    if padreP == madreP:
                        PMP=5 # no se corre el indice de los posteriores
                    else:
                        PMP=sqrt((compradores[padreP][1]-compradores[madreP][1])**2 + (compradores[padreP][2]-compradores[madreP][2])**2)
                        # de momento no son selectivos, pues han sobrevivido
                        distanciaPP.append(PMP) 
                        # se selecciona por distancia   
       
                if distanciaPP:
                    PPmin = np.argmin(np.array(distanciaPP)) # novia mas proxima
                
                    if distanciaPP[PPmin] <= 0.1:
                        compradores[padreP][7] += 1 # contador de eventos reproductivos
                        compradores[PPmin][7] += 1                   
                        numcompradoreshijos += 1
                        # para reproducirse van a gastar energía
                        compradores[padreP][8] -= 1 / pasostiempo 
                        compradores[PPmin][8] -= 1 / pasostiempo  
                        
                        # compradores que se reproducen  
                        atributosP_hijo1 = [] # inicializacion 
                        atributosP_hijo1 = hijo(compradores[padreP][13],compradores[PPmin][13])
                        # los valores de los inputs no se heredan, si su seleccion
                        
                        # ahora con el numero de procesadores de la capa oculta
                        Np = compradores[padreP][17]
                        Mp = compradores[PPmin][17]
                        # tamaño más próximo a la media de N y M
                        numprocesadores_hijo1 = int(np.round((Np + Mp) / 2))
    
                        atributos_hijo1 = []
                        exitostrainP = [] # lista vacia de dos valores con un numero variable de entradas
                        exitostestP = []
                        exitostestRP = []
                        tiemposP = [] # lista vacia de un valor con un numero variable de entradas
                      
                        identificadorP += 1
                        comprador_hijo1= [identificadorP,
                                     compradores[padreP][1], # posicion aleatoria de cada comprador
                                     compradores[PPmin][2],
                                     np.random.uniform(0,drmaxP), # angulo
                                     np.random.uniform(0,velocidadP), # velocidad
                                     [], # 5, juegos
                                     [], # 6, contador de eventos compraventa
                                     0, # 7, contador de eventos reproductivos           
                                     1, # 8, salario disponible
                                     0, # 9, antiguedad 
                                     exitostestP,
                                     exitostestRP, #11, exitos reproductivos
                                     exitostrainP, # 12, tabla de angulos y velocidades con exito (entrenamiento)
                                     atributosP_hijo1, # 13,lista de atributos seleccionados
                                     [], # 14, contador de entrenamientos
                                     0, # 15, contador de conexiones
                                     # los hijos heredan los valores y las configuraciones de los padres
                                     atributos_hijo1, # lista de los valores
                                     numprocesadores_hijo1,
                                     [],
                                     # los pesos no se heredan (de Lamarck a Darwin)
                                     [], # Inicializacion entre la entrada y la capa oculta (solo hereda el numero, no los pesos)           
                                     [], # Inicializacion dentro de la capa oculta
                                     [],# Inicialización de los pesos de las conexiones entre la capa oculta y la capa de salida
                                     0, # 22, tiempo medio de decision
                                     (compradores[padreP][23]+compradores[PPmin][23])/2, # 23, conectividad 5G, wifi, bluetooth,
                                     (compradores[padreP][24]+compradores[PPmin][24])/2, # 24, duracion de la bateria                  
                                     (compradores[padreP][25]+compradores[PPmin][25])/2, # 25, funciones inteligencia artificial
                                     (compradores[padreP][26]+compradores[PPmin][26])/2, # 26, ecocertificacion, sostenibilidad
                                     (compradores[padreP][27]+compradores[PPmin][27])/2, # 27, seguridad y privacidad
                                     (compradores[padreP][28]+compradores[PPmin][28])/2, # 28, camara y software de video
                                     (compradores[padreP][29]+compradores[PPmin][29])/2, # 29, memoria
                                     (compradores[padreP][30]+compradores[PPmin][30])/2, # 30, resolución y tamaño de pantalla
                                     (compradores[padreP][31]+compradores[PPmin][31])/2, # 31, realidad aumentada y realidad virtual
                                     (compradores[padreP][32]+compradores[PPmin][32])/2, # 32, integracion con otros sensores y dispositivos
                                     (compradores[padreP][33]+compradores[PPmin][33])/2, # 33, diseño y estética
                                     (compradores[padreP][34]+compradores[PPmin][34])/2, # 34, sensores integrados IoT 
                                     (compradores[padreP][35]+compradores[PPmin][35])/2, # 35, usabilidad y ergonomía
                                     (compradores[padreP][36]+compradores[PPmin][36])/2, # 36, sensores y funcionalidades de bienestar y salud
                                     (compradores[padreP][37]+compradores[PPmin][37])/2, # 37, funcionalidades adicionales
                                     (compradores[padreP][38]+compradores[PPmin][38])/2, # 38, accesorios disponibles
                                     tiemposP,
                                     None
                                     ]
                        
                        # MUTACION
                        atributosP_hijo1, numprocesadores_hijo1 = mutacion (comprador_hijo1[13],comprador_hijo1[17])
                        comprador_hijo1[13] = atributosP_hijo1
                        comprador_hijo1[17] = numprocesadores_hijo1
                        #print('la cria ha nacido con', len(comprador_hijo1[13]) ,'atributos y', comprador_hijo1[17] ,'procesadores')

                        for i in range(23,39):  # Campos entre 23 y 38 (inclusive)
                            if i not in atributosP_hijo1:
                                comprador_hijo1[i] = 0
                            atributos_hijo1.append(comprador_hijo1[i])
                                       
                        compradoreshijos.append(comprador_hijo1)

                        if compradores[PPmin][8] <= compradores[padreP][8] :
                            # si el padre está bien alimentado, puede reproducirse mejor
                            numcompradoreshijos += 1 # HERMANO GEMELO
                            compradores[padreP][8] -= 1 / pasostiempo # gasta energia
                            compradores[PPmin][8] -= 1 / pasostiempo  # gasta energia
    
                            atributosP_hijo2 = []
                            atributosP_hijo2 = hijo(compradores[padreP][13],compradores[PPmin][13])
                            
                            # ahora con el numero de procesadores de la capa oculta
                            Np = compradores[PPmin][17]
                            Mp = compradores[padreP][17]
                            # tamaño más próximo por arriba a la media de N y M                      
                            numprocesadores_hijo2= int(np.round((Np + Mp) / 2))
                            
                            atributos_hijo2=[]
                            exitostrainP=[] # lista vacia de dos valores con un numero variable de entradas
                            exitostestP=[]
                            exitostestRP=[]
                            tiemposP=[] # lista vacia de un valor con un numero variable de entradas
                            # identificar
                            identificadorP += 1
                            comprador_hijo2= [identificadorP,
                                         compradores[PPmin][1], # posicion aleatoria de cada comprador
                                         compradores[padreP][2],
                                         np.random.uniform(0,drmaxP), # angulo
                                         np.random.uniform(0,velocidadP), # velocidad
                                         [], # juegos
                                         [], # 6, contador de eventos compraventa
                                         0, # 7, contador de eventos reproductivos           
                                         1, # 8, salario disponible 
                                         0, # 9, antiguedad 
                                         exitostestP, #10, contador de exitos de huida
                                         exitostestRP, # 11, contador de exitos reproductivos
                                         exitostrainP, # 12, tabla de angulos y velocidades con exito
                                         atributosP_hijo2, # 13, tabla de angulos y velocidades que no han conseguido atrapar a la comprador (entrenamiento)
                                         [], # 14, contador de entrenamientos
                                         0, # 15, contador de conexiones
                                         # los hijos heredan los valores y las configuraciones de los padres
                                         atributos_hijo2, # valores y 0's
                                         numprocesadores_hijo2,
                                         [],
                                         # los pesos no se heredan (de Lamarck a Darwin)
                                         [], # pesos capa input-hidden
                                         [], # pesos capa recurrente hidden-hidden
                                         [], # pesos capa hidden-output
                                         0, # tiempo medio de decision
                                         (compradores[padreP][23]+compradores[PPmin][23])/2, # 23, conectividad 5G, wifi, bluetooth,
                                         (compradores[padreP][24]+compradores[PPmin][24])/2, # 24, duracion de la bateria                  
                                         (compradores[padreP][25]+compradores[PPmin][25])/2, # 25, funciones inteligencia artificial
                                         (compradores[padreP][26]+compradores[PPmin][26])/2, # 26, ecocertificacion, sostenibilidad
                                         (compradores[padreP][27]+compradores[PPmin][27])/2, # 27, seguridad y privacidad
                                         (compradores[padreP][28]+compradores[PPmin][28])/2, # 28, camara y software de video
                                         (compradores[padreP][29]+compradores[PPmin][29])/2, # 29, memoria
                                         (compradores[padreP][30]+compradores[PPmin][30])/2, # 30, resolución y tamaño de pantalla
                                         (compradores[padreP][31]+compradores[PPmin][31])/2, # 31, realidad aumentada y realidad virtual
                                         (compradores[padreP][32]+compradores[PPmin][32])/2, # 32, integracion con otros sensores y dispositivos
                                         (compradores[padreP][33]+compradores[PPmin][33])/2, # 33, diseño y estética
                                         (compradores[padreP][34]+compradores[PPmin][34])/2, # 34, sensores integrados IoT 
                                         (compradores[padreP][35]+compradores[PPmin][35])/2, # 35, usabilidad y ergonomía
                                         (compradores[padreP][36]+compradores[PPmin][36])/2, # 36, sensores y funcionalidades de bienestar y salud
                                         (compradores[padreP][37]+compradores[PPmin][37])/2, # 37, funcionalidades adicionales
                                         (compradores[padreP][38]+compradores[PPmin][38])/2, # 38, accesorios disponibles
                                         tiemposP,
                                         None
                                         ]
                            
                            # MUTACION
                            atributosP_hijo2, numprocesadores_hijo2 = mutacion (comprador_hijo2[13],comprador_hijo2[17])
                            
                            # ACTUALIZACION DE LA comprador PARA ENTRAR EN LA SIGUIENTE GENERACION
                            comprador_hijo2[13]=atributosP_hijo2
                            comprador_hijo2[17]=numprocesadores_hijo2
                            #print('el gemelo ha nacido con', len(comprador_hijo2[13]) ,'atributos y', comprador_hijo2[17] ,'procesadores')
                            
                            for i in range(23,39):  # Campos entre 23 y 38 (inclusive)
                                if i not in atributosP_hijo2:
                                    comprador_hijo2[i] = 0
                                atributos_hijo2.append(comprador_hijo2[i])
                                                       
                            compradoreshijos.append(comprador_hijo2)

                        # el padre se va a por tabaco
                        compradores[padreP][1] = np.random.uniform(x_min,x_max)
                        compradores[padreP][2] = np.random.uniform(y_min,y_max)
    
                    else: # no ha habido ligoteo y el padre busca a la madre
                        # PADRE CORTEJA, les va a costar energía 
                        compradores[padreP][8] = ratioCV / pasostiempo
                        compradores[PPmin][8] = ratioCV / pasostiempo
                        
                        dx = compradores[PPmin][1] - compradores[padreP][1]
                        dy = compradores[PPmin][2] - compradores[padreP][2]
                        alfa = degrees(atan2(dy,dx)) - (compradores[PPmin][3] - compradores[padreP][3]) # ambos estan rotando
                        # si se encuentran, ligan
                        alphaP = alfa * np.pi / 180
                        while alphaP > np.pi * 2:
                            alphaP=alphaP - np.pi * 2 # si hay mas de una vuelta de mas 
                        while alphaD < -np.pi * 2:
                            alphaD=alphaD + np.pi * 2 # si es al reves, tambien
                        
                        # enfilada la proxima comprador, va a por ella
                        compradores[padreP][1]=compradores[padreP][1] + compradores[padreP][4] * cos(radians(alphaP))
                        compradores[padreP][2]=compradores[padreP][2] + compradores[padreP][4] * sin(radians(alphaP))
                        # ella busca a quien le pretende 
                        compradores[PPmin][1]=compradores[PPmin][1] + compradores[PPmin][4] * cos(radians(alphaP+np.pi*2))
                        compradores[PPmin][2]=compradores[PPmin][2] + compradores[PPmin][4] * sin(radians(alphaP+np.pi*2))    
                        
                controlfrontera(compradores[padreP],compradores[PPmin]) # para no salirse del cuadro

        # bucle de reproduccion hasta restablecer el numeroD de vendedores
        if numvendedoreshijos + SVD <= numeroD:# SVD mayor o igual que numpadresD
            # LOS vendedores SALEN DE LIGUE    (solo experimentados)               
            for padreD,vendedor in enumerate(vendedores):
                distanciaDD = []
                for madreD,vendedor in enumerate(vendedores):
                    if padreD == madreD:
                        PMD = 5
                    else:    
                        PMD=sqrt((vendedores[padreD][1]-vendedores[madreD][1])**2 + (vendedores[padreD][2]-vendedores[madreD][2])**2)   
                        distanciaDD.append(PMD)      
               
                DDmin=np.argmin(np.array(distanciaDD)) # novia mas proxima
                
                if distanciaDD[DDmin] <= 0.1: # si se encuentran, se enamoran
                    vendedores[padreD][7] += 1 # contador de eventos reproductivos
                    vendedores[DDmin][7] += 1               
                    numvendedoreshijos += 1 # primer cachorro  
                    vendedores[padreD][8] -= 1 / pasostiempo # gasta energia en reproducirse
                    vendedores[DDmin][8] -= 1 / pasostiempo  # gasta energia
                    
                    # vendedores que se reproducen  
                    atributosD_hijo1 = []
                    atributosD_hijo1 = hijo(vendedores[padreD][13],vendedores[DDmin][13])
                    
                    # ahora con el numero de procesadores de la capa oculta
                    Np = vendedores[padreD][17]
                    Mp = vendedores[DDmin][17]
                    # tamaño más próximo por arriba a la media de N y M
                    numprocesadores_hijo1 = int(np.round((Np + Mp) / 2))

                    atributos_hijo1 = []
                    exitostrainD = [] # lista vacia de dos valores con un numero variable de entradas
                    exitostestD = []
                    exitostestRD = []
                    tiemposD = [] # lista vacia de un valor con un numero variable de entradas
                    
                    identificadorD += 1
                    vendedor_hijo1 = [identificadorD,
                                        vendedores[padreD][1],# posicion aleatoria de cada vendedor
                                        vendedores[DDmin][2],
                                        np.random.uniform(0,drmaxD),# angulo
                                        np.random.uniform(0,velocidadD),# velocidad
                                        [],# 5, juegos
                                        [],# 6, contador de intentosdecompraventa
                                        0,# 7, contador de eventos reproductivos                  
                                        ratioCV,# 8, fitness al nacer                  
                                        0, # 9, antiguedad
                                        exitostestD,
                                        exitostestRD,
                                        exitostrainD, # 12, tabla de angulos y velocidades con exito (entrenamiento)
                                        atributosD_hijo1, # 13, tabla de angulos y velocidades que no han conseguido atrapar a la comprador (entrenamiento)
                                        [], # 14, contador de entrenamientos
                                        0, # 15, contador de conexiones
                                        atributos_hijo1,
                                        numprocesadores_hijo1,
                                        [],
                                        # los pesos no se heredan (de Lamarck a Darwin)
                                        [],
                                        [],
                                        [],
                                        0, # tiempo de decision
                                        (vendedores[padreD][23]+vendedores[DDmin][23])/2, # 23, conectividad 5G, wifi, bluetooth,
                                        (vendedores[padreD][24]+vendedores[DDmin][24])/2, # 24, duracion de la bateria                  
                                        (vendedores[padreD][25]+vendedores[DDmin][25])/2, # 25, funciones inteligencia artificial
                                        (vendedores[padreD][26]+vendedores[DDmin][26])/2, # 26, ecocertificacion, sostenibilidad
                                        (vendedores[padreD][27]+vendedores[DDmin][27])/2, # 27, seguridad y privacidad
                                        (vendedores[padreD][28]+vendedores[DDmin][28])/2, # 28, camara y software de video
                                        (vendedores[padreD][29]+vendedores[DDmin][29])/2, # 29, memoria
                                        (vendedores[padreD][30]+vendedores[DDmin][30])/2, # 30, resolución y tamaño de pantalla
                                        (vendedores[padreD][31]+vendedores[DDmin][31])/2, # 31, realidad aumentada y realidad virtual
                                        (vendedores[padreD][32]+vendedores[DDmin][32])/2, # 32, integracion con otros sensores y dispositivos
                                        (vendedores[padreD][33]+vendedores[DDmin][33])/2, # 33, diseño y estética
                                        (vendedores[padreD][34]+vendedores[DDmin][34])/2, # 34, sensores integrados IoT 
                                        (vendedores[padreD][35]+vendedores[DDmin][35])/2, # 35, usabilidad y ergonomía
                                        (vendedores[padreD][36]+vendedores[DDmin][36])/2, # 36, sensores y funcionalidades de bienestar y salud
                                        (vendedores[padreD][37]+vendedores[DDmin][37])/2, # 37, funcionalidades adicionales
                                        (vendedores[padreD][38]+vendedores[DDmin][38])/2, # 38, accesorios disponibles
                                        tiemposD,
                                        None                                            
                                        ]
                    
                    # MUTACION
                    atributosD_hijo1, numprocesadores_hijo1 = mutacion (vendedor_hijo1[13],vendedor_hijo1[17])

                    # ACTUALIZACION PARA ENTRAR EN LA SIGUIENTE GENERACION
                    vendedor_hijo1[13] = atributosD_hijo1
                    vendedor_hijo1[17] = numprocesadores_hijo1
                    #print('el cachorro ha nacido con', len(vendedor_hijo1[13]) ,'atributos y', vendedor_hijo1[17] ,'procesadores')
                    
                    for i in range(23,39):  # Campos entre 23 y 38(inclusive)
                        if i not in atributosD_hijo1:
                            vendedor_hijo1[i] = 0
                        atributos_hijo1.append(vendedor_hijo1[i])
 
                    # añadimos el nuevo vendedor                                                        
                    vendedoreshijos.append(vendedor_hijo1)

                    # si está bien alimentado, puede permitirse un segundo cachorro    
                    if vendedores[DDmin][8] <= vendedores[padreD][8] : 
                        # Eliminar valores repetidos en ambos padres
                        numvendedoreshijos += 1 # 2 cachorros por generacion
                        vendedores[padreD][8] -= 1 / pasostiempo # gasta energia en reproducirse
                        vendedores[DDmin][8] -= 1 / pasostiempo  # gasta energia
                        
                        atributosD_hijo2 = []
                        atributosD_hijo2 = hijo(vendedores[padreD][13],vendedores[DDmin][13])
                        
                        # ahora con el numero de procesadores de la capa oculta
                        Np = vendedores[DDmin][17]
                        Mp = vendedores[padreD][17]
                        # tamaño más próximo por arriba a la media de N y M
                        numprocesadores_hijo2 = int(np.round((Np + Mp) / 2))  

                        atributos_hijo2 = []
                        exitostrainD = [] # lista vacia de dos valores con un numero variable de entradas
                        exitostestD = []
                        exitostestRD = []
                        tiemposD = [] # lista vacia de un valor con un numero variable de entradas
                        # identificar
                        identificadorD += 1
                        vendedor_hijo2 = [identificadorD,
                                            vendedores[DDmin][1],# posicion aleatoria de cada vendedor
                                            vendedores[padreD][2],
                                            np.random.uniform(0,drmaxD),# angulo
                                            np.random.uniform(0,velocidadD),# velocidad
                                            [],# juegos
                                            [], # 6, contador de intentosdecompraventa
                                            0, # 7, contador de eventos reproductivos                  
                                            ratioCV, # 8, fitness al nacer                  
                                            0, # 9, antiguedad 
                                            exitostestD,
                                            exitostestRD,
                                            exitostrainD, # 12, tabla de angulos y velocidades con exito (entrenamiento)
                                            atributosD_hijo2, # 13, tabla de angulos y velocidades que no han conseguido atrapar a la comprador (entrenamiento)
                                            [], # 14, contador de entrenamientos
                                            0, # 15, contador de conexiones
                                            atributos_hijo2,
                                            numprocesadores_hijo2,
                                            [],
                                            [],
                                            [],
                                            [],
                                            0, # tiempo de decision
                                            (vendedores[padreD][23]+vendedores[DDmin][23])/2, # 23, conectividad 5G, wifi, bluetooth,
                                            (vendedores[padreD][24]+vendedores[DDmin][24])/2, # 24, duracion de la bateria                  
                                            (vendedores[padreD][25]+vendedores[DDmin][25])/2, # 25, funciones inteligencia artificial
                                            (vendedores[padreD][26]+vendedores[DDmin][26])/2, # 26, ecocertificacion, sostenibilidad
                                            (vendedores[padreD][27]+vendedores[DDmin][27])/2, # 27, seguridad y privacidad
                                            (vendedores[padreD][28]+vendedores[DDmin][28])/2, # 28, camara y software de video
                                            (vendedores[padreD][29]+vendedores[DDmin][29])/2, # 29, memoria
                                            (vendedores[padreD][30]+vendedores[DDmin][30])/2, # 30, resolución y tamaño de pantalla
                                            (vendedores[padreD][31]+vendedores[DDmin][31])/2, # 31, realidad aumentada y realidad virtual
                                            (vendedores[padreD][32]+vendedores[DDmin][32])/2, # 32, integracion con otros sensores y dispositivos
                                            (vendedores[padreD][33]+vendedores[DDmin][33])/2, # 33, diseño y estética
                                            (vendedores[padreD][34]+vendedores[DDmin][34])/2, # 34, sensores integrados IoT 
                                            (vendedores[padreD][35]+vendedores[DDmin][35])/2, # 35, usabilidad y ergonomía
                                            (vendedores[padreD][36]+vendedores[DDmin][36])/2, # 36, sensores y funcionalidades de bienestar y salud
                                            (vendedores[padreD][37]+vendedores[DDmin][37])/2, # 37, funcionalidades adicionales
                                            (vendedores[padreD][38]+vendedores[DDmin][38])/2, # 38, accesorios disponibles
                                            tiemposD,
                                            None
                                            ]
                        
                        # MUTACION
                        atributosD_hijo2, numprocesadores_hijo2 = mutacion (vendedor_hijo2[13],vendedor_hijo2[17])
                        
                        # ACTUALIZACION DE LA comprador PARA ENTRAR EN LA SIGUIENTE GENERACION
                        vendedor_hijo2[13] = atributosD_hijo2
                        vendedor_hijo2[17] = numprocesadores_hijo2
                        #print('el gemelo ha nacido con', len(vendedor_hijo2[13]) ,'atributos y', vendedor_hijo2[17] ,'procesadores')
                        
                        for i in range(23,39):  # Campos entre 23 y 38 (inclusive)
                            if i not in atributosD_hijo2:
                                vendedor_hijo2[i] = 0
                            atributos_hijo2.append(vendedor_hijo2[i])
                        
                        # añadimos el hermano gemelo                                                        
                        vendedoreshijos.append(vendedor_hijo2)  

                    # el padre se va a por tabaco
                    vendedores[padreD][1] = np.random.uniform(x_min,x_max)
                    vendedores[padreD][2] = np.random.uniform(y_min,y_max)
                    
                else:
                    # PADRE PERSIGUE
                    vendedores[DDmin][8] = 1 / pasostiempo
                    vendedores[padreD][8] = 1 / pasostiempo
                    
                    dx = vendedores[DDmin][1] - vendedores[padreD][1]
                    dy = vendedores[DDmin][2] - vendedores[padreD][2]
                    alfa = degrees(atan2(dy,dx)) - (vendedores[DDmin][3] - vendedores[padreD][3]) # ambos estan rotando
                    # si se encuentran, ligan
                    alphaD = alfa * np.pi / 180
                    while alphaD > np.pi * 2:
                        alphaD = alphaD - np.pi * 2 
                    while alphaD < -np.pi * 2:
                        alphaD= alphaD + np.pi * 2 # el paso de tiempo es 1, así que la la distancia de la carrera es la velocidadD=vendedor[4]
                    
                    # enfilada la proxima novia, va a por ella
                    vendedores[padreD][1] = vendedores[padreD][1] + vendedores[padreD][4]*cos(radians(alphaD))
                    vendedores[padreD][2] = vendedores[padreD][2] + vendedores[padreD][4]*sin(radians(alphaD))

                    vendedores[madreD][1] = vendedores[madreD][1] + vendedores[madreD][4]*cos(radians(alphaD+np.pi*2))
                    vendedores[madreD][2] = vendedores[madreD][2] + vendedores[madreD][4]*sin(radians(alphaD+np.pi*2))
                    
                controlfrontera(vendedores[padreD],vendedores[DDmin])
        
        # GRAFICO (llama a la funcion dibujar) PARA CADA PASO  
        #dibujarreproduccion(vendedores,vendedoreshijos,compradores,compradoreshijos,generacion,tiempo,pasostiempo)
            # devuelve grafico png
    
        # al final del celo los hijos se independizan de los padres y se esparcen
        for P,comprador in enumerate(compradoreshijos):
            comprador[1] = np.random.uniform(x_min,x_max)# posicion aleatoria de cada comprador           
            comprador[2] = np.random.uniform(y_min,y_max) 
        for D,vendedor in enumerate(vendedoreshijos):
            vendedor[1] = np.random.uniform(x_min,x_max)# posicion aleatoria de cada comprador           
            vendedor[2] = np.random.uniform(y_min,y_max) 

    reproduccionP = [comprador[7] for comprador in compradores]
    MreproduccionP = st.mean(reproduccionP)
    reproduccionD = [vendedor[7] for vendedor in vendedores]
    MreproduccionD = st.mean(reproduccionD)

    # de los compradores y vendedores que han tomado alguna decision (las habrá que 
    # pasan sin pena ni gloria), nos interesa la media de tiempo en decidir que
    # cada comprador y vendedor experimentado gasta en cada generacion
    for P,comprador in enumerate(compradores):
        comprador[9] += 1 # cumpleaños
        if compradores[P][39]: # si no esta vacia (las crias la tienen vacia)
            compradores[P][22] = st.mean(comprador[39]) # media de millonesimas sg en decidir
        # para evitar errores en la reproduccion, que a veces cuela el 39
        compradores[P][13] = [valor for valor in comprador[13] if 23 <= valor <= 38]

    for P in comprador[13]:
        if comprador[16][P-23] == 0:
            comprador[16][P-23] = 0.01            

    for D,vendedor in enumerate(vendedores):
        vendedor[9] += 1
        if vendedores[D][39]: # si es un hijo, todavia no tiene experiencia
            vendedores[D][22] = st.mean (vendedor[39])  # media para decidir en millonesimas  
        vendedores[D][13] = [valor for valor in vendedor[13] if 23 <= valor <= 38] 
    
    for D in vendedor[13]:
        if vendedor[16][D - 23] == 0:
            vendedor[16][D - 23] = 0.01
        
    for P,comprador in enumerate(compradores):
        comprador[39] = []
        resumengenP.append(lista_a_diccionario([generacion, comprador[0], len(comprador[6]), comprador[8], comprador[7], comprador[9], len(comprador[13]), len(comprador[14]), comprador[15], comprador[17], comprador[22], comprador[23], comprador[24], comprador[25], comprador[26], comprador[27], comprador[28], comprador[29], comprador[30], comprador[31], comprador[32], comprador[33], comprador[34], comprador[35], comprador[36], comprador[37], comprador[38]]))

    #etiquetas = ['jornada', 'identificador', 'transacciones', 'beneficios', 'nuevos agentes', 'antiguedad', 'atributos', 'entrenamientos', 'modularidad', 'procesadores', 'tiempo de proceso', 'conectividad 5G, wifi, bluetooth, radiofrecuencia','duracion de la bateria','funciones inteligencia artificial','ecocertificacion, sostenibilidad','seguridad y privacidad','camara y software de video','memoria','resolución y tamaño de pantalla','realidad aumentada y realidad virtual','integracion con otros sensores y dispositivos','diseño y estética','sensores integrados IoT','usabilidad y ergonomia','sensores y funcionalidades de bienestar y salud','otras funcionalidades adicionales','accesorios disponibles']

    for D, vendedor in enumerate(vendedores):
        vendedor[39] = [] # no necesitamos conservar cada uno de los tiempos de decision, sino su media              
        resumengenD.append(lista_a_diccionario([generacion, vendedor[0], len(vendedor[6]), vendedor[8], vendedor[7], vendedor[9], len(vendedor[13]), len(vendedor[14]), vendedor[15], vendedor[17], vendedor[22], vendedor[23], vendedor[24], vendedor[25], vendedor[26], vendedor[27], vendedor[28], vendedor[29], vendedor[30], vendedor[31], vendedor[32], vendedor[33], vendedor[34], vendedor[35], vendedor[36], vendedor[37], vendedor[38]]))

    print('generacion',generacion)
    print('vendedores',len(vendedores))
    print('nuevos vendedores', len(vendedoreshijos))
    print('compradores',len(compradores))
    print('nuevos consumidores', len(compradoreshijos))
    
    #seguimos dentro de una generacion: experimentados   
    evoluciondelmercado.append([generacion,len(compradores),falsosconsumidores,numcompradoreshijos,len(compradoresquecompranP),compradoresfuerademercado,len(vendedores),falsosvendedores,numvendedoreshijos,len(dep),vendedoresfuerademercado,comprasventas,comprasventas,nocompraventa,paseo,MtiempoprocesoP,MexitosP,MbeneficioP,MreproduccionP,M23P,M24P,M25P,M26P,M27P,M28P,M29P,M30P,M31P,M32P,M33P,M34P,M35P,M36P,M37P,M38P,MtiempoprocesoD,MexitosD,MbeneficioD,MreproduccionD,M23D,M24D,M25D,M26D,M27D,M28D,M29D,M30D,M31D,M32D,M33D,M34D,M35D,M36D,M37D,M38D])  
    # al final de la generacion
    compradores.extend(compradoreshijos) # al acabar el celo y el aprendizaje, todos son iguales
    vendedores.extend(vendedoreshijos)

    #print('comenzaremos la generacion siguiente con',len(vendedores), 'vendedores y',len(compradores), 'compradores')
    
    guardar_listas_generacion(generacion, vendedores, compradores, evoluciondelmercado, resumengenD, resumengenP, folder_path)
    
# vector de intereses-atributos-argumentos-valores modales de la ultima generacion
print('argumentos de los vendedores', contarno0D(vendedores))    
print('razones de las compradores', contarno0P(compradores)) 

# hemos ido guardando en una carpeta los 3 ficheros siguientes, que vamos a graficar   
for gen in range(1,generaciones):
    vendedores, compradores, evoluciondelmercado, resumengenD, resumengenP = cargar_listas_generacion(gen, folder_path)

generacion = [item[0] for item in evoluciondelmercado]
millonesdecompradores = [item[1] for item in evoluciondelmercado]
compradoressatisfechos = [item[4] for item in evoluciondelmercado]
falsoscompradores = [item[5] for item in evoluciondelmercado]
nuevoscompradores = [item[3] for item in evoluciondelmercado]                  
compradoressalidosdelmercado = [item[2] for item in evoluciondelmercado]
numerovendedores = [item[6] for item in evoluciondelmercado]
vendedoresenquiebra = [item[9] for item in evoluciondelmercado]
vendedoresfracasados = [item[10] for item in evoluciondelmercado]
nuevosvendedores = [item[8] for item in evoluciondelmercado]
vendedoressalidosdelmercado = [item[7] for item in evoluciondelmercado]

segundamano = [item[11] for item in evoluciondelmercado]
comprasventas = [item[12] for item in evoluciondelmercado]
nocompraventa = [item[13] for item in evoluciondelmercado]
paseo = [item[14] for item in evoluciondelmercado]

tiempodeprocesoP = [item[15] for item in evoluciondelmercado]
exitosdeP = [item[16] for item in evoluciondelmercado]
beneficiodeP = [item[17] for item in evoluciondelmercado]
reproducciondeP = [item[18] for item in evoluciondelmercado]
MPde23 = [item[19] for item in evoluciondelmercado]
MPde24 = [item[20] for item in evoluciondelmercado]
MPde25 = [item[21] for item in evoluciondelmercado]
MPde26 = [item[22] for item in evoluciondelmercado]
MPde27 = [item[23] for item in evoluciondelmercado]
MPde28 = [item[24] for item in evoluciondelmercado]
MPde29 = [item[25] for item in evoluciondelmercado]
MPde30 = [item[26] for item in evoluciondelmercado]
MPde31 = [item[27] for item in evoluciondelmercado]
MPde32 = [item[28] for item in evoluciondelmercado]
MPde33 = [item[29] for item in evoluciondelmercado]
MPde34 = [item[30] for item in evoluciondelmercado]
MPde35 = [item[31] for item in evoluciondelmercado]
MPde36 = [item[32] for item in evoluciondelmercado]
MPde37 = [item[33] for item in evoluciondelmercado]
MPde38 = [item[34] for item in evoluciondelmercado]
tiempodeprocesoD = [item[35] for item in evoluciondelmercado]
exitosdeD = [item[36] for item in evoluciondelmercado]
beneficiodeD = [item[37] for item in evoluciondelmercado]
reproducciondeD = [item[38] for item in evoluciondelmercado]
MDde23 = [item[39] for item in evoluciondelmercado]
MDde24 = [item[40] for item in evoluciondelmercado]
MDde25 = [item[41] for item in evoluciondelmercado]
MDde26 = [item[42] for item in evoluciondelmercado]
MDde27 = [item[43] for item in evoluciondelmercado]
MDde28 = [item[44] for item in evoluciondelmercado]
MDde29 = [item[45] for item in evoluciondelmercado]
MDde30 = [item[46] for item in evoluciondelmercado]
MDde31 = [item[47] for item in evoluciondelmercado]
MDde32 = [item[48] for item in evoluciondelmercado]
MDde33 = [item[49] for item in evoluciondelmercado]
MDde34 = [item[50] for item in evoluciondelmercado]
MDde35 = [item[51] for item in evoluciondelmercado]
MDde36 = [item[52] for item in evoluciondelmercado]
MDde37 = [item[53] for item in evoluciondelmercado]
MDde38 = [item[54] for item in evoluciondelmercado]

# evolucion de las poblaciones de vendedores y compradores
dibujar_grafica(evoluciondelmercado) 

graficar_generacion(generacion, nocompraventa, comprasventas, 'exitos', 0)
graficar_generacion(generacion, exitosdeP , exitosdeD, 'exitos', 0)

graficar_generacion(generacion, tiempodeprocesoP , tiempodeprocesoD, 'tiempo de proceso', 0)
graficar_generacion(generacion, beneficiodeP , beneficiodeD, 'beneficio', 0)

graficar_generacion(generacion, MPde23 , MDde23, 'Cobertura 5G, wifi y bluetooth', 1)
graficar_generacion(generacion, MPde24 , MDde24, 'Duración de la batería', 1)
graficar_generacion(generacion, MPde25 , MDde25, 'Incorporación de licencias chatGPT, Bing, Dall-E, voice chatbot,...', 1)
graficar_generacion(generacion, MPde26 , MDde26, 'Ecocertificación', 1)
graficar_generacion(generacion, MPde27 , MDde27, 'Seguridad y privacidad', 1)
graficar_generacion(generacion, MPde28 , MDde28, 'Resolución de la cámara y software de enfoque, edición de fotos y videos', 1)
graficar_generacion(generacion, MPde29 , MDde29, 'Memoria y almacenamiento', 1)
graficar_generacion(generacion, MPde30 , MDde30, 'Tamaño, resolución de la pantalla', 1)
graficar_generacion(generacion, MPde31 , MDde31, 'Realidad Aumentada y gafas de realidad virtual', 1)
graficar_generacion(generacion, MPde32 , MDde32, 'Integración con otros sensores y dispositivos', 1)
graficar_generacion(generacion, MPde33 , MDde33, 'Diseño,ergonomía y estética', 1)
graficar_generacion(generacion, MPde34 , MDde34, 'Sensores, integración y funcionalidades IoT', 1)
graficar_generacion(generacion, MPde35 , MDde35, 'Usabilidad y accesibilidad', 1)
graficar_generacion(generacion, MPde36 , MDde36, 'Sensores, integración y funcionalidades de salud y bienestar', 1)
graficar_generacion(generacion, MPde37 , MDde37, 'Funcionalidades extra', 1)
graficar_generacion(generacion, MPde38 , MDde38, 'Accesorios disponibles', 1)

# evolucion de los parametros por cada individuo
graficarevolucion(resumengenD,resumengenP,'transacciones', 0) 
graficarevolucion(resumengenD,resumengenP,'beneficios', 0) 
graficarevolucion(resumengenD,resumengenP,'nuevos agentes', 0) 
graficarevolucion(resumengenD,resumengenP,'antiguedad', 0) 
graficarevolucion(resumengenD,resumengenP,'atributos', 0) 
graficarevolucion(resumengenD,resumengenP,'entrenamientos', 0) 
graficarevolucion(resumengenD,resumengenP,'modularidad', 1) 
graficarevolucion(resumengenD,resumengenP,'procesadores', 0) 
graficarevolucion(resumengenD,resumengenP,'tiempo de proceso', 0.1) 
#,'memoria','resolución y tamaño de pantalla','realidad aumentada y realidad virtual','integracion con otros sensores y dispositivos','diseño y estética','sensores integrados IoT','usabilidad y ergonomia','sensores y funcionalidades de bienestar y salud','otras funcionalidades adicionales','accesorios disponibles']

# otros por variables:    
graficarevolucion(resumengenD,resumengenP,'conectividad 5G, wifi, bluetooth, radiofrecuencia', 1)
graficarevolucion(resumengenD,resumengenP,'duracion de la bateria', 1)
graficarevolucion(resumengenD,resumengenP,'funciones inteligencia artificial', 1) 
graficarevolucion(resumengenD,resumengenP,'ecocertificacion, sostenibilidad', 1)
graficarevolucion(resumengenD,resumengenP,'seguridad y privacidad', 1)    
graficarevolucion(resumengenD,resumengenP,'camara y software de video', 1)
graficarevolucion(resumengenD,resumengenP,'memoria', 1)
graficarevolucion(resumengenD,resumengenP,'resolución y tamaño de pantalla', 1)
graficarevolucion(resumengenD,resumengenP,'realidad aumentada y realidad virtual', 1)
graficarevolucion(resumengenD,resumengenP,'integracion con otros sensores y dispositivos', 1) 
graficarevolucion(resumengenD,resumengenP,'diseño y estética', 1)
graficarevolucion(resumengenD,resumengenP,'sensores integrados IoT', 1)    
graficarevolucion(resumengenD,resumengenP,'usabilidad y ergonomia', 1)
graficarevolucion(resumengenD,resumengenP,'sensores y funcionalidades de bienestar y salud', 1)
graficarevolucion(resumengenD,resumengenP,'otras funcionalidades adicionales', 1)
graficarevolucion(resumengenD,resumengenP,'accesorios disponibles', 1) 

pass

# medir tiempo
tiempofinal= time.process_time()
tiempototal= (tiempofinal - tiempoinicial)
print('tiempo', tiempototal ,'para', generaciones ,'generaciones y', numeroD ,'vendedores')

# fichero animacion.gif que cuenta la historia
animacion()