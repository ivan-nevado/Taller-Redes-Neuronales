# Plan de Implementación: CNN con Metodología López de Prado

## 📋 Resumen Ejecutivo

Este plan detalla cómo adaptar tu implementación de **CNN** para usar las técnicas avanzadas de **López de Prado** que ya aplicaste exitosamente en RNN:

1. **Portfolio Dollar Bars** (en lugar de Time Bars)
2. **Diferenciación Fraccionaria FFD** (en lugar de retornos logarítmicos simples)
3. **Triple Barrier Method** (clasificación en lugar de regresión)

---

## 🎯 Objetivo

Crear `redes_convolucionales_lopezDePrado.ipynb` que combine:
- La arquitectura CNN de tu notebook actual
- Las técnicas de preprocesamiento de datos de López de Prado de tu RNN
- Mantener la misma estructura de experimentación (16 configuraciones)

---

## 📊 Comparación: CNN Actual vs CNN López de Prado

| Aspecto | CNN Actual | CNN López de Prado |
|---------|------------|-------------------|
| **Datos de entrada** | Time Bars (16,190 obs) | Dollar Bars (2,504 obs) |
| **Transformación** | Retornos log simples | FracDiff (d=0.45) |
| **Etiquetado** | Regresión (predecir retornos) | Clasificación (Triple Barrera) |
| **Salida del modelo** | 23 valores continuos | 3 clases (Sube/Plano/Baja) |
| **Función de pérdida** | MAE | Categorical Crossentropy |
| **Métrica** | MAE | Accuracy |
| **Baselines** | Naive, SMA, Buy&Hold | Always Buy, Always Flat, Most Frequent |

---

## 🔧 Estructura del Notebook (8 Celdas)

### Celda 1: Imports y Configuración
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import random
import yfinance as yf

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

# Semilla fija
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

warnings.simplefilter(action="ignore", category=FutureWarning)
```

---

### Celda 2: Preparación de Datos - Portfolio Dollar Bars

**COPIAR DIRECTAMENTE DE TU RNN** (Celda 3):

```python
# =====================================================================
# 1. DESCARGA Y PREPARACIÓN DE DATOS: PORTFOLIO DOLLAR BARS
# =====================================================================
print("Descargando datos de Yahoo Finance (Precios y Volumen)...")
start_date = '1960-01-01'
tickers_validos = ['AEP', 'BA', 'CAT', 'CNP', 'CVX', 'DIS', 'DTE', 'ED', 'GD', 'GE', 
                   'HON', 'HPQ', 'IBM', 'IP', 'JNJ', 'KO', 'KR', 'MMM', 'MO', 'MRK', 
                   'MSI', 'PG', 'XOM']

# Descargamos todos los datos (OHLCV)
datos_yahoo = yf.download(tickers_validos, start=start_date, auto_adjust=True, progress=False)

# Separamos Precios de Cierre y Volúmenes
precios_close = datos_yahoo['Close'].dropna(axis=1)
volumenes = datos_yahoo['Volume'].dropna(axis=1)

print(f"Datos cronológicos originales (Time Bars): {precios_close.shape}")

# A) Cálculo del Dollar Volume del Portfolio
dollar_volume_individual = precios_close * volumenes
portfolio_dollar_volume = dollar_volume_individual.sum(axis=1)

# B) Función de Muestreo de Dollar Bars
def sample_portfolio_dollar_bars(df_precios, serie_dollar_vol, threshold):
    """
    Recorre la serie cronológica acumulando dólares negociados.
    Cuando el acumulado supera el 'threshold', guarda la fecha y resetea.
    """
    fechas_barras = []
    vol_acumulado = 0.0
    
    fechas = serie_dollar_vol.index
    volumenes_diarios = serie_dollar_vol.values
    
    for fecha, vol in zip(fechas, volumenes_diarios):
        if np.isnan(vol):
            continue
            
        vol_acumulado += vol
        
        if vol_acumulado >= threshold:
            fechas_barras.append(fecha)
            vol_acumulado = 0.0
            
    return df_precios.loc[fechas_barras]

# C) Generación de la nueva matriz indexada por Información
umbral_dolares = portfolio_dollar_volume.mean() * 5 
precios_dollar_bars = sample_portfolio_dollar_bars(precios_close, portfolio_dollar_volume, umbral_dolares)

print(f"Nueva matriz tras compresión (Portfolio Dollar Bars): {precios_dollar_bars.shape}")
```

---

### Celda 3: Diferenciación Fraccionaria (FFD)

**COPIAR DIRECTAMENTE DE TU RNN** (Celda 4):

```python
# =====================================================================
# 2. TRANSFORMACIÓN LÓPEZ DE PRADO: DIFERENCIACIÓN FRACCIONARIA (FFD)
# =====================================================================

def obtener_pesos_fracdiff(d, umbral=1e-4):
    """
    Calcula los pesos para la diferenciación fraccionaria.
    """
    w = [1.]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < umbral:
            break
        w.append(w_k)
        k += 1
    
    return np.array(w[::-1])

def aplicar_fracdiff_ffd(df_precios, d, umbral=1e-4):
    """
    Aplica FracDiff a un DataFrame entero usando una ventana fija (FFD).
    """
    print(f"Calculando Diferenciación Fraccionaria con d={d}...")
    
    w = obtener_pesos_fracdiff(d, umbral)
    ventana = len(w)
    print(f"Tamaño de la ventana de memoria (días/barras retenidas): {ventana}")
    
    df_log = np.log(df_precios)
    df_diff = pd.DataFrame(index=df_log.index, columns=df_log.columns)
    
    for i in range(ventana - 1, len(df_log)):
        corte = df_log.iloc[i - ventana + 1 : i + 1]
        df_diff.iloc[i] = np.dot(w, corte.values)
        
    return df_diff.dropna().astype(float)

# EJECUCIÓN DEL ALGORITMO FRACDIFF
grado_d = 0.45 
fracdiff_features = aplicar_fracdiff_ffd(precios_dollar_bars, d=grado_d)

print(f"Forma de los datos FracDiff finales: {fracdiff_features.shape}")
```

---

### Celda 4: Triple Barrier Method y Arquitectura CNN

**ADAPTAR DE TU RNN + CNN**:

```python
from tensorflow.keras.utils import to_categorical

# =====================================================================
# 3. TRIPLE BARRIER METHOD Y ARQUITECTURA CNN
# =====================================================================

def create_triple_barrier_data(data_features, data_prices, input_window_size, output_window_size, pt_limit, sl_limit):
    """
    Genera etiquetas de clasificación usando el Método de la Triple Barrera.
    IDÉNTICO A TU RNN.
    """
    X, y = [], []
    
    features_array = data_features.values if isinstance(data_features, pd.DataFrame) else data_features
    precios_array = data_prices.mean(axis=1).values if isinstance(data_prices, pd.DataFrame) else data_prices

    for i in range(len(features_array) - input_window_size - output_window_size + 1):
        # 1. Ventana de Entrada (Input Features: FracDiff)
        input_sequence = features_array[i : i + input_window_size]
        X.append(input_sequence)
        
        # 2. Trayectoria Futura (Output: Precios reales para la barrera)
        p0 = precios_array[i + input_window_size - 1]
        future_prices = precios_array[i + input_window_size : i + input_window_size + output_window_size]
        
        path_returns = (future_prices / p0) - 1
        
        # 3. Lógica de la Triple Barrera
        etiqueta = 1  # Por defecto, Clase 1 (Plano)
        
        for r in path_returns:
            if r >= pt_limit:
                etiqueta = 2  # Clase 2 (Sube)
                break
            elif r <= -sl_limit:
                etiqueta = 0  # Clase 0 (Baja)
                break
                
        y.append(etiqueta)

    X = np.array(X)
    y = to_categorical(y, num_classes=3)
    
    return X, y


def construir_modelo_cnn_clasificacion(config, input_shape):
    """
    Construye un modelo CNN 1D para CLASIFICACIÓN (3 clases).
    ADAPTADO DE TU CNN ACTUAL.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # L2 regularization
    l2_reg = regularizers.l2(config.get('l2', 0.0))
    
    # Primera capa convolucional
    model.add(Conv1D(filters=config['filters'], 
                     kernel_size=config['kernel_size'], 
                     activation='relu', 
                     padding='same',
                     kernel_regularizer=l2_reg))
    
    # Pooling (opcional)
    if config.get('use_pooling', False):
        model.add(MaxPooling1D(pool_size=2))
    
    # Segunda capa convolucional (opcional)
    if config.get('double_conv', False):
        model.add(Conv1D(filters=config['filters']*2, 
                         kernel_size=config['kernel_size'], 
                         activation='relu', 
                         padding='same',
                         kernel_regularizer=l2_reg))
    
    # Aplanar o GlobalAveragePooling
    if config.get('use_flatten', False):
        model.add(Flatten())
    else:
        model.add(GlobalAveragePooling1D())
    
    model.add(Dropout(config['dropout']))
    
    # CAMBIO CRÍTICO: 3 neuronas de salida con Softmax
    model.add(Dense(3, activation='softmax', kernel_regularizer=l2_reg))
    
    optimizador = Adam(learning_rate=config['lr'])
    # CAMBIO CRÍTICO: Categorical Crossentropy + Accuracy
    model.compile(optimizer=optimizador, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def calcular_baselines_clasificacion(y_test, y_train):
    """
    Calcula la Precisión (Accuracy) para modelos base simples.
    IDÉNTICO A TU RNN.
    """
    y_test_classes = np.argmax(y_test, axis=1)
    y_train_classes = np.argmax(y_train, axis=1)
    
    # 1. Always Buy
    acc_always_buy = np.mean(y_test_classes == 2)
    
    # 2. Always Flat
    acc_always_flat = np.mean(y_test_classes == 1)
    
    # 3. Most Frequent (Buy & Hold equivalente)
    clase_mas_frecuente = np.bincount(y_train_classes).argmax()
    acc_most_frequent = np.mean(y_test_classes == clase_mas_frecuente)
    
    return acc_always_buy, acc_always_flat, acc_most_frequent


# Crear carpeta para gráficas
os.makedirs('graficas_convergencia_cnn_tbm', exist_ok=True)
```

---

## ⏭️ Continuación en Siguiente Mensaje

El plan continúa con:
- Celda 5: Configuración de Hiperparámetros (8 bancos)
- Celda 6: Bucle Principal de Entrenamiento
- Celda 7: Resultados Finales
- Celda 8: Gráficas Consolidadas

¿Continúo con la siguiente parte del plan?


---

### Celda 5: Configuración de Hiperparámetros

**ADAPTAR DE TU CNN ACTUAL** - Mantener la misma estructura de 8 bancos:

```python
# =====================================================================
# 4. CONFIGURACIÓN DEL EXPERIMENTO
# =====================================================================

input_windows = [5, 10, 30, 90]
output_windows = [1, 5, 30, 90]

# =====================================================================
# BANCOS DE HIPERPARÁMETROS ESPECÍFICOS POR VENTANA
# =====================================================================

# Ventanas CORTAS de entrada (5 días) - Predicciones CORTAS (1, 5 días)
hp_in5_corto = [
    {'filters': 8, 'kernel_size': 3, 'dropout': 0.0, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 16, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 8, 'kernel_size': 3, 'dropout': 0.0, 'lr': 0.0001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
]

# Ventanas CORTAS de entrada (5 días) - Predicciones LARGAS (30, 90 días)
hp_in5_largo = [
    {'filters': 32, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.001, 'l2': 0.01, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 64, 'kernel_size': 3, 'dropout': 0.3, 'lr': 0.0005, 'l2': 0.01, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 32, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.0005, 'l2': 0.005, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
]

# Ventanas MEDIAS de entrada (10 días) - Predicciones CORTAS (1, 5 días)
hp_in10_corto = [
    {'filters': 16, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 32, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.001, 'l2': 0.005, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 16, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.0005, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
]

# Ventanas MEDIAS de entrada (10 días) - Predicciones LARGAS (30, 90 días)
hp_in10_largo = [
    {'filters': 64, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 128, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 64, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.0005, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 128, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.0005, 'l2': 0.0, 'double_conv': True, 'use_pooling': False, 'use_flatten': False},
]

# Ventanas LARGAS de entrada (30 días) - Predicciones CORTAS (1, 5 días)
hp_in30_corto = [
    {'filters': 32, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 64, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 64, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.0005, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 128, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
]

# Ventanas LARGAS de entrada (30 días) - Predicciones LARGAS (30, 90 días)
hp_in30_largo = [
    {'filters': 64, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 128, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 128, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.0005, 'l2': 0.0, 'double_conv': True, 'use_pooling': False, 'use_flatten': False},
    {'filters': 256, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.0005, 'l2': 0.0, 'double_conv': True, 'use_pooling': False, 'use_flatten': False},
]

# Ventanas MUY LARGAS de entrada (90 días) - Predicciones CORTAS (1, 5 días)
hp_in90_corto = [
    {'filters': 16, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 32, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 64, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 128, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.0005, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
]

# Ventanas MUY LARGAS de entrada (90 días) - Predicciones LARGAS (30, 90 días)
hp_in90_largo = [
    {'filters': 64, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 128, 'kernel_size': 3, 'dropout': 0.1, 'lr': 0.001, 'l2': 0.0, 'double_conv': False, 'use_pooling': False, 'use_flatten': False},
    {'filters': 128, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.0005, 'l2': 0.0, 'double_conv': True, 'use_pooling': False, 'use_flatten': False},
    {'filters': 256, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.0005, 'l2': 0.0, 'double_conv': True, 'use_pooling': False, 'use_flatten': False},
]

# Matrices para reportar resultados finales (TRAIN, VAL, TEST)
matriz_acc_cnn_train = np.zeros((4, 4))
matriz_acc_cnn_val = np.zeros((4, 4))
matriz_acc_cnn_test = np.zeros((4, 4))
matriz_num_params = np.zeros((4, 4))

# Matrices para baselines
matriz_acc_naive_val = np.zeros((4, 4))
matriz_acc_most_frequent_val = np.zeros((4, 4))
matriz_acc_naive_test = np.zeros((4, 4))
matriz_acc_most_frequent_test = np.zeros((4, 4))

# Lista para guardar info detallada
resultados_detallados = []

# Early stopping con patience aumentado
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
```

---


### Celda 6: Bucle Principal de Entrenamiento

**COMBINAR LÓGICA DE TU RNN + CNN**:

```python
# =====================================================================
# 5. BUCLE PRINCIPAL DE ENTRENAMIENTO
# =====================================================================

print("\nIniciando entrenamiento de modelos CNN con Triple Barrier Method...")

# Definimos los límites de la Triple Barrera (2% de subida o bajada)
PT_LIMIT = 0.02 
SL_LIMIT = 0.02

for i, in_w in enumerate(input_windows):
    for j, out_w in enumerate(output_windows):
        print(f"\n=======================================================")
        print(f" Ventana Entrada (Memoria): {in_w} barras | Salida (Horizonte): {out_w} barras")
        print(f"=======================================================")
        
        # 1. Crear datos con FracDiff y Triple Barrera
        X, y = create_triple_barrier_data(
            fracdiff_features, 
            precios_dollar_bars, 
            in_w, 
            out_w, 
            pt_limit=PT_LIMIT, 
            sl_limit=SL_LIMIT
        )
        
        # 2. Separación CRONOLÓGICA: 70% Train, 20% Validacion, 10% Test
        split_1 = int(len(X) * 0.70)
        split_2 = int(len(X) * 0.90)
        
        X_train, y_train = X[:split_1], y[:split_1]
        X_val, y_val = X[split_1:split_2], y[split_1:split_2]
        X_test, y_test = X[split_2:], y[split_2:]
        
        # 3. Baselines de Clasificación
        # Calcular en Validación
        acc_buy_val, acc_flat_val, acc_freq_val = calcular_baselines_clasificacion(y_val, y_train)
        matriz_acc_naive_val[i, j] = acc_buy_val
        matriz_acc_most_frequent_val[i, j] = acc_freq_val
        
        # Calcular en Test
        acc_buy_test, acc_flat_test, acc_freq_test = calcular_baselines_clasificacion(y_test, y_train)
        matriz_acc_naive_test[i, j] = acc_buy_test
        matriz_acc_most_frequent_test[i, j] = acc_freq_test

        print("--- Baselines (Accuracy) VALIDACIÓN ---")
        print(f"Always Buy: {acc_buy_val:.4f} | Always Flat: {acc_flat_val:.4f} | Most Frequent: {acc_freq_val:.4f}")
        print("--- Baselines (Accuracy) TEST ---")
        print(f"Always Buy: {acc_buy_test:.4f} | Always Flat: {acc_flat_test:.4f} | Most Frequent: {acc_freq_test:.4f}\n")

        # 4. Búsqueda del mejor modelo CNN
        mejor_val_loss = float('inf')
        mejor_modelo = None
        mejor_historial = None
        mejor_config = None
        
        # SELECCIÓN DE BANCO DE HIPERPARÁMETROS
        if in_w == 5:
            lista_a_probar = hp_in5_corto if out_w in [1, 5] else hp_in5_largo
            nombre_lista = "In:5 Corto" if out_w in [1, 5] else "In:5 Largo"
        elif in_w == 10:
            lista_a_probar = hp_in10_corto if out_w in [1, 5] else hp_in10_largo
            nombre_lista = "In:10 Corto" if out_w in [1, 5] else "In:10 Largo"
        elif in_w == 30:
            lista_a_probar = hp_in30_corto if out_w in [1, 5] else hp_in30_largo
            nombre_lista = "In:30 Corto" if out_w in [1, 5] else "In:30 Largo"
        elif in_w == 90:
            lista_a_probar = hp_in90_corto if out_w in [1, 5] else hp_in90_largo
            nombre_lista = "In:90 Corto" if out_w in [1, 5] else "In:90 Largo"

        print(f" -> Usando banco de pruebas: [{nombre_lista}]")
        
        for config in lista_a_probar:
            print(f" -> Entrenando CNN: Filtros={config['filters']}, Kernel={config['kernel_size']}, LR={config['lr']}, Dropout={config['dropout']}, L2={config['l2']}...")
            
            # Construimos el modelo de CLASIFICACIÓN
            modelo = construir_modelo_cnn_clasificacion(config, input_shape=(in_w, 23))
            
            historial = modelo.fit(X_train, y_train, 
                                   validation_data=(X_val, y_val),
                                   epochs=50, 
                                   batch_size=64, 
                                   callbacks=[early_stop], 
                                   verbose=0)
            
            val_loss_actual = min(historial.history['val_loss'])
            
            if val_loss_actual < mejor_val_loss:
                mejor_val_loss = val_loss_actual
                mejor_modelo = modelo
                mejor_historial = historial
                mejor_config = config
        
        print(f"\n[GANADOR] CNN con {mejor_config['filters']} filtros, kernel={mejor_config['kernel_size']}")
        
        # 5. Evaluación final del GANADOR en TRAIN, VALIDACIÓN y TEST
        # evaluate ahora devuelve [loss, accuracy]
        loss_train, acc_train_ganador = mejor_modelo.evaluate(X_train, y_train, verbose=0)
        loss_val, acc_val_ganador = mejor_modelo.evaluate(X_val, y_val, verbose=0)
        loss_test, acc_test_ganador = mejor_modelo.evaluate(X_test, y_test, verbose=0)
        num_params = mejor_modelo.count_params()
        
        # Guardar en sus respectivas matrices
        matriz_acc_cnn_train[i, j] = acc_train_ganador
        matriz_acc_cnn_val[i, j] = acc_val_ganador
        matriz_acc_cnn_test[i, j] = acc_test_ganador
        matriz_num_params[i, j] = num_params
        
        print(f"Accuracy del Modelo Ganador en TRAIN:      {acc_train_ganador:.4f}")
        print(f"Accuracy del Modelo Ganador en VALIDACIÓN: {acc_val_ganador:.4f}")
        print(f"Accuracy del Modelo Ganador en TEST:       {acc_test_ganador:.4f}")
        print(f"Parámetros:   {num_params:,}")
        
        # Guardar info detallada
        resultados_detallados.append({
            'in_window': in_w,
            'out_window': out_w,
            'filters': mejor_config['filters'],
            'kernel_size': mejor_config['kernel_size'],
            'lr': mejor_config['lr'],
            'dropout': mejor_config['dropout'],
            'l2': mejor_config.get('l2', 0.0),
            'acc_train': acc_train_ganador,
            'acc_val': acc_val_ganador,
            'acc_test': acc_test_ganador,
            'num_params': num_params
        })
        
        # 6. Guardar Gráficas de Convergencia (Pérdida y Precisión)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        filtros = mejor_config['filters']
        kernel = mejor_config['kernel_size']
        l_rate = mejor_config['lr']
        d_out = mejor_config['dropout']
        fig.suptitle(f"Clasificación CNN | Filtros: {filtros} | Kernel: {kernel} | LR: {l_rate} | Drop: {d_out}\n(In:{in_w} - Out:{out_w})")
        
        # Gráfica de Pérdida (Loss)
        ax1.plot(mejor_historial.history['loss'], label='Loss Entrenamiento')
        ax1.plot(mejor_historial.history['val_loss'], label='Loss Validación')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Categorical Crossentropy')
        ax1.legend()
        ax1.grid(True)
        
        # Gráfica de Precisión (Accuracy)
        ax2.plot(mejor_historial.history['accuracy'], label='Accuracy Entrenamiento')
        ax2.plot(mejor_historial.history['val_accuracy'], label='Accuracy Validación')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Guardar la imagen
        nombre_archivo = f"graficas_convergencia_cnn_tbm/conver_in{in_w}_out{out_w}.png"
        plt.savefig(nombre_archivo)
        plt.close()
```

---


### Celda 7: Resultados Finales

**ADAPTAR DE TU RNN**:

```python
# =====================================================================
# 6. RESULTADOS FINALES EN CLASIFICACIÓN
# =====================================================================

print("\n\n" + "="*60)
print("MATRIZ DE RESULTADOS FINALES EN ENTRENAMIENTO (ACCURACY)")
print("="*60)
df_cnn_train_acc = pd.DataFrame(matriz_acc_cnn_train, 
                            index=[f'In_{w}' for w in input_windows], 
                            columns=[f'Out_{w}' for w in output_windows])
print(df_cnn_train_acc)

print("\n" + "="*60)
print("MATRIZ DE RESULTADOS FINALES EN VALIDACIÓN (ACCURACY)")
print("="*60)
df_cnn_val_acc = pd.DataFrame(matriz_acc_cnn_val, 
                          index=[f'In_{w}' for w in input_windows], 
                          columns=[f'Out_{w}' for w in output_windows])
print(df_cnn_val_acc)

print("\n" + "="*60)
print("MATRIZ DE RESULTADOS FINALES EN TEST (ACCURACY)")
print("="*60)
df_cnn_test_acc = pd.DataFrame(matriz_acc_cnn_test, 
                      index=[f'In_{w}' for w in input_windows], 
                      columns=[f'Out_{w}' for w in output_windows])
print(df_cnn_test_acc)

print("\n" + "="*60)
print("MATRIZ DE NÚMERO DE PARÁMETROS")
print("="*60)
df_params = pd.DataFrame(matriz_num_params, 
                         index=[f'In_{w}' for w in input_windows], 
                         columns=[f'Out_{w}' for w in output_windows])
print(df_params.astype(int))

print("\n" + "="*60)
print("MATRIZ BASELINE 'ALWAYS BUY' (VALIDACIÓN)")
print("="*60)
df_always_buy_val = pd.DataFrame(matriz_acc_naive_val, 
                        index=[f'In_{w}' for w in input_windows], 
                        columns=[f'Out_{w}' for w in output_windows])
print(df_always_buy_val)

print("\n" + "="*60)
print("MATRIZ BASELINE 'ALWAYS BUY' (TEST)")
print("="*60)
df_always_buy_test = pd.DataFrame(matriz_acc_naive_test, 
                        index=[f'In_{w}' for w in input_windows], 
                        columns=[f'Out_{w}' for w in output_windows])
print(df_always_buy_test)

print("\n" + "="*60)
print("MATRIZ BASELINE 'MOST FREQUENT' (VALIDACIÓN) - Eq. Buy&Hold")
print("="*60)
df_most_freq_val = pd.DataFrame(matriz_acc_most_frequent_val, 
                      index=[f'In_{w}' for w in input_windows], 
                      columns=[f'Out_{w}' for w in output_windows])
print(df_most_freq_val)

print("\n" + "="*60)
print("MATRIZ BASELINE 'MOST FREQUENT' (TEST) - Eq. Buy&Hold")
print("="*60)
df_most_freq_test = pd.DataFrame(matriz_acc_most_frequent_test, 
                     index=[f'In_{w}' for w in input_windows], 
                     columns=[f'Out_{w}' for w in output_windows])
print(df_most_freq_test)

print("\n" + "="*60)
print("TABLA DETALLADA DE RESULTADOS")
print("="*60)
df_detallado = pd.DataFrame(resultados_detallados)
print(df_detallado.to_string(index=False))
```

---

### Celda 8: Gráficas Consolidadas

**ADAPTAR DE TU CNN ACTUAL**:

```python
# =====================================================================
# 7. GRÁFICAS CONSOLIDADAS POR VENTANA DE SALIDA
# =====================================================================

print("\nGenerando gráficas consolidadas...")

for j, out_w in enumerate(output_windows):
    plt.figure(figsize=(10, 6))
    
    # Extraer resultados para esta ventana de salida
    resultados_out = []
    for i, in_w in enumerate(input_windows):
        resultados_out.append(matriz_acc_cnn_test[i, j])
    
    # Crear gráfica de barras
    x_pos = np.arange(len(input_windows))
    plt.bar(x_pos, resultados_out, alpha=0.7, color='steelblue')
    plt.xticks(x_pos, [f'In_{w}' for w in input_windows])
    plt.ylabel('Accuracy en Test')
    plt.xlabel('Ventana de Entrada')
    plt.title(f'Comparación de Modelos CNN (Triple Barrier) para Ventana de Salida = {out_w} barras')
    plt.grid(True, alpha=0.3)
    
    # Añadir valores encima de las barras
    for idx, val in enumerate(resultados_out):
        plt.text(idx, val, f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'graficas_convergencia_cnn_tbm/consolidada_out{out_w}.png')
    plt.close()

print("✅ Gráficas consolidadas guardadas en 'graficas_convergencia_cnn_tbm/'")
```

---

## 🔍 Diferencias Clave vs CNN Actual

### 1. Datos de Entrada
- **Antes**: `returns = np.log(precios_close).diff().dropna()` → 16,190 observaciones
- **Ahora**: `fracdiff_features` → 2,267 observaciones (Dollar Bars + FracDiff)

### 2. Función de Creación de Datos
- **Antes**: `create_time_series_data()` → Regresión (predecir retornos promedio)
- **Ahora**: `create_triple_barrier_data()` → Clasificación (predecir clase 0/1/2)

### 3. Arquitectura del Modelo
- **Antes**: `model.add(Dense(23))` → Salida continua
- **Ahora**: `model.add(Dense(3, activation='softmax'))` → Salida categórica

### 4. Compilación
- **Antes**: `loss='mae'`
- **Ahora**: `loss='categorical_crossentropy', metrics=['accuracy']`

### 5. Baselines
- **Antes**: `calcular_baselines()` → MAE (Naive, SMA, Buy&Hold)
- **Ahora**: `calcular_baselines_clasificacion()` → Accuracy (Always Buy, Always Flat, Most Frequent)

### 6. Evaluación
- **Antes**: `mae_test = modelo.evaluate(X_test, y_test)`
- **Ahora**: `loss_test, acc_test = modelo.evaluate(X_test, y_test)`

---

## ✅ Checklist de Implementación

### Paso 1: Preparación
- [ ] Crear archivo `redes_convolucionales_lopezDePrado.ipynb`
- [ ] Copiar estructura de 8 celdas

### Paso 2: Celdas 1-3 (Datos)
- [ ] Celda 1: Imports (copiar de RNN + añadir Conv1D)
- [ ] Celda 2: Portfolio Dollar Bars (copiar EXACTO de RNN)
- [ ] Celda 3: FracDiff (copiar EXACTO de RNN)

### Paso 3: Celda 4 (Arquitectura)
- [ ] Copiar `create_triple_barrier_data()` de RNN
- [ ] Adaptar `construir_modelo_cnn()` → `construir_modelo_cnn_clasificacion()`
- [ ] Copiar `calcular_baselines_clasificacion()` de RNN

### Paso 4: Celda 5 (Hiperparámetros)
- [ ] Copiar 8 bancos de hiperparámetros de CNN actual
- [ ] Crear matrices de accuracy (no MAE)

### Paso 5: Celda 6 (Entrenamiento)
- [ ] Adaptar bucle principal combinando RNN + CNN
- [ ] Cambiar evaluación a accuracy
- [ ] Guardar gráficas con loss + accuracy

### Paso 6: Celdas 7-8 (Resultados)
- [ ] Adaptar tablas de resultados (accuracy en lugar de MAE)
- [ ] Crear gráficas consolidadas

### Paso 7: Validación
- [ ] Ejecutar notebook completo
- [ ] Verificar que genera 16 modelos
- [ ] Comprobar que las gráficas se guardan correctamente
- [ ] Comparar resultados con RNN López de Prado

---

## 🎯 Resultados Esperados

### Matrices de Salida
1. **Accuracy en Train** (4x4)
2. **Accuracy en Validación** (4x4)
3. **Accuracy en Test** (4x4)
4. **Número de Parámetros** (4x4)
5. **Baselines Always Buy** (Validación y Test)
6. **Baselines Most Frequent** (Validación y Test)

### Gráficas Generadas
- 16 gráficas de convergencia individuales (loss + accuracy)
- 4 gráficas consolidadas por ventana de salida

### Tabla Detallada
Columnas: `in_window`, `out_window`, `filters`, `kernel_size`, `lr`, `dropout`, `l2`, `acc_train`, `acc_val`, `acc_test`, `num_params`

---

## 📊 Comparación Final: RNN vs CNN (López de Prado)

Una vez ejecutado, podrás comparar:

| Métrica | RNN (LSTM/GRU) | CNN (Conv1D) |
|---------|----------------|--------------|
| Accuracy Test (Out_1) | ~97% | ? |
| Accuracy Test (Out_5) | ~37% | ? |
| Accuracy Test (Out_30) | ~63% | ? |
| Accuracy Test (Out_90) | ~56% | ? |
| Parámetros (promedio) | ~50K | ? |
| Tiempo de entrenamiento | Alto | Bajo |

**Hipótesis**: CNN debería ser más rápida pero con accuracy similar, ya que ambas usan los mismos datos (Dollar Bars + FracDiff + Triple Barrier).

---

## 🚀 Próximos Pasos

1. **Ejecutar el notebook completo**
2. **Documentar resultados** en un nuevo `.md`
3. **Comparar CNN vs RNN** con López de Prado
4. **Analizar si CNN captura mejor patrones locales** que RNN
5. **Probar ensemble** (combinar predicciones de CNN + RNN)

---

## 📝 Notas Importantes

- **No cambies los datos de entrada**: Usa exactamente los mismos Dollar Bars y FracDiff que en RNN
- **Mantén los mismos hiperparámetros**: Usa los 8 bancos que ya funcionan en tu CNN actual
- **Respeta la división cronológica**: 70/20/10 sin shuffle
- **Usa las mismas barreras**: PT_LIMIT = SL_LIMIT = 0.02 (2%)
- **Compara manzanas con manzanas**: Ambos modelos (RNN y CNN) deben usar exactamente los mismos datos de entrada

---

¿Listo para implementar? 🎯
