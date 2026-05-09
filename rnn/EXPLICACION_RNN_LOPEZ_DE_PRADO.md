# Implementación de RNN con Metodología López de Prado

## Resumen Ejecutivo

Este notebook implementa un sistema de predicción financiera basado en **Redes Neuronales Recurrentes (LSTM/GRU)** aplicando las técnicas avanzadas de **Marcos López de Prado** para el procesamiento de datos financieros:

1. **Portfolio Dollar Bars** - Muestreo basado en información en lugar de tiempo
2. **Diferenciación Fraccionaria (FFD)** - Estacionariedad con memoria
3. **Triple Barrier Method** - Etiquetado de clasificación para trading

---

## 1. Preparación de Datos: Portfolio Dollar Bars

### ¿Qué problema resuelve?

Los datos financieros tradicionales están indexados por **tiempo** (1 día = 1 observación). Esto genera dos problemas:
- **Días tranquilos** (poco volumen) tienen el mismo peso que **días volátiles** (mucho volumen)
- La información no está distribuida uniformemente en el tiempo

### Solución: Dollar Bars

En lugar de muestrear cada día, se crea una nueva "barra" cada vez que se negocia una **cantidad fija de dólares** en el mercado.

### Implementación

```python
# 1. Calcular el volumen en dólares de cada activo
dollar_volume_individual = precios_close * volumenes

# 2. Sumar el volumen del portfolio completo (23 empresas)
portfolio_dollar_volume = dollar_volume_individual.sum(axis=1)

# 3. Definir umbral: cantidad de dólares que debe acumular cada barra
umbral_dolares = portfolio_dollar_volume.mean() * 5  # 5 días promedio

# 4. Muestrear: cerrar barra cuando se supera el umbral
def sample_portfolio_dollar_bars(df_precios, serie_dollar_vol, threshold):
    fechas_barras = []
    vol_acumulado = 0.0
    
    for fecha, vol in zip(fechas, volumenes_diarios):
        vol_acumulado += vol
        
        if vol_acumulado >= threshold:
            fechas_barras.append(fecha)  # Cerrar barra
            vol_acumulado = 0.0          # Reset
            
    return df_precios.loc[fechas_barras]
```

### Resultado

- **Datos originales (Time Bars)**: 16,192 observaciones
- **Datos comprimidos (Dollar Bars)**: 2,504 observaciones
- **Compresión**: ~85% (solo se guardan los momentos con información relevante)

---

## 2. Transformación: Diferenciación Fraccionaria (FFD)

### ¿Qué problema resuelve?

Los precios financieros son **no estacionarios** (tienen tendencia). Las soluciones tradicionales:
- **Diferenciación simple** (`diff()`): Pierde toda la memoria histórica
- **Retornos logarítmicos**: Pierde la estructura de dependencia temporal

### Solución: Fractional Differentiation

Aplica una diferenciación de **orden fraccionario** `d` (entre 0 y 1):
- `d = 0`: Serie original (no estacionaria, máxima memoria)
- `d = 1`: Diferenciación completa (estacionaria, sin memoria)
- `d = 0.45`: **Punto óptimo** (estacionaria con 70% de memoria)

### Implementación

```python
def obtener_pesos_fracdiff(d, umbral=1e-4):
    """Calcula los pesos matemáticos de la diferenciación fraccionaria"""
    w = [1.]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)  # Fórmula recursiva de López de Prado
        if abs(w_k) < umbral:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])

def aplicar_fracdiff_ffd(df_precios, d, umbral=1e-4):
    """Aplica FracDiff con ventana fija (FFD)"""
    w = obtener_pesos_fracdiff(d, umbral)
    ventana = len(w)  # Tamaño de la memoria (238 barras)
    
    df_log = np.log(df_precios)  # Logaritmos para estabilizar varianza
    df_diff = pd.DataFrame(index=df_log.index, columns=df_log.columns)
    
    # Producto punto deslizante: suma ponderada de la historia
    for i in range(ventana - 1, len(df_log)):
        corte = df_log.iloc[i - ventana + 1 : i + 1]
        df_diff.iloc[i] = np.dot(w, corte.values)
        
    return df_diff.dropna().astype(float)
```

### Parámetros Utilizados

- **d = 0.45**: Grado de diferenciación (punto dulce para S&P 500)
- **Ventana de memoria**: 238 barras (días históricos que influyen en cada observación)
- **Datos finales**: 2,267 observaciones (se pierden las primeras 237 por falta de historia)

---

## 3. Etiquetado: Triple Barrier Method

### ¿Qué problema resuelve?

En trading, no queremos predecir el precio exacto de mañana, sino **si debemos comprar, vender o esperar**.

### Solución: Triple Barrera

Se define una "trayectoria futura" de `output_window_size` días y se etiqueta según qué barrera toca primero:

```
Precio
  ^
  |     ┌─────────────────────  Barrera Superior (Take Profit +2%)
  |     │                       → Clase 2: SUBE
  |─────┼───────────────────────  Precio Inicial (t0)
  |     │                       → Clase 1: PLANO (si expira el tiempo)
  |     └─────────────────────  Barrera Inferior (Stop Loss -2%)
  |                             → Clase 0: BAJA
  └──────────────────────────> Tiempo
```

### Implementación

```python
def create_triple_barrier_data(data_features, data_prices, 
                                input_window_size, output_window_size, 
                                pt_limit, sl_limit):
    X, y = [], []
    
    for i in range(len(features_array) - input_window_size - output_window_size + 1):
        # 1. Ventana de entrada (FracDiff)
        input_sequence = features_array[i : i + input_window_size]
        X.append(input_sequence)
        
        # 2. Trayectoria futura (precios reales)
        p0 = precios_array[i + input_window_size - 1]
        future_prices = precios_array[i + input_window_size : 
                                      i + input_window_size + output_window_size]
        path_returns = (future_prices / p0) - 1
        
        # 3. Lógica de la Triple Barrera
        etiqueta = 1  # Por defecto: PLANO
        
        for r in path_returns:
            if r >= pt_limit:      # +2%
                etiqueta = 2       # SUBE
                break
            elif r <= -sl_limit:   # -2%
                etiqueta = 0       # BAJA
                break
                
        y.append(etiqueta)
    
    return np.array(X), to_categorical(y, num_classes=3)
```

### Parámetros

- **PT_LIMIT = 0.02** (2%): Barrera de Take Profit
- **SL_LIMIT = 0.02** (2%): Barrera de Stop Loss
- **Clases**:
  - `0`: Baja (toca -2% primero)
  - `1`: Plano (expira el tiempo sin tocar barreras)
  - `2`: Sube (toca +2% primero)

---

## 4. Arquitectura de Modelos

### Configuración Base

```python
def construir_modelo_prado(config, input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # Capa recurrente (LSTM o GRU)
    CapaRecurrente = config['tipo_capa']
    model.add(CapaRecurrente(config['neuronas'], return_sequences=False))
    
    model.add(Dropout(config['dropout']))
    
    # Salida: 3 neuronas con Softmax (clasificación)
    model.add(Dense(3, activation='softmax'))
    
    optimizador = Adam(learning_rate=config['lr'])
    model.compile(optimizer=optimizador, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
```

### Hiperparámetros Probados

Se definieron **8 bancos de pruebas** según la ventana de entrada y salida:

#### Ventanas Cortas (Out: 1 o 5 días)
- **In:5 Corto**: GRU de 4-16 neuronas, LR 0.001
- **In:10 Corto**: LSTM/GRU de 4-16 neuronas, LR 0.0005-0.001
- **In:30 Corto**: LSTM/GRU de 4-128 neuronas, LR 0.00005-0.001
- **In:90 Corto**: LSTM/GRU de 8-16 neuronas, LR 0.0005-0.001

#### Ventanas Largas (Out: 30 o 90 días)
- **In:5 Largo**: LSTM de 8-128 neuronas, Dropout 0.1-0.3, LR 0.0005
- **In:10 Largo**: LSTM de 8-128 neuronas, Dropout 0.0-0.25, LR 0.00005-0.001
- **In:30 Largo**: LSTM/GRU de 16-128 neuronas, Dropout 0.0-0.3, LR 0.0005-0.001
- **In:90 Largo**: LSTM/GRU de 16-128 neuronas, Dropout 0.1-0.2, LR 0.00005-0.0005

### Estrategia de Entrenamiento

```python
# Early Stopping con paciencia de 20 épocas
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=20,
    restore_best_weights=True
)

# División cronológica (sin shuffle)
# 70% Train | 20% Validación | 10% Test
split_1 = int(len(X) * 0.70)
split_2 = int(len(X) * 0.90)
```

---

## 5. Baselines de Comparación

Para evaluar si los modelos realmente aprenden, se comparan contra 3 estrategias simples:

### 1. Always Buy (Naive)
Siempre predice **Clase 2 (Sube)**. Representa la creencia de que "el mercado siempre sube".

### 2. Always Flat
Siempre predice **Clase 1 (Plano)**. Representa la estrategia de "no hacer nada".

### 3. Most Frequent (Buy & Hold)
Predice la clase que más apareció en el **conjunto de entrenamiento**. Equivalente a seguir la tendencia histórica.

```python
def calcular_baselines_clasificacion(y_test, y_train):
    y_test_classes = np.argmax(y_test, axis=1)
    y_train_classes = np.argmax(y_train, axis=1)
    
    acc_always_buy = np.mean(y_test_classes == 2)
    acc_always_flat = np.mean(y_test_classes == 1)
    
    clase_mas_frecuente = np.bincount(y_train_classes).argmax()
    acc_most_frequent = np.mean(y_test_classes == clase_mas_frecuente)
    
    return acc_always_buy, acc_always_flat, acc_most_frequent
```

---

## 6. Resultados Finales

### Matriz de Accuracy en TEST

|        | Out_1  | Out_5  | Out_30 | Out_90 |
|--------|--------|--------|--------|--------|
| **In_5**  | 0.9736 | 0.3673 | 0.6295 | 0.5642 |
| **In_10** | 0.9735 | 0.3673 | 0.3677 | 0.5622 |
| **In_30** | 0.9732 | 0.3705 | 0.6335 | 0.5674 |
| **In_90** | 0.9771 | 0.3670 | 0.6279 | 0.5550 |

### Interpretación

#### Predicción a 1 día (Out_1)
- **Accuracy ~97%**: Parece excelente, pero...
- **Baseline "Most Frequent" = 97%**: El modelo simplemente aprendió que la clase "Plano" domina
- **Conclusión**: No hay poder predictivo real

#### Predicción a 5 días (Out_5)
- **Accuracy ~37%**: Peor que el azar (33.3%)
- **Baseline "Always Flat" = 41%**: Hacer nada es mejor
- **Conclusión**: Horizonte demasiado corto, dominado por ruido

#### Predicción a 30 días (Out_30)
- **Accuracy ~63%**: Mejor que el azar
- **Baseline "Always Buy" = 63%**: Empate técnico
- **Conclusión**: El modelo replica la tendencia alcista histórica

#### Predicción a 90 días (Out_90)
- **Accuracy ~56%**: Ligeramente mejor que el azar
- **Baseline "Always Buy" = 56%**: Empate técnico
- **Conclusión**: Similar a Out_30, sin ventaja real

---

## 7. Modelos Ganadores por Configuración

| Ventana | Modelo Ganador | Neuronas | Dropout | LR |
|---------|----------------|----------|---------|-----|
| In:5 Out:1 | GRU | 16 | 0.1 | 0.001 |
| In:5 Out:5 | GRU | 8 | 0.0 | 0.001 |
| In:5 Out:30 | LSTM | 128 | 0.3 | 0.0005 |
| In:5 Out:90 | LSTM | 128 | 0.3 | 0.0005 |
| In:10 Out:1 | GRU | 4 | 0.0 | 0.001 |
| In:10 Out:5 | GRU | 8 | 0.0 | 0.0005 |
| In:10 Out:30 | LSTM | 128 | 0.25 | 0.00005 |
| In:10 Out:90 | LSTM | 32 | 0.0 | 0.001 |
| In:30 Out:1 | LSTM | 16 | 0.0 | 0.0005 |
| In:30 Out:5 | GRU | 4 | 0.0 | 0.001 |
| In:30 Out:30 | LSTM | 64 | 0.0 | 0.001 |
| In:30 Out:90 | LSTM | 64 | 0.0 | 0.001 |
| In:90 Out:1 | LSTM | 16 | 0.0 | 0.001 |
| In:90 Out:5 | LSTM | 8 | 0.0 | 0.001 |
| In:90 Out:30 | GRU | 32 | 0.0 | 0.0005 |
| In:90 Out:90 | LSTM | 128 | 0.2 | 0.00005 |

### Patrones Observados

1. **Horizontes cortos (1-5 días)**: Modelos pequeños (4-16 neuronas), sin Dropout
2. **Horizontes largos (30-90 días)**: Modelos grandes (64-128 neuronas), con Dropout
3. **GRU vs LSTM**: GRU domina en horizontes cortos, LSTM en largos
4. **Learning Rate**: Más bajo (0.00005-0.0005) para modelos grandes

---

## 8. Conclusiones

### ✅ Implementación Técnica Exitosa

1. **Portfolio Dollar Bars**: Compresión de 85% manteniendo información relevante
2. **Diferenciación Fraccionaria**: Estacionariedad con memoria (d=0.45, ventana=238)
3. **Triple Barrier Method**: Etiquetado robusto para clasificación de trading
4. **Arquitecturas RNN**: 16 configuraciones probadas con búsqueda exhaustiva

### ⚠️ Limitaciones Predictivas

1. **Horizontes cortos (1-5 días)**: Dominados por ruido, sin poder predictivo
2. **Horizontes largos (30-90 días)**: Empate con "Always Buy", solo capturan tendencia alcista
3. **Overfitting**: Alta accuracy en Train (~75%), baja en Test (~37-63%)
4. **Desbalance de clases**: Clase "Plano" domina en horizontes cortos

### 🎯 Aplicabilidad a CNN

Esta metodología es **directamente transferible** a CNNs:

1. **Datos de entrada**: Usar las mismas Dollar Bars + FracDiff
2. **Etiquetado**: Mantener Triple Barrier Method
3. **Arquitectura**: Reemplazar LSTM/GRU por capas Conv1D
4. **Ventaja de CNN**: Mejor captura de patrones locales (ej: formaciones de velas)

### 📊 Próximos Pasos

1. **Balanceo de clases**: SMOTE o pesos en la función de pérdida
2. **Features adicionales**: Indicadores técnicos (RSI, MACD, Bollinger Bands)
3. **Ensemble**: Combinar predicciones de múltiples modelos
4. **Backtesting**: Evaluar en estrategia de trading real con costos de transacción

---

## Referencias

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Capítulo 2: Financial Data Structures (Dollar Bars)
- Capítulo 5: Fractional Differentiation
- Capítulo 3: Triple Barrier Method
