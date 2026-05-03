# 📋 PLAN DE ACCIÓN: REDES NEURONALES CONVOLUCIONALES (CNN) PARA FORECASTING FINANCIERO

## 🎯 OBJETIVO PRINCIPAL

Implementar y evaluar modelos de **Redes Neuronales Convolucionales (CNN)** para predicción de series temporales financieras, siguiendo la misma estructura que las RNN pero adaptando las arquitecturas para aprovechar las capacidades de las convoluciones 1D.

---

## 📊 CONTEXTO DEL PROYECTO

### Datos disponibles:
- **23 activos del SP500** desde 1960 (~16,000 días)
- **Retornos logarítmicos** calculados
- **16 combinaciones** de ventanas temporales:
  - Entrada: [5, 10, 30, 90] días
  - Salida: [1, 5, 30, 90] días

### Métricas:
- **MAE (Mean Absolute Error)** como métrica principal
- Comparación con **baselines**: Naive, SMA, Buy & Hold

---

## 🗂️ ESTRUCTURA DE ARCHIVOS PROPUESTA

```
cnn/
│
├── PLAN_DE_ACCION_CNN.md          # Este archivo
├── datos_preparacion.ipynb         # Carga y preparación de datos
├── cnn_1d_basico.ipynb            # Modelos CNN 1D simples
├── cnn_avanzado.ipynb             # Arquitecturas CNN más complejas
├── cnn_hibrido.ipynb              # CNN + Dense / CNN + LSTM
├── resultados_finales.ipynb       # Comparación y análisis
│
├── modelos/                        # Modelos guardados (.h5 o .keras)
│   ├── cnn_in5_out1_best.keras
│   ├── cnn_in10_out5_best.keras
│   └── ...
│
├── graficas_convergencia/          # Curvas de entrenamiento
│   ├── cnn_in5_out1.png
│   └── ...
│
├── resultados/                     # CSVs y matrices de resultados
│   ├── matriz_mae_cnn.csv
│   ├── comparacion_rnn_vs_cnn.csv
│   └── mejores_modelos.csv
│
└── utils/                          # Funciones auxiliares
    ├── data_loader.py
    ├── model_builder.py
    └── evaluator.py
```

---

## 🚀 FASES DEL PROYECTO

---

## 📌 FASE 1: PREPARACIÓN DE DATOS (datos_preparacion.ipynb)

### Objetivos:
- Reutilizar el código de carga de datos de RNN
- Verificar que las ventanas temporales sean compatibles con CNN 1D
- Crear funciones auxiliares reutilizables

### Tareas:

#### 1.1 Descarga y carga de datos
```python
# Descargar datos de Yahoo Finance
# Calcular retornos logarítmicos
# Verificar forma: (16189, 23)
```

#### 1.2 Función de creación de ventanas
```python
def create_time_series_data(data, input_window_size, output_window_size):
    """
    Crea ventanas temporales para CNN 1D
    Input shape: (samples, timesteps, features)
    Output shape: (samples, features)
    """
    # Reutilizar código de RNN
    pass
```

#### 1.3 Split cronológico
```python
# 80% Train
# 10% Validation
# 10% Test
# IMPORTANTE: Sin shuffle para mantener orden temporal
```

#### 1.4 Verificación de shapes
```python
# Para CNN 1D necesitamos: (batch, timesteps, channels)
# Ejemplo: (12000, 30, 23) para ventana de entrada de 30 días
```

**Entregable:** Notebook con datos listos y funciones auxiliares exportadas a `utils/data_loader.py`

---

## 📌 FASE 2: CNN 1D BÁSICO (cnn_1d_basico.ipynb)

### Objetivos:
- Implementar arquitecturas CNN 1D simples
- Establecer baseline de rendimiento para CNN
- Probar diferentes configuraciones de filtros y kernels

### Arquitecturas a implementar:

#### 2.1 CNN Simple (1 capa convolucional)
```python
model = Sequential([
    Input(shape=(input_window, 23)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(23)
])
```

#### 2.2 CNN con MaxPooling
```python
model = Sequential([
    Input(shape=(input_window, 23)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(23)
])
```

#### 2.3 CNN con Dropout
```python
model = Sequential([
    Input(shape=(input_window, 23)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(23)
])
```

### Hiperparámetros a explorar:

| Parámetro | Valores a probar |
|-----------|------------------|
| **Filtros** | [32, 64, 128] |
| **Kernel size** | [3, 5, 7] |
| **Dropout** | [0.1, 0.2, 0.3] |
| **Learning rate** | [0.001, 0.0005, 0.0001] |

### Proceso de entrenamiento:

```python
# Para cada combinación de ventanas (16 total)
for input_w in [5, 10, 30, 90]:
    for output_w in [1, 5, 30, 90]:
        # 1. Crear datos
        X, y = create_time_series_data(returns, input_w, output_w)
        
        # 2. Split train/val/test
        
        # 3. Probar 3-5 configuraciones
        
        # 4. Seleccionar mejor modelo (menor val_loss)
        
        # 5. Evaluar en test
        
        # 6. Guardar resultados y gráficas
```

**Entregable:** 
- Notebook con 16 modelos CNN básicos entrenados
- Matriz de resultados MAE (4x4)
- Gráficas de convergencia

---

## 📌 FASE 3: CNN AVANZADO (cnn_avanzado.ipynb)

### Objetivos:
- Implementar arquitecturas CNN más profundas
- Explorar técnicas de regularización avanzadas
- Mejorar el rendimiento sobre los modelos básicos

### Arquitecturas a implementar:

#### 3.1 CNN Multi-capa (Stacked CNN)
```python
model = Sequential([
    Input(shape=(input_window, 23)),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(23)
])
```

#### 3.2 CNN con Residual Connections (inspirado en ResNet)
```python
def residual_block(x, filters):
    shortcut = x
    x = Conv1D(filters, 3, padding='same', activation='relu')(x)
    x = Conv1D(filters, 3, padding='same')(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

# Modelo con bloques residuales
```

#### 3.3 CNN con Dilated Convolutions
```python
model = Sequential([
    Input(shape=(input_window, 23)),
    Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu', padding='causal'),
    Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu', padding='causal'),
    Conv1D(filters=64, kernel_size=3, dilation_rate=4, activation='relu', padding='causal'),
    GlobalAveragePooling1D(),
    Dense(23)
])
```

#### 3.4 CNN con Batch Normalization
```python
model = Sequential([
    Input(shape=(input_window, 23)),
    Conv1D(filters=64, kernel_size=3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv1D(filters=128, kernel_size=3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    GlobalAveragePooling1D(),
    Dense(23)
])
```

### Técnicas de regularización:

- **Dropout** (0.2 - 0.5)
- **L2 Regularization** (kernel_regularizer)
- **Batch Normalization**
- **Early Stopping** (patience=10)
- **ReduceLROnPlateau** (reducir LR cuando val_loss se estanca)

**Entregable:**
- Notebook con arquitecturas avanzadas
- Comparación de rendimiento vs CNN básico
- Análisis de qué técnicas funcionan mejor

---

## 📌 FASE 4: MODELOS HÍBRIDOS (cnn_hibrido.ipynb)

### Objetivos:
- Combinar CNN con otras arquitecturas
- Aprovechar lo mejor de cada tipo de capa
- Explorar arquitecturas innovadoras

### Arquitecturas híbridas:

#### 4.1 CNN + Dense (Feature Extraction + MLP)
```python
model = Sequential([
    Input(shape=(input_window, 23)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(23)
])
```

#### 4.2 CNN + LSTM (Convolutional LSTM)
```python
model = Sequential([
    Input(shape=(input_window, 23)),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    LSTM(50, return_sequences=False),
    Dense(23)
])
```

#### 4.3 CNN + GRU
```python
model = Sequential([
    Input(shape=(input_window, 23)),
    Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    GRU(64),
    Dense(23)
])
```

#### 4.4 Multi-Scale CNN (diferentes kernel sizes en paralelo)
```python
from tensorflow.keras.layers import Concatenate

input_layer = Input(shape=(input_window, 23))

# Branch 1: kernel size 3
conv1 = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
pool1 = GlobalAveragePooling1D()(conv1)

# Branch 2: kernel size 5
conv2 = Conv1D(32, 5, activation='relu', padding='same')(input_layer)
pool2 = GlobalAveragePooling1D()(conv2)

# Branch 3: kernel size 7
conv3 = Conv1D(32, 7, activation='relu', padding='same')(input_layer)
pool3 = GlobalAveragePooling1D()(conv3)

# Concatenar
concat = Concatenate()([pool1, pool2, pool3])
output = Dense(23)(concat)

model = Model(inputs=input_layer, outputs=output)
```

**Entregable:**
- Notebook con modelos híbridos
- Comparación CNN puro vs híbridos
- Análisis de complejidad vs rendimiento

---

## 📌 FASE 5: RESULTADOS FINALES Y COMPARACIÓN (resultados_finales.ipynb)

### Objetivos:
- Consolidar todos los resultados
- Comparar CNN vs RNN vs Baselines
- Generar visualizaciones para la presentación

### Tareas:

#### 5.1 Matriz de resultados consolidada

```python
# Crear DataFrame con todos los modelos
resultados = pd.DataFrame({
    'Ventana_In': [...],
    'Ventana_Out': [...],
    'MAE_Naive': [...],
    'MAE_SMA': [...],
    'MAE_BuyHold': [...],
    'MAE_RNN_Best': [...],
    'MAE_CNN_Basic': [...],
    'MAE_CNN_Advanced': [...],
    'MAE_CNN_Hybrid': [...]
})
```

#### 5.2 Visualizaciones clave

**Gráfica 1: Heatmap de MAE por ventana**
```python
import seaborn as sns

# Heatmap para cada tipo de modelo
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.heatmap(matriz_cnn_basic, annot=True, fmt='.4f', ax=axes[0,0])
sns.heatmap(matriz_cnn_advanced, annot=True, fmt='.4f', ax=axes[0,1])
sns.heatmap(matriz_cnn_hybrid, annot=True, fmt='.4f', ax=axes[1,0])
sns.heatmap(matriz_rnn, annot=True, fmt='.4f', ax=axes[1,1])
```

**Gráfica 2: Comparación directa RNN vs CNN**
```python
# Barplot comparando mejor RNN vs mejor CNN para cada ventana
```

**Gráfica 3: Mejora sobre baselines**
```python
# Porcentaje de mejora de cada modelo sobre Buy & Hold
mejora_rnn = (mae_bh - mae_rnn) / mae_bh * 100
mejora_cnn = (mae_bh - mae_cnn) / mae_bh * 100
```

#### 5.3 Análisis estadístico

- **Test de Wilcoxon** para comparar RNN vs CNN
- **Análisis de varianza** del MAE
- **Identificar en qué ventanas CNN supera a RNN**

#### 5.4 Tabla resumen para GitHub

```markdown
| Ventana In | Ventana Out | Mejor Modelo | MAE Test | Mejora vs Baseline |
|------------|-------------|--------------|----------|-------------------|
| 5          | 1           | LSTM-128     | 0.0123   | +0.5%            |
| 5          | 5           | CNN-Hybrid   | 0.0055   | +1.2%            |
| ...        | ...         | ...          | ...      | ...              |
```

**Entregable:**
- Notebook con análisis completo
- Gráficas de alta calidad para presentación
- Tabla resumen en formato Markdown
- Conclusiones y recomendaciones

---

## 📌 FASE 6: DOCUMENTACIÓN Y PRESENTACIÓN

### 6.1 README.md principal del proyecto

```markdown
# Taller: Redes Neuronales para Forecasting Financiero

## Resumen
Implementación y comparación de RNN (LSTM/GRU) y CNN para predicción de retornos de 23 activos del SP500.

## Resultados principales
- Mejor modelo RNN: LSTM-128 (MAE: 0.0012 en ventana 5-90)
- Mejor modelo CNN: CNN-Hybrid (MAE: 0.0011 en ventana 30-90)
- CNN supera a RNN en ventanas largas (30-90 días)
- RNN mejor para predicciones a corto plazo (1-5 días)

## Estructura del proyecto
...
```

### 6.2 Presentación (PDF)

**Diapositivas sugeridas:**

1. **Introducción**
   - Objetivo del taller
   - Datos utilizados
   - Métricas de evaluación

2. **Metodología**
   - Arquitecturas implementadas
   - Ventanas temporales
   - Proceso de entrenamiento

3. **Resultados: RNN**
   - Matriz de MAE
   - Mejores modelos
   - Curvas de convergencia

4. **Resultados: CNN**
   - Matriz de MAE
   - Comparación CNN básico vs avanzado
   - Modelos híbridos

5. **Comparación RNN vs CNN**
   - Gráficas comparativas
   - Análisis estadístico
   - Ventajas y desventajas

6. **Conclusiones**
   - ¿Cuándo usar RNN?
   - ¿Cuándo usar CNN?
   - Trabajo futuro

7. **Carteras 2025** (Parte de investigación)
   - Cartera sin predicciones
   - Cartera con predicciones (mejor modelo 90 días)
   - Comparación de rendimiento

---

## 🎯 CHECKLIST DE ENTREGABLES

### GitHub:
- [ ] Código completo y comentado
- [ ] README.md con instrucciones
- [ ] Matrices de resultados (CSV)
- [ ] Gráficas de convergencia (PNG)
- [ ] Modelos guardados (.keras)
- [ ] Requirements.txt con dependencias

### Presentación (PDF):
- [ ] Matriz de resultados competición
- [ ] Reflexión sobre modelos útiles
- [ ] Técnicas de preprocesado
- [ ] Resultados de carteras 2025

---

## 📚 RECURSOS Y REFERENCIAS

### Papers relevantes:
- **WaveNet** (van den Oord et al., 2016) - Dilated convolutions
- **Temporal Convolutional Networks** (Bai et al., 2018)
- **CNN for Time Series** (Zhao et al., 2017)

### Tutoriales:
- [Keras Conv1D documentation](https://keras.io/api/layers/convolution_layers/convolution1d/)
- [Time Series Forecasting with CNNs](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/)

---

## ⏱️ ESTIMACIÓN DE TIEMPO

| Fase | Tiempo estimado | Prioridad |
|------|----------------|-----------|
| Fase 1: Preparación datos | 2-3 horas | 🔴 Alta |
| Fase 2: CNN básico | 4-6 horas | 🔴 Alta |
| Fase 3: CNN avanzado | 6-8 horas | 🟡 Media |
| Fase 4: Modelos híbridos | 4-6 horas | 🟡 Media |
| Fase 5: Resultados finales | 3-4 horas | 🔴 Alta |
| Fase 6: Documentación | 2-3 horas | 🔴 Alta |
| **TOTAL** | **21-30 horas** | |

---

## 🚨 NOTAS IMPORTANTES

### Diferencias clave CNN vs RNN:

| Aspecto | RNN (LSTM/GRU) | CNN 1D |
|---------|----------------|--------|
| **Memoria** | Memoria explícita (hidden state) | Memoria implícita (receptive field) |
| **Paralelización** | Secuencial (más lento) | Paralelo (más rápido) |
| **Ventanas largas** | Puede olvidar información lejana | Dilated convolutions capturan largo plazo |
| **Interpretabilidad** | Difícil | Filtros pueden visualizarse |
| **Parámetros** | Más parámetros | Menos parámetros (compartidos) |

### Ventajas de CNN para series temporales:
✅ **Más rápido** de entrenar (paralelización)
✅ **Menos propenso a overfitting** (menos parámetros)
✅ **Captura patrones locales** muy bien
✅ **Dilated convolutions** para largo plazo sin perder resolución

### Desventajas de CNN:
❌ No tiene memoria explícita como RNN
❌ Requiere ventanas de entrada fijas
❌ Puede perder información de orden temporal

---

## 🎓 CRITERIOS DE ÉXITO

### Mínimo viable:
- ✅ 16 modelos CNN entrenados (uno por combinación de ventanas)
- ✅ Matriz de resultados MAE
- ✅ Comparación con baselines
- ✅ Gráficas de convergencia

### Objetivo ideal:
- ✅ Todo lo anterior +
- ✅ Modelos CNN avanzados y híbridos
- ✅ Comparación exhaustiva RNN vs CNN
- ✅ Análisis estadístico robusto
- ✅ Implementación de carteras 2025
- ✅ Documentación completa y profesional

---

## 📞 PRÓXIMOS PASOS INMEDIATOS

1. **Crear estructura de carpetas**
2. **Copiar y adaptar código de carga de datos de RNN**
3. **Implementar primer modelo CNN básico**
4. **Verificar que funciona con una ventana (ej: 5-1)**
5. **Automatizar para las 16 combinaciones**

---

**Fecha de creación:** 2025-01-XX
**Última actualización:** 2025-01-XX
**Autor:** Iván (MIAX - Taller Redes Neuronales)

---

## 🤝 ¿NECESITAS AYUDA?

Si tienes dudas sobre:
- Implementación de arquitecturas específicas
- Debugging de errores
- Interpretación de resultados
- Optimización de hiperparámetros

**¡Pregunta! Estoy aquí para ayudarte.** 🚀
