# Taller-Redes-Neuronales

# 📈 S&P 500 Forecasting: Deep Learning & Advanced Preprocessing (López de Prado)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Finance](https://img.shields.io/badge/Domain-Quantitative_Finance-green.svg)

## 📌 Resumen del Proyecto
Este proyecto de investigación evalúa la viabilidad de predecir los movimientos del índice S&P 500 utilizando redes neuronales profundas. El estudio se centra en contrastar el modelado clásico de series temporales (reloj cronológico y regresión) frente al ecosistema avanzado de preprocesamiento financiero propuesto por Marcos López de Prado (reloj de información y clasificación estocástica).

El objetivo principal no es solo la búsqueda de rentabilidad absoluta, sino la optimización del binomio rentabilidad-riesgo (reducción del *Máximo Drawdown*) en regímenes de alta volatilidad.

## 🧠 Arquitecturas Evaluadas
Se ha realizado un *Grid Search* exhaustivo cruzando ventanas de memoria histórica (`In: 5, 10, 30, 90 días`) con horizontes de predicción (`Out: 1, 5, 30, 90 días`). Se evaluaron 4 familias de modelos:
* **DNN (Redes Densas / MLP):** Evaluando topologías Shallow, Deep y optimizaciones con Dropout.
* **CNN (Redes Convolucionales 1D):** Extracción de patrones espaciales y filtrado de ruido con distintos tamaños de Kernel.
* **RNN (Recurrentes):** Modelos secuenciales LSTM y GRU.
* **Redes Mixtas (Híbridas):** Extracción de características convolucionales inyectadas en memorias recurrentes.

**🏆 Conclusión Arquitectónica:** Los resultados empíricos demostraron una hegemonía de las Redes Densas (MLP) sobre arquitecturas más complejas. En series financieras con una relación señal-ruido marginal, la ausencia de memoria secuencial estricta en las redes densas previene el *overfitting* estructural que penaliza a las LSTMs y GRUs.

## ⚙️ Metodología de Preprocesado (López de Prado)
Para aislar la señal del ruido del mercado, se implementó el siguiente *pipeline* de preprocesamiento avanzado:
1.  **Temporalidad por Volumen (Dollar Bars):** Sustitución del reloj cronológico por un reloj basado en eventos de liquidez.
2.  **FracDiff (Diferenciación Fraccional):** Estacionariedad preservando la memoria a largo plazo.
3.  **Filtrado RMT (Random Matrix Theory):** Limpieza de matrices de covarianza.
4.  **Triple Barrier Method (TBM):** Etiquetado dinámico de triple salida (Take Profit, Stop Loss, Tiempo expirado) basado en volatilidad local.
5.  **Purged K-Fold Cross Validation:** Prevención de *Data Leakage* eliminando el solapamiento de ventanas.

## 📊 Estrategias de Inversión y *Backtesting* (Test 2025)

El proyecto simula el rendimiento de carteras algorítmicas frente al índice *Benchmark* (S&P 500) implementando controles de exposición dinámica:

* **Estrategia Clásica (Regresión):** Utiliza un percentil umbral dinámico sobre el retorno continuo esperado para filtrar señales de bajo nivel, logrando batir el *Drawdown* del mercado.
* **Estrategia Avanzada (Triple Barrera):** Detecta y explota el "Colapso de Tasa Base" (*Base Rate Collapse*) de la red neuronal mediante un umbral de probabilidad ajustado al sesgo estadístico histórico, logrando fases críticas de preservación de capital (100% liquidez) durante caídas del índice.
