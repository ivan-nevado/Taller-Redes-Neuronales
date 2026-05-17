
'''
IMPORTANTE, ESTE CODIGO LO EJECUTO DENTRO DEL JUPITER DE LAS CARTERAS, PARA QUE PUEDA COGER LAS GRAFICAS QUE SE GENERAS
DE LAS CARTERAS.
EL PDF GENERADO TAMBIEN ESTA EN LA CARPTEA DE CARTERAS
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# ESTILO Y PALETA PROFESIONAL (Basado en Theme_4)
# ─────────────────────────────────────────────────────────────────────
DARK_BG = "#000000"
CARD_BG = "#1A1A1A"
ACCENT_GREEN = "#DEFF9A"
TEXT_WHITE = "#F5F5F5"
TEXT_SUB = "#DAFFDE"
GRID_COL = "#323232"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COL,
    "axes.labelcolor": TEXT_WHITE,
    "axes.titlecolor": ACCENT_GREEN,
    "text.color": TEXT_WHITE,
    "font.family": "sans-serif",
})

INPUT_WINDOWS = [5, 10, 30, 90]
OUTPUT_WINDOWS = [1, 5, 30, 90]
ROW_LABELS = [f"In: {w}d" for w in INPUT_WINDOWS]
COL_LABELS = [f"Out: {w}d" for w in OUTPUT_WINDOWS]

# ─────────────────────────────────────────────────────────────────────
# DATOS DE LA COMPETICIÓN (Mejor MAE entre todas las arquitecturas)
# ─────────────────────────────────────────────────────────────────────

# Matriz final de MAE (Valores seleccionados tras comparar Densa, CNN, RNN y Mixta)
best_mae_matrix = np.array([
    [0.01224, 0.00551, 0.00222, 0.00111], # In 5
    [0.01190, 0.00481, 0.00197, 0.00115], # In 10
    [0.01187, 0.00478, 0.00194, 0.00111], # In 30
    [0.01186, 0.00477, 0.00182, 0.00112]  # In 90
])

# Arquitectura ganadora por celda
winning_archs = [
    ["CNN-8F", "CNN-8F", "CNN-32F", "CNN-32F"],
    ["CNN-16F", "CNN-16F", "DNN-Deep", "DNN-Dropout"],
    ["CNN-64F", "CNN-64F", "RNN-LSTM", "DNN-Dropout"],
    ["RNN-LSTM", "Mixed-K5", "RNN-LSTM", "Mixed-K10"]
]

# ─────────────────────────────────────────────────────────────────────
# SLIDES DE LA PRESENTACIÓN
# ─────────────────────────────────────────────────────────────────────

def slide_portada(pdf):
    fig = plt.figure(figsize=(14, 8))
    fig.text(0.5, 0.6, "Forecasting SP500\nResultados de la Competición", 
             ha="center", fontsize=34, fontweight="bold", color=ACCENT_GREEN)
    fig.text(0.5, 0.45, "Análisis de Arquitecturas Híbridas y Preprocesado Avanzado", 
             ha="center", fontsize=18, color=TEXT_SUB)
    fig.text(0.5, 0.15, "MIAX 2025 | Taller de Redes Neuronales", 
             ha="center", fontsize=12, color=TEXT_WHITE)
    pdf.savefig(fig)
    plt.close()

def slide_matriz_ganadores(pdf):
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(best_mae_matrix, cmap="YlGn_r", aspect="auto")
    
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(COL_LABELS)
    ax.set_yticklabels(ROW_LABELS)
    ax.set_title("Matriz de Competición: Mejor MAE por Combinación", pad=20, fontsize=20)

    for i in range(4):
        for j in range(4):
            text = f"{best_mae_matrix[i, j]:.5f}\n({winning_archs[i][j]})"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="MAE (Menor es mejor)")
    fig.text(0.5, 0.05, "* Los valores representan el MAE mínimo alcanzado comparando DNN, CNN, RNN y Mixtas.", 
             ha="center", fontsize=10, style="italic")
    pdf.savefig(fig)
    plt.close()

def slide_reflexion_modelos(pdf):
    fig = plt.figure(figsize=(14, 8))
    fig.text(0.1, 0.85, "Reflexión: Utilidad según Ventanas", fontsize=24, color=ACCENT_GREEN)
    
    temas = [
        ("Corto Plazo (Out 1-5d):", "Las CNN y RNN (LSTM) dominan. La capacidad de extraer patrones locales supera a las redes densas."),
        ("Largo Plazo (Out 30-90d):", "Los modelos Mixtos y DNN con alto Dropout ganan estabilidad al filtrar el ruido diario."),
        ("Profundidad de Entrada:", "In 90d ofrece la mejor convergencia, pero exige Kernels más grandes (K5-K10) para evitar overfitting."),
        ("Conclusión:", "No existe una arquitectura única; la regularización es más crítica que la complejidad de la red.")
    ]
    
    y = 0.65
    for titulo, desc in temas:
        fig.text(0.12, y, f"● {titulo}", fontsize=16, fontweight="bold", color=TEXT_SUB)
        fig.text(0.14, y-0.05, desc, fontsize=14, color=TEXT_WHITE)
        y -= 0.18
        
    pdf.savefig(fig)
    plt.close()

def slide_preprocesado_prado(pdf):
    fig = plt.figure(figsize=(14, 8))
    fig.text(0.1, 0.88, "Técnicas de Preprocesado (López de Prado)", fontsize=22, color=ACCENT_GREEN)
    
    tecnicas = [
        ("T1 - Dollar Bars", "Sincroniza los datos con el flujo de capital en lugar del reloj, reduciendo la heterocedasticidad."),
        ("T2 - FracDiff (FFD)", "Elimina la no-estacionariedad preservando la memoria máxima de la serie temporal (d ≈ 0.4)."),
        ("T3 - Random Matrix Theory", "Limpia la matriz de covarianza eliminando eigenvalores asociados al ruido estadístico."),
        ("T4 - Triple Barrier Method", "Etiquetado dinámico basado en volatilidad real (Profit, Loss o Tiempo), evitando el sesgo del signo."),
        ("T5 - Purged K-Fold", "Validación cruzada que elimina el solapamiento temporal, garantizando resultados sin data leakage.")
    ]
    
    y = 0.72
    for t, d in tecnicas:
        rect = FancyBboxPatch((0.1, y-0.08), 0.8, 0.1, boxstyle="round,pad=0.02", facecolor=CARD_BG, edgecolor=ACCENT_GREEN)
        fig.add_artist(rect)
        fig.text(0.15, y-0.02, t, fontsize=15, fontweight="bold", color=ACCENT_GREEN)
        fig.text(0.15, y-0.06, d, fontsize=12, color=TEXT_WHITE)
        y -= 0.14

    pdf.savefig(fig)
    plt.close()

def slide_carteras_2025(pdf):
    # Creamos una diapositiva ancha con 2 paneles (1 fila, 2 columnas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # ─────────────────────────────────────────────────────
    # PANEL 1: IA CLÁSICA (REGRESIÓN - SIN PRADO)
    # ─────────────────────────────────────────────────────
    ax1.plot(cap_bh_reg, color=TEXT_WHITE, linestyle="--", alpha=0.6, label="Benchmark (Mercado SP500)")
    ax1.plot(cap_reg, color="orange", linewidth=2, alpha=0.8, label="IA Clásica (Regresión)")
    
    rent_bh_reg = ((cap_bh_reg[-1] - 10000) / 10000) * 100
    rent_reg = ((cap_reg[-1] - 10000) / 10000) * 100
    txt_reg = f"Benchmark: {cap_bh_reg[-1]:.2f}€ ({rent_bh_reg:.2f}%)\nIA Regresión: {cap_reg[-1]:.2f}€ ({rent_reg:.2f}%)"
    
    ax1.set_title("IA Clásica vs Mercado (Reloj Cronológico)", fontsize=13, color="orange", fontweight="bold")
    ax1.set_xlabel("Horizonte Temporal (Días Naturales)", fontsize=10)
    ax1.set_ylabel("Capital (€)", fontsize=10)
    ax1.legend(facecolor=CARD_BG, edgecolor=GRID_COL, fontsize=9)
    ax1.grid(color=GRID_COL, linestyle=":")
    ax1.text(0.05, 0.05, txt_reg, transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor=DARK_BG, alpha=0.85, edgecolor=GRID_COL))
    
    # ─────────────────────────────────────────────────────
    # PANEL 2: IA AVANZADA (LÓPEZ DE PRADO - CON PRADO)
    # ─────────────────────────────────────────────────────
    ax2.plot(cap_bh_pra, color=TEXT_WHITE, linestyle="--", alpha=0.6, label="Benchmark (Mercado SP500)")
    
    # Eje X natural: Dollar bars
    ax2.step(range(len(cap_pra)), cap_pra, color=ACCENT_GREEN, linewidth=2, label="IA Avanzada (Prado)")
    
    rent_bh_pra = ((cap_bh_pra[-1] - 10000) / 10000) * 100
    rent_pra = ((cap_pra[-1] - 10000) / 10000) * 100
    txt_pra = f"Benchmark: {cap_bh_pra[-1]:.2f}€ ({rent_bh_pra:.2f}%)\nIA Prado: {cap_pra[-1]:.2f}€ ({rent_pra:.2f}%)"
    
    ax2.set_title("IA Avanzada vs Mercado (Reloj de Información)", fontsize=13, color=ACCENT_GREEN, fontweight="bold")
    ax2.set_xlabel("Horizonte Temporal (Dollar Bars)", fontsize=10)
    ax2.set_ylabel("Capital (€)", fontsize=10)
    ax2.legend(facecolor=CARD_BG, edgecolor=GRID_COL, fontsize=9)
    ax2.grid(color=GRID_COL, linestyle=":")
    ax2.text(0.05, 0.05, txt_pra, transform=ax2.transAxes, fontsize=10, bbox=dict(facecolor=DARK_BG, alpha=0.85, edgecolor=GRID_COL))
    
    # ─────────────────────────────────────────────────────
    # TÍTULO Y CONCLUSIÓN GENERAL DE LA DIAPOSITIVA
    # ─────────────────────────────────────────────────────
    fig.suptitle("Backtesting 2025: Paradigma de Tiempo Cronológico vs Tiempo de Información", fontsize=18, fontweight="bold", y=0.96)
    
    analisis_txt = (
        "*Nota Metodológica: Los Benchmark difieren porque miden espacios distintos. La IA Clásica opera sobre días de calendario continuo.\n"
        "La IA Avanzada (Prado) opera sobre 'Dollar Bars' (eventos de volumen), contrayendo o dilatando el tiempo según la liquidez del mercado.\n"
        "Resultado: La IA Clásica sucumbe al ruido y replica el mercado. La IA de Prado se defiende en efectivo, logrando un 'Drawdown' del 0%."
    )
        
    fig.text(0.5, 0.08, analisis_txt, ha="center", fontsize=11, bbox=dict(facecolor=CARD_BG, alpha=0.9, edgecolor=GRID_COL))
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.92])
    pdf.savefig(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────
# GENERACIÓN DEL PDF
# ─────────────────────────────────────────────────────────────────────

filename = "Presentacion_Final_Redes_SP500.pdf"
with PdfPages(filename) as pdf:
    print("Mejorando PDF de presentación...")
    slide_portada(pdf)
    slide_matriz_ganadores(pdf)
    slide_reflexion_modelos(pdf)
    slide_preprocesado_prado(pdf)
    slide_carteras_2025(pdf)

print(f"Éxito: Presentación generada en {filename}")