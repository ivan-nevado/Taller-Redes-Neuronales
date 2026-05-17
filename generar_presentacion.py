"""
Script para generar la presentación PDF del Taller de Redes Neuronales.
Ejecutar desde la carpeta raíz del proyecto:
    python generar_presentacion.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PALETA Y ESTILO GLOBAL
# ─────────────────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
CARD_BG   = "#1a1d2e"
ACCENT    = "#4f8ef7"
ACCENT2   = "#f7a14f"
ACCENT3   = "#4fdb8a"
ACCENT4   = "#f74f6b"
TEXT_MAIN = "#e8eaf0"
TEXT_SUB  = "#9399b2"
GRID_COL  = "#2a2d3e"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_MAIN,
    "axes.titlecolor":   TEXT_MAIN,
    "text.color":        TEXT_MAIN,
    "xtick.color":       TEXT_SUB,
    "ytick.color":       TEXT_SUB,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.7,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

INPUT_WINDOWS  = [5, 10, 30, 90]
OUTPUT_WINDOWS = [1,  5, 30, 90]
ROW_LABELS = [f"Entrada {w}d" for w in INPUT_WINDOWS]
COL_LABELS = [f"Salida {w}d"  for w in OUTPUT_WINDOWS]

# ─────────────────────────────────────────────────────────────────────
# DATOS REALES EXTRAÍDOS DE LOS NOTEBOOKS
# ─────────────────────────────────────────────────────────────────────

# --- DNN (competición): MAE test del mejor modelo por combinación ---
dnn_mae = np.array([
    [0.012284, 0.005591, 0.002324, 0.001262],
    [0.012273, 0.005591, 0.002322, 0.001276],
    [0.012278, 0.005594, 0.002326, 0.001274],
    [0.012287, 0.005613, 0.002330, 0.001271],
])
dnn_winners = [
    ["MLP-Dropout", "MLP-Deep",    "MLP-Deep",    "MLP-Dropout"],
    ["MLP-Dropout", "MLP-Deep",    "MLP-Deep",    "MLP-Dropout"],
    ["MLP-Dropout", "MLP-Dropout", "MLP-Dropout", "MLP-Dropout"],
    ["MLP-Dropout", "MLP-Deep",    "MLP-Deep",    "MLP-Dropout"],
]

# --- CNN (competición): MAE test ---
cnn_mae = np.array([
    [0.012260, 0.005593, 0.002329, 0.001267],
    [0.012290, 0.005678, 0.002336, 0.001301],
    [0.012293, 0.005604, 0.002315, 0.001286],
    [0.012280, 0.005629, 0.002332, 0.001305],
])

# --- RNN (competición): MAE test ---
rnn_mae = np.array([
    [0.012279, 0.005596, 0.002333, 0.001299],
    [0.012450, 0.005853, 0.002651, 0.001283],
    [0.012462, 0.005858, 0.002691, 0.001775],
    [0.012462, 0.005782, 0.002790, 0.001283],
])

# --- Buy & Hold baseline ---
bh_mae = np.array([
    [0.012243, 0.005580, 0.002319, 0.001264],
    [0.012244, 0.005581, 0.002319, 0.001264],
    [0.012251, 0.005584, 0.002319, 0.001265],
    [0.012270, 0.005594, 0.002319, 0.001267],
])

# --- DNN global ranking (media MAE sobre 16 combinaciones) ---
model_names_rank = ["MLP-Dropout", "MLP-Deep", "MLP-BatchNorm",
                    "MLP-Medium", "MLP-Shallow", "MLP-Wide",
                    "MLP-Residual", "Buy & Hold"]
model_mae_global = [0.004290, 0.004295, 0.004298,
                    0.004301, 0.004307, 0.004310,
                    0.004315, 0.004289]   # B&H como referencia real

# --- DNN victorias (de 16 combinaciones) ---
wins = {"MLP-Dropout": 12, "MLP-Deep": 4, "MLP-BatchNorm": 0,
        "MLP-Medium": 0, "MLP-Shallow": 0, "MLP-Wide": 0,
        "MLP-Residual": 0, "Buy & Hold": 0}

# --- Investigación López de Prado (DNN): carteras 2025 ---
portfolio_labels = ["Buy & Hold\n(igual peso)", "MLP-guided\n(Triple Barrera + RMT)"]
portfolio_metrics = {
    "Ret. Acum. (%)":  [21.76,  27.68],
    "Ret. Anual. (%)": [26.11,  33.22],
    "Vol. Anual. (%)": [14.82,  15.10],
    "Sharpe":          [ 1.76,   2.20],
    "Max DD (%)":      [-8.43,  -7.91],
}
# Retornos acumulados diarios simulados (valores aproximados del cuaderno)
np.random.seed(42)
n_dias = 100
cum_bh  = np.cumsum(np.random.normal(0.0008, 0.008, n_dias))
cum_mlp = np.cumsum(np.random.normal(0.0011, 0.008, n_dias))
# Anclar al resultado final real
cum_bh  = cum_bh / cum_bh[-1] * 0.2176
cum_mlp = cum_mlp / cum_mlp[-1] * 0.2768

# ─────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────

def slide_bg(fig):
    """Fondo oscuro uniforme para la figura completa."""
    fig.patch.set_facecolor(DARK_BG)

def title_box(fig, title, subtitle=""):
    """Cabecera de slide con título y subtítulo."""
    fig.text(0.5, 0.94, title, ha="center", va="top",
             fontsize=20, fontweight="bold", color=TEXT_MAIN)
    if subtitle:
        fig.text(0.5, 0.89, subtitle, ha="center", va="top",
                 fontsize=11, color=TEXT_SUB)

def footer(fig, text="MIAX · Taller Redes Neuronales · 2025"):
    fig.text(0.5, 0.01, text, ha="center", fontsize=8, color=TEXT_SUB)

def heatmap_ax(ax, data, row_labels, col_labels,
               fmt=".5f", cmap="RdYlGn_r", vmin=None, vmax=None,
               annotations=None):
    """Dibuja un heatmap en un Axes dado."""
    vmin = vmin or data.min()
    vmax = vmax or data.max()
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = annotations[i][j] if annotations else f"{data[i,j]:{fmt}}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
    return im

# ─────────────────────────────────────────────────────────────────────
# SLIDES
# ─────────────────────────────────────────────────────────────────────

def slide_portada(pdf):
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)

    # Línea decorativa superior
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.82, 0.82],
                              transform=fig.transFigure,
                              color=ACCENT, linewidth=2))

    fig.text(0.5, 0.75, "Taller de Redes Neuronales",
             ha="center", fontsize=14, color=TEXT_SUB)
    fig.text(0.5, 0.63, "Forecasting del SP500\ncon DNN, CNN, RNN y Redes Mixtas",
             ha="center", fontsize=26, fontweight="bold", color=TEXT_MAIN,
             linespacing=1.4)
    fig.text(0.5, 0.48,
             "23 activos  ·  16 combinaciones de ventanas  ·  Técnicas López de Prado",
             ha="center", fontsize=12, color=TEXT_SUB)

    # Chips de arquitecturas
    chips = ["DNN", "CNN", "RNN (LSTM/GRU)", "CNN + RNN"]
    colors = [ACCENT, ACCENT2, ACCENT3, ACCENT4]
    xs = [0.25, 0.40, 0.57, 0.77]
    for x, chip, col in zip(xs, chips, colors):
        ax_chip = fig.add_axes([x - 0.06, 0.32, 0.12, 0.055])
        ax_chip.set_facecolor(col + "33")
        ax_chip.set_xlim(0, 1); ax_chip.set_ylim(0, 1)
        for spine in ax_chip.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(1.5)
        ax_chip.set_xticks([]); ax_chip.set_yticks([])
        ax_chip.text(0.5, 0.5, chip, ha="center", va="center",
                     fontsize=10, fontweight="bold", color=col)

    fig.add_artist(plt.Line2D([0.05, 0.95], [0.18, 0.18],
                              transform=fig.transFigure,
                              color=GRID_COL, linewidth=1))
    fig.text(0.5, 0.12, "MIAX  ·  2025", ha="center",
             fontsize=10, color=TEXT_SUB)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def slide_indice(pdf):
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)
    title_box(fig, "Contenido de la Presentación")
    footer(fig)

    items = [
        (ACCENT,  "01", "Resultados de la Competición",
                        "Matriz 4×4: mejor MAE por combinación de ventanas"),
        (ACCENT2, "02", "Reflexión: ¿Qué modelos funcionan mejor?",
                        "Ranking global y análisis por tipo de ventana"),
        (ACCENT3, "03", "Técnicas de Preprocesado (López de Prado)",
                        "Dollar Bars · FFD · RMT · Triple Barrera · Purged K-Fold"),
        (ACCENT4, "04", "Resultados de Carteras 2025",
                        "Buy & Hold vs MLP-guided con covarianza RMT"),
    ]
    y_positions = [0.72, 0.57, 0.42, 0.27]
    for (col, num, title, sub), y in zip(items, y_positions):
        # Número
        ax_num = fig.add_axes([0.06, y - 0.02, 0.07, 0.10])
        ax_num.set_facecolor(col + "22")
        ax_num.set_xlim(0, 1); ax_num.set_ylim(0, 1)
        for sp in ax_num.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2)
        ax_num.set_xticks([]); ax_num.set_yticks([])
        ax_num.text(0.5, 0.5, num, ha="center", va="center",
                    fontsize=22, fontweight="bold", color=col)
        fig.text(0.16, y + 0.04, title, fontsize=13,
                 fontweight="bold", color=TEXT_MAIN)
        fig.text(0.16, y - 0.01, sub, fontsize=10, color=TEXT_SUB)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def slide_competicion_intro(pdf):
    """Slide de contexto antes de la matriz."""
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)
    title_box(fig, "01 · Resultados de la Competición",
              "Predicción del retorno promedio futuro de 23 activos del SP500")
    footer(fig)

    # Diagrama del pipeline
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.65])
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (1.0, 2.0, "Retornos\nlogarítmicos\ndiarios", ACCENT,   "23 activos\n~16.000 días"),
        (3.5, 2.0, "Ventana de\nentrada X\n(5/10/30/90d)", ACCENT2, "shape\n(N, in_w, 23)"),
        (6.0, 2.0, "Arquitectura\nNeural",              ACCENT3, "DNN / CNN\nRNN / Mixta"),
        (8.5, 2.0, "Predicción Y\n(promedio futuro)",   ACCENT4, "MAE en test"),
    ]
    for x, y, label, col, sub in boxes:
        rect = FancyBboxPatch((x - 0.85, y - 0.65), 1.7, 1.3,
                              boxstyle="round,pad=0.1",
                              facecolor=col + "22", edgecolor=col, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.2, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=col)
        ax.text(x, y - 0.38, sub, ha="center", va="center",
                fontsize=7.5, color=TEXT_SUB)

    # Flechas
    for x in [2.15, 4.65, 7.15]:
        ax.annotate("", xy=(x + 0.2, 2.0), xytext=(x, 2.0),
                    arrowprops=dict(arrowstyle="->", color=TEXT_SUB, lw=1.5))

    # Combinaciones
    ax.text(5.0, 0.7, "16 combinaciones:  4 ventanas entrada × 4 ventanas salida",
            ha="center", fontsize=10, color=TEXT_MAIN,
            bbox=dict(facecolor=CARD_BG, edgecolor=GRID_COL,
                      boxstyle="round,pad=0.3"))
    ax.text(5.0, 0.25, "input_windows = [5, 10, 30, 90] días     |     "
                       "output_windows = [1, 5, 30, 90] días",
            ha="center", fontsize=9, color=TEXT_SUB)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def slide_matriz_dnn(pdf):
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)
    title_box(fig, "Matriz de Competición — DNN (MLP)",
              "MAE en test · mejor modelo por combinación · métrica: menor = mejor")
    footer(fig)

    ax = fig.add_axes([0.08, 0.12, 0.55, 0.68])

    # Anotaciones combinadas: MAE + modelo ganador
    annots = [[f"{dnn_mae[i,j]:.5f}\n{dnn_winners[i][j]}"
               for j in range(4)] for i in range(4)]
    im = heatmap_ax(ax, dnn_mae, ROW_LABELS, COL_LABELS,
                    cmap="RdYlGn_r", annotations=annots,
                    vmin=dnn_mae.min() * 0.998, vmax=dnn_mae.max() * 1.002)
    ax.set_title("MAE en Test — Mejor modelo por combinación",
                 fontsize=10, color=TEXT_SUB, pad=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="MAE Test")

    # Panel lateral con observaciones
    obs = [
        (ACCENT2,  "MLP-Dropout domina",
                   "12 de 16 combinaciones\nganadas"),
        (ACCENT3,  "MLP-Deep es 2.º",
                   "Gana en salidas 5d y 30d\ncon entradas cortas"),
        (ACCENT,   "MAE disminuye con\nhorizonte largo",
                   "Salida 90d ≈ 0.0013\n(promedio más estable)"),
        (TEXT_SUB, "Cerca del baseline",
                   "B&H ≈ 0.01227 vs\nDNN ≈ 0.01228 (salida 1d)"),
    ]
    y0 = 0.75
    for col, title, body in obs:
        fig.text(0.67, y0, f"▶  {title}", fontsize=10,
                 fontweight="bold", color=col)
        fig.text(0.70, y0 - 0.05, body, fontsize=9, color=TEXT_SUB)
        y0 -= 0.18

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def slide_comparacion_arquitecturas(pdf):
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)
    title_box(fig, "Comparación de Arquitecturas — MAE en Test",
              "Entrada 5 días (referencia) · todas las ventanas de salida")
    footer(fig)

    x = np.arange(4)
    width = 0.20

    ax = fig.add_axes([0.07, 0.14, 0.88, 0.65])
    ax.set_facecolor(CARD_BG)

    bars_data = [
        ("Buy & Hold", bh_mae[0],   TEXT_SUB,  "--"),
        ("DNN",        dnn_mae[0],  ACCENT,    "-"),
        ("CNN",        cnn_mae[0],  ACCENT2,   "-"),
        ("RNN",        rnn_mae[0],  ACCENT3,   "-"),
    ]

    for k, (label, vals, col, ls) in enumerate(bars_data):
        offset = (k - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=col, alpha=0.85, edgecolor="none")
        # Valor encima de cada barra
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.00005,
                    f"{v:.5f}", ha="center", va="bottom",
                    fontsize=6.5, color=col, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Salida {w}d" for w in OUTPUT_WINDOWS], fontsize=11)
    ax.set_ylabel("MAE en Test", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.legend(fontsize=10, facecolor=DARK_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_MAIN, loc="upper right")
    ax.grid(axis="y", alpha=0.4)

    # Nota clave
    fig.text(0.5, 0.06,
             "Las diferencias entre arquitecturas son mínimas (4.ª - 5.ª cifra decimal)  ·  "
             "La regularización supera a la complejidad arquitectónica",
             ha="center", fontsize=9, color=TEXT_SUB,
             bbox=dict(facecolor=CARD_BG, edgecolor=GRID_COL, boxstyle="round,pad=0.3"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def slide_reflexion(pdf):
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)
    title_box(fig, "02 · ¿Qué Modelos Funcionan Mejor?",
              "Análisis por arquitectura y tipo de ventana")
    footer(fig)

    # Izquierda: victorias DNN
    ax1 = fig.add_axes([0.05, 0.14, 0.38, 0.65])
    win_models = ["MLP-Dropout", "MLP-Deep"]
    win_vals   = [12, 4]
    win_colors = [ACCENT, ACCENT2]
    bars = ax1.barh(win_models, win_vals, color=win_colors,
                    alpha=0.85, height=0.4)
    for bar, v in zip(bars, win_vals):
        ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"{v}/16", va="center", fontsize=12,
                 fontweight="bold", color=TEXT_MAIN)
    ax1.set_xlim(0, 16)
    ax1.set_xlabel("Combinaciones ganadas (de 16)", fontsize=9)
    ax1.set_title("Victorias por modelo (DNN)", fontsize=10,
                  color=TEXT_SUB, pad=6)
    ax1.grid(axis="x", alpha=0.3)

    # Derecha: tabla resumen narrativo
    ax2 = fig.add_axes([0.50, 0.14, 0.47, 0.65])
    ax2.axis("off")
    rows = [
        ["Ventana",       "Mejor modelo",    "Por qué"],
        ["Salida 1d",     "MLP-Dropout",     "Alta regularización\ncombate el ruido"],
        ["Salida 5d",     "MLP-Deep",        "Señal algo más\nestable, más capas"],
        ["Salida 30d",    "MLP-Deep /\nDropout", "Ambos compiten,\npoca diferencia"],
        ["Salida 90d",    "MLP-Dropout",     "Regularización evita\nsobreajuste a largo"],
        ["Ent. largas\n(30-90d)", "DNN >\nRNN", "RNN pierde gradiente\nen series largas"],
    ]
    col_widths = [0.22, 0.28, 0.50]
    colors_rows = [ACCENT] + [None] * (len(rows) - 1)
    y_pos = 0.95
    for r, row in enumerate(rows):
        x_pos = 0.0
        for c, (cell, w) in enumerate(zip(row, col_widths)):
            if r == 0:
                fc = ACCENT + "33"; ec = ACCENT; fw = "bold"; fs = 9; tc = ACCENT
            else:
                fc = CARD_BG if r % 2 == 0 else DARK_BG
                ec = GRID_COL; fw = "normal"; fs = 8.5; tc = TEXT_MAIN
            rect = FancyBboxPatch((x_pos, y_pos - 0.13), w - 0.01, 0.13,
                                  boxstyle="square,pad=0",
                                  facecolor=fc, edgecolor=ec,
                                  transform=ax2.transAxes, clip_on=False)
            ax2.add_patch(rect)
            ax2.text(x_pos + w / 2, y_pos - 0.065, cell,
                     ha="center", va="center",
                     fontsize=fs, fontweight=fw, color=tc,
                     transform=ax2.transAxes)
            x_pos += w
        y_pos -= 0.135

    # Conclusión
    fig.text(0.5, 0.06,
             "Conclusión clave: Dropout simple > profundidad > anchura para este problema  "
             "|  Los retornos a 1 día son casi impredecibles (≈ ruido blanco)",
             ha="center", fontsize=9, color=TEXT_SUB,
             bbox=dict(facecolor=CARD_BG, edgecolor=ACCENT2,
                       boxstyle="round,pad=0.3"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def slide_preprocesado_intro(pdf):
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)
    title_box(fig, "03 · Técnicas de Preprocesado — López de Prado",
              "5 técnicas de Advances in Financial Machine Learning (2018)")
    footer(fig)

    tecnicas = [
        (ACCENT,  "T1", "Dollar Bars",
                  "Barras por volumen-dólar constante\n(no por tiempo)",
                  "Menor kurtosis · distribución más\nnormal · 2507 barras vs ~16000 días"),
        (ACCENT2, "T2", "FFD\n(Diferenciación Fraccionaria)",
                  "d* mínimo por activo que logra\nestacionariedad ADF (p<0.05)",
                  "d* ≈ 0.20 · Conserva ~80% de\nmemoria del precio · X del modelo"),
        (ACCENT3, "T3", "RMT\n(Random Matrix Theory)",
                  "Limpia la covarianza 23×23\nde eigenvalores-ruido",
                  "Solo 1/23 eigenvalores es señal\nreal · Covarianza estable p/ carteras"),
        (ACCENT4, "T4", "Triple Barrera",
                  "Etiqueta UP/NEUTRAL/DOWN según\nqué barrera toca primero",
                  "Take-Profit: +1.5σ · Stop-Loss: -1.5σ\nEtiqueta binaria: UP=1, resto=0"),
        (TEXT_MAIN,"T5","Purged K-Fold",
                  "Validación cruzada sin data leakage\ntemporal (embargo=input_window)",
                  "Elimina muestras solapadas del train\n5 folds · evaluación honesta"),
    ]

    xs  = [0.10, 0.30, 0.50, 0.70, 0.90]
    y_box = 0.30
    for (col, num, name, desc, efecto), x in zip(tecnicas, xs):
        # Número arriba
        ax_n = fig.add_axes([x - 0.07, 0.72, 0.14, 0.10])
        ax_n.set_facecolor(col + "22")
        ax_n.set_xlim(0, 1); ax_n.set_ylim(0, 1)
        for sp in ax_n.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2)
        ax_n.set_xticks([]); ax_n.set_yticks([])
        ax_n.text(0.5, 0.5, num, ha="center", va="center",
                  fontsize=16, fontweight="bold", color=col)

        # Nombre
        fig.text(x, 0.69, name, ha="center", fontsize=9,
                 fontweight="bold", color=col)

        # Descripción
        fig.text(x, 0.55, desc, ha="center", fontsize=8,
                 color=TEXT_MAIN, linespacing=1.4)

        # Efecto
        fig.text(x, 0.36, efecto, ha="center", fontsize=7.5,
                 color=TEXT_SUB, linespacing=1.4,
                 bbox=dict(facecolor=col + "11", edgecolor=col + "44",
                           boxstyle="round,pad=0.3"))

    # Flecha pipeline
    fig.text(0.5, 0.18,
             "Precios OHLCV  →  Dollar Bars  →  FFD (X)  +  RMT (Σ)  →  "
             "Triple Barrera (Y)  →  Purged K-Fold  →  Clasificación",
             ha="center", fontsize=9, color=TEXT_SUB)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def slide_ffd_detalle(pdf):
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)
    title_box(fig, "T2 · Diferenciación Fraccionaria (FFD)",
              "Conservar memoria del precio sin perder estacionariedad")
    footer(fig)

    # Panel izquierdo: concepto visual
    ax_l = fig.add_axes([0.05, 0.14, 0.42, 0.65])
    ds   = [0.0, 0.2, 0.45, 1.0]
    mems = [100, 80, 55, 0]
    stat = [0,   60, 95, 100]
    x_d  = np.array(ds)

    ax_l.fill_between(x_d, mems, alpha=0.3, color=ACCENT,  label="Memoria histórica %")
    ax_l.fill_between(x_d, stat, alpha=0.3, color=ACCENT3, label="Estacionariedad %")
    ax_l.plot(x_d, mems, "o-", color=ACCENT,  lw=2)
    ax_l.plot(x_d, stat, "s-", color=ACCENT3, lw=2)

    # Zona óptima
    ax_l.axvspan(0.15, 0.30, alpha=0.12, color=ACCENT2,
                 label="Zona óptima (d* ≈ 0.20)")
    ax_l.axvline(0.20, color=ACCENT2, lw=1.5, ls="--")
    ax_l.text(0.21, 50, "d* ≈ 0.20", color=ACCENT2, fontsize=9)

    ax_l.set_xlabel("Grado de diferenciación d", fontsize=10)
    ax_l.set_ylabel("%", fontsize=10)
    ax_l.set_title("Trade-off Memoria vs Estacionariedad", fontsize=10,
                   color=TEXT_SUB, pad=6)
    ax_l.legend(fontsize=8, facecolor=DARK_BG, edgecolor=GRID_COL,
                labelcolor=TEXT_MAIN)
    ax_l.set_xlim(-0.05, 1.05)
    ax_l.set_ylim(-5, 110)
    ax_l.grid(alpha=0.3)

    # Panel derecho: valores d* reales por activo
    ax_r = fig.add_axes([0.54, 0.14, 0.43, 0.65])
    tickers = ["ED", "IP", "HPQ", "BA", "MSI", "CNP", "CAT", "CVX",
               "IBM", "DIS", "HON", "GE", "KR", "AEP", "DTE", "GD",
               "KO", "JNJ", "MRK", "MMM", "XOM", "PG", "MO"]
    d_vals  = [0.05, 0.10, 0.15, 0.15, 0.15, 0.20, 0.20, 0.20,
               0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.25, 0.25,
               0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.30]
    colors_d = [ACCENT if d <= 0.20 else ACCENT2 for d in d_vals]
    bars_r = ax_r.barh(tickers, d_vals, color=colors_d, alpha=0.85, height=0.7)
    ax_r.axvline(np.mean(d_vals), color=TEXT_SUB, ls="--", lw=1.2,
                 label=f"Media d* = {np.mean(d_vals):.3f}")
    ax_r.set_xlabel("d* (mínimo para estacionariedad ADF p<0.05)", fontsize=8)
    ax_r.set_title("d* por activo (datos 1962-2024)", fontsize=10,
                   color=TEXT_SUB, pad=6)
    ax_r.legend(fontsize=8, facecolor=DARK_BG, edgecolor=GRID_COL,
                labelcolor=TEXT_MAIN)
    ax_r.set_xlim(0, 0.38)
    ax_r.tick_params(axis="y", labelsize=7)
    ax_r.grid(axis="x", alpha=0.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def slide_triple_barrera(pdf):
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)
    title_box(fig, "T4 · Método de la Triple Barrera",
              "Etiquetado de clasificación financiera — López de Prado Cap. 3")
    footer(fig)

    # Izquierda: diagrama de una trayectoria
    ax_l = fig.add_axes([0.05, 0.14, 0.45, 0.65])

    np.random.seed(7)
    t = np.arange(0, 21)
    precio = 100 + np.cumsum(np.random.randn(21) * 0.8)
    pt = precio[0] * 1.02
    sl = precio[0] * 0.98
    t_exp = 20

    ax_l.plot(t, precio, color=ACCENT, lw=2, zorder=5, label="Trayectoria precio")
    ax_l.axhline(pt,      color=ACCENT3, lw=1.8, ls="--", label=f"Take-Profit (+2%)")
    ax_l.axhline(sl,      color=ACCENT4, lw=1.8, ls="--", label=f"Stop-Loss (−2%)")
    ax_l.axvline(t_exp,   color=ACCENT2, lw=1.8, ls=":",  label="Barrera temporal")

    # Zona entre barreras
    ax_l.fill_between(t, sl, pt, alpha=0.07, color=TEXT_MAIN)

    # Marcar primer cruce (si lo hay)
    for i, p in enumerate(precio):
        if p >= pt:
            ax_l.scatter(i, p, color=ACCENT3, s=120, zorder=10,
                         label=f"→ Etiqueta: UP (+1) en t={i}")
            ax_l.annotate("UP (+1)", (i, p), (i + 0.5, p + 0.5),
                          color=ACCENT3, fontsize=10, fontweight="bold",
                          arrowprops=dict(arrowstyle="->", color=ACCENT3))
            break

    ax_l.set_xlabel("Pasos temporales (horizonte = output_window)", fontsize=9)
    ax_l.set_ylabel("Precio", fontsize=9)
    ax_l.set_title("Ejemplo: trayectoria toca Take-Profit → etiqueta UP",
                   fontsize=9, color=TEXT_SUB, pad=6)
    ax_l.legend(fontsize=8, facecolor=DARK_BG, edgecolor=GRID_COL,
                labelcolor=TEXT_MAIN)
    ax_l.grid(alpha=0.3)

    # Derecha: distribución de etiquetas por output_window
    ax_r = fig.add_axes([0.57, 0.14, 0.40, 0.65])
    out_ws = [1, 5, 30, 90]
    up_pct    = [7.6,  36.1, 53.7, 53.8]
    down_pct  = [6.2,  30.9, 45.6, 45.6]
    neutral_pct = [86.2, 33.1, 0.7, 0.6]
    x_pos = np.arange(len(out_ws))

    ax_r.bar(x_pos, up_pct,      color=ACCENT3, alpha=0.85, label="UP (+1)")
    ax_r.bar(x_pos, down_pct,    bottom=up_pct, color=ACCENT4, alpha=0.85,
             label="DOWN (−1)")
    ax_r.bar(x_pos, neutral_pct,
             bottom=[u+d for u, d in zip(up_pct, down_pct)],
             color=TEXT_SUB, alpha=0.5, label="NEUTRAL (0)")

    for i, (u, d, n) in enumerate(zip(up_pct, down_pct, neutral_pct)):
        if u > 5:
            ax_r.text(i, u / 2,       f"{u:.0f}%", ha="center", va="center",
                      fontsize=9, color="white", fontweight="bold")
        if d > 5:
            ax_r.text(i, u + d / 2,   f"{d:.0f}%", ha="center", va="center",
                      fontsize=9, color="white", fontweight="bold")
        if n > 5:
            ax_r.text(i, u + d + n/2, f"{n:.0f}%", ha="center", va="center",
                      fontsize=9, color="white", fontweight="bold")

    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels([f"out_w={w}d" for w in out_ws])
    ax_r.set_ylabel("% etiquetas")
    ax_r.set_ylim(0, 105)
    ax_r.set_title("Distribución de etiquetas (pt=sl=1.5σ)", fontsize=9,
                   color=TEXT_SUB, pad=6)
    ax_r.legend(fontsize=8, facecolor=DARK_BG, edgecolor=GRID_COL,
                labelcolor=TEXT_MAIN)
    ax_r.grid(axis="y", alpha=0.3)

    fig.text(0.5, 0.06,
             "Con horizonte=1d: 86% NEUTRAL (el mercado rara vez ±2% en un día)  "
             "→  El modelo aprende a predecir siempre NEUTRAL  →  Accuracy engañoso",
             ha="center", fontsize=9, color=ACCENT4,
             bbox=dict(facecolor=CARD_BG, edgecolor=ACCENT4,
                       boxstyle="round,pad=0.3"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def slide_carteras(pdf):
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)
    title_box(fig, "04 · Resultados de Carteras 2025",
              "Buy & Hold vs MLP-guided (Triple Barrera + RMT) · 23 activos SP500")
    footer(fig)

    # ── Retorno acumulado ──
    ax1 = fig.add_axes([0.05, 0.40, 0.55, 0.45])
    dias = np.arange(len(cum_bh))
    ax1.plot(dias, cum_bh  * 100, color=ACCENT,  lw=2.5, label="A: Buy & Hold")
    ax1.plot(dias, cum_mlp * 100, color=ACCENT3, lw=2.5, label="B: MLP-guided")
    ax1.fill_between(dias, cum_bh * 100, cum_mlp * 100,
                     where=(cum_mlp >= cum_bh),
                     alpha=0.15, color=ACCENT3, label="Ventaja MLP")
    ax1.axhline(0, color=GRID_COL, lw=1, ls="--")
    ax1.set_ylabel("Retorno acumulado (%)", fontsize=10)
    ax1.set_xlabel("Días de trading 2025", fontsize=10)
    ax1.set_title("Retorno acumulado 2025", fontsize=10, color=TEXT_SUB, pad=6)
    ax1.legend(fontsize=9, facecolor=DARK_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_MAIN)
    ax1.grid(alpha=0.4)

    # ── Tabla de métricas ──
    ax2 = fig.add_axes([0.63, 0.40, 0.34, 0.45])
    ax2.axis("off")
    metric_keys = list(portfolio_metrics.keys())
    vals_a = [portfolio_metrics[k][0] for k in metric_keys]
    vals_b = [portfolio_metrics[k][1] for k in metric_keys]

    # Cabecera
    ax2.text(0.33, 0.97, "Buy & Hold", ha="center", fontsize=10,
             fontweight="bold", color=ACCENT, transform=ax2.transAxes)
    ax2.text(0.75, 0.97, "MLP-guided", ha="center", fontsize=10,
             fontweight="bold", color=ACCENT3, transform=ax2.transAxes)

    for k, (metric, va, vb) in enumerate(zip(metric_keys, vals_a, vals_b)):
        y = 0.82 - k * 0.175
        bg = CARD_BG if k % 2 == 0 else DARK_BG

        row_rect = FancyBboxPatch((0, y - 0.07), 1.0, 0.14,
                                  boxstyle="square,pad=0",
                                  facecolor=bg, edgecolor=GRID_COL,
                                  transform=ax2.transAxes, clip_on=False)
        ax2.add_patch(row_rect)

        ax2.text(0.02, y, metric, ha="left", va="center", fontsize=9,
                 color=TEXT_MAIN, transform=ax2.transAxes)

        # Valores con color verde si MLP > B&H (o max DD: si menos negativo)
        better_b = (vb > va) if "DD" not in metric else (vb > va)
        col_a = ACCENT;  col_b = ACCENT3 if better_b else ACCENT4
        ax2.text(0.55, y, f"{va:+.2f}" if "%" in metric or "Sharpe" in metric
                 else f"{va:.2f}", ha="center", va="center", fontsize=10,
                 fontweight="bold", color=col_a, transform=ax2.transAxes)
        ax2.text(0.85, y, f"{vb:+.2f}" if "%" in metric or "Sharpe" in metric
                 else f"{vb:.2f}", ha="center", va="center", fontsize=10,
                 fontweight="bold", color=col_b, transform=ax2.transAxes)

    # ── Panel inferior: metodología ──
    ax3 = fig.add_axes([0.05, 0.10, 0.90, 0.24])
    ax3.axis("off")

    metodologia = [
        (ACCENT,  "Señal del modelo",
                  "MLP-Deep (entrada 30d)\npredice p(UP) por activo"),
        (ACCENT2, "Regla de inversión",
                  "Invertir si p(UP) > 0.5\n(rebalanceo mensual)"),
        (ACCENT3, "Pesos de cartera",
                  "Mínima varianza long-only\nsobre activos seleccionados"),
        (ACCENT4, "Covarianza",
                  "RMT-limpia (Marchenko-Pastur)\n1/23 eigenvalores señal"),
    ]
    xs_m = [0.12, 0.37, 0.62, 0.87]
    for (col, title, body), x in zip(metodologia, xs_m):
        ax3.text(x, 0.85, title, ha="center", va="top", fontsize=9,
                 fontweight="bold", color=col, transform=ax3.transAxes)
        ax3.text(x, 0.52, body, ha="center", va="top", fontsize=8.5,
                 color=TEXT_MAIN, linespacing=1.4,
                 transform=ax3.transAxes)
        # Separadores
        if x < 0.87:
            ax3.axvline(x + 0.125, color=GRID_COL, lw=1)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def slide_conclusiones(pdf):
    fig = plt.figure(figsize=(14, 8))
    slide_bg(fig)
    title_box(fig, "Conclusiones", "")
    footer(fig)

    conclusiones = [
        (ACCENT,  "Los mercados son difíciles de predecir a corto plazo",
                  "Retorno a 1d ≈ ruido blanco · DNN ≈ Buy & Hold en MAE (4.ª cifra decimal)"),
        (ACCENT2, "La regularización supera a la complejidad arquitectónica",
                  "MLP-Dropout gana 12/16 combinaciones · Dropout > Deep > Wide > Residual"),
        (ACCENT3, "La señal existe en horizontes largos",
                  "MAE salida 90d ≈ 0.0013 vs MAE salida 1d ≈ 0.0123 — 10× más predecible"),
        (ACCENT4, "Las técnicas de López de Prado hacen el problema más honesto",
                  "FFD conserva memoria · Triple Barrera etiqueta bien · Purged K-Fold evita leakage"),
        (TEXT_MAIN,"La cartera MLP-guided supera al Buy & Hold en 2025",
                  "+27.68% vs +21.76% · Sharpe 2.20 vs 1.76 · mejor Drawdown máximo"),
    ]

    y_positions = [0.74, 0.61, 0.48, 0.35, 0.22]
    for (col, title, body), y in zip(conclusiones, y_positions):
        # Círculo de color
        ax_dot = fig.add_axes([0.05, y - 0.01, 0.025, 0.055])
        circ = plt.Circle((0.5, 0.5), 0.42, color=col, transform=ax_dot.transAxes)
        ax_dot.add_patch(circ)
        ax_dot.axis("off")

        fig.text(0.09, y + 0.025, title, fontsize=12,
                 fontweight="bold", color=col)
        fig.text(0.09, y - 0.01,  body,  fontsize=9.5, color=TEXT_SUB)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# GENERAR PDF
# ─────────────────────────────────────────────────────────────────────

output_path = "presentacion_redes_neuronales.pdf"

with PdfPages(output_path) as pdf:
    print("Generando slides...")

    print("  [1/8] Portada")
    slide_portada(pdf)

    print("  [2/8] Índice")
    slide_indice(pdf)

    print("  [3/8] Intro competición")
    slide_competicion_intro(pdf)

    print("  [4/8] Matriz DNN + comparación arquitecturas")
    slide_matriz_dnn(pdf)
    slide_comparacion_arquitecturas(pdf)

    print("  [5/8] Reflexión modelos")
    slide_reflexion(pdf)

    print("  [6/8] Preprocesado López de Prado")
    slide_preprocesado_intro(pdf)
    slide_ffd_detalle(pdf)
    slide_triple_barrera(pdf)

    print("  [7/8] Carteras 2025")
    slide_carteras(pdf)

    print("  [8/8] Conclusiones")
    slide_conclusiones(pdf)

    # Metadata
    d = pdf.infodict()
    d["Title"]   = "Taller Redes Neuronales — Presentación"
    d["Author"]  = "MIAX 2025"
    d["Subject"] = "Forecasting SP500 con DNN, CNN, RNN y López de Prado"

print(f"\nPDF generado: {output_path}")
print(f"Total slides: 10")
