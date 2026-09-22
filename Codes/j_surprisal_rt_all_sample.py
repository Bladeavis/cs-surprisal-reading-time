"""
Übersichtsraster: monolinguale und multilinguale Sätze getrennt
===============================================================

Wiederholt die Einzelsatz-Grafik (Surprisal als Balken, durchschnittliche First-Pass-
Lesezeit als Linie, CS-Punkte hervorgehoben) verkleinert in einem Raster.
Es entstehen zwei Abbildungen: eine für die monolingualen und eine für die
multilingualen (Code-Switching-)Sätze.

Eingabe : Bspr_result/main_analysis_with_surprisal_filtered.csv, word_cs.xlsx
Ausgabe : Bspr_result/monolingual_sentences_grid.png
          Bspr_result/multilingual_sentences_grid.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

IN_PATH = Path("Bspr_result/main_analysis_with_surprisal_filtered.csv")
CS_PATH = Path("word_cs.xlsx")
OUT_DIR = Path("Bspr_result")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIRST_ITEM_ORDER = 2

df = pd.read_csv(IN_PATH)

# Hinweis: experimental_condition (aus dem PCIbex-Rohlog) enthält zwei bekannte
# Fehlkodierungen (item_order 20 und 37) – dort passt das Design-Label nicht zum
# tatsächlichen sprachlichen Inhalt. Sentence-Type aus word_cs.xlsx ist manuell
# geprüft und wird deshalb für die Aufteilung mono-/multilingual verwendet.
cs_xlsx = pd.read_excel(CS_PATH)
sentence_order = cs_xlsx["Sentence"].drop_duplicates().tolist()
item_to_sentence_type = {
    FIRST_ITEM_ORDER + i: cs_xlsx.loc[cs_xlsx["Sentence"] == s, "Sentence-Type"].iloc[0]
    for i, s in enumerate(sentence_order)
}
df["sentence_type"] = df["item_order"].map(item_to_sentence_type)

word_level = (
    df.groupby(["item_order", "word_position", "word", "cs_point", "sentence_type"], sort=False)
    .agg(surprisal=("surprisal_raw", "first"),
         mean_rt=("first_pass_reading_time", "mean"))
    .reset_index()
)

LEGEND_ELEMS = [
    Patch(facecolor="#d64545", alpha=0.85, label="CS-Punkt (Surprisal)"),
    Patch(facecolor="#8aa0c8", alpha=0.85, label="Nicht-CS (Surprisal)"),
    Line2D([0], [0], color="black", marker="o", label="Durchschnittliche Lesezeit"),
]


def plot_grid(data, title, out_name, n_cols=4):
    """Zeichnet je Satz ein kleines Diagramm mit dynamischer Wortabstands- und Schriftanpassung."""
    item_orders = sorted(data["item_order"].unique())
    n = len(item_orders)
    n_rows = int(np.ceil(n / n_cols))

    # Breitere Abbildung fuer laengere Saetze im Raster
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 9.0, n_rows * 4.4))
    axes = np.atleast_1d(axes).flatten()

    for ax, item_order in zip(axes, item_orders):
        sub = data[data["item_order"] == item_order].sort_values("word_position")
        n_words = len(sub)

        # Schrittweite vergroessern, um den horizontalen Abstand zwischen den Woertern zu dehnen
        step = 2.4
        x = np.arange(n_words) * step

        colors = ["#d64545" if cs == 1 else "#8aa0c8" for cs in sub["cs_point"]]

        # Balkenbreite proportional zur Schrittweite setzen, damit Luecken sichtbar bleiben
        ax.bar(x, sub["surprisal"], width=1.3, color=colors, alpha=0.85)

        # Dynamische Schriftgroesse: Saetze mit mehr als 12 Woertern erhalten eine leicht kleinere Schrift
        current_font = 8.0 if n_words > 12 else 9.5

        ax.set_xticks(x)
        # 70-Grad-Winkel verhindert horizontales Ueberlappen bei langen Woertern
        ax.set_xticklabels(
            sub["word"],
            rotation=70,
            ha="right",
            rotation_mode="anchor",
            fontsize=current_font,
        )
        ax.tick_params(axis="y", labelsize=15, labelcolor="#444444")
        ax.set_title(f"Item {item_order}", fontsize=30, loc="left", pad=4, weight="bold")

        ax.set_ylabel("LLM-Surprisal (bits)", fontsize=12, color="#222222")
        ax.set_xlabel("Wort", fontsize=12, labelpad=8)

        # Raender links und rechts anpassen, damit aeussere Balken nicht am Achsenrand kleben
        ax.set_xlim(-1.6, (n_words - 1) * step + 1.6)

        # Lesezeitkurve auf denselben gestreckten x-Koordinaten zeichnen
        ax2 = ax.twinx()
        ax2.plot(x, sub["mean_rt"], color="black", marker="o", markersize=3.5, linewidth=1.2)
        ax2.tick_params(axis="y", labelsize=15, labelcolor="#000000")
        ax2.set_ylabel("Durchschnittliche First-Pass-Lesezeit (ms)", fontsize=12, color="#000000")

    # Nicht benoetigte Subplots abschalten
    for ax in axes[n:]:
        ax.axis("off")

    # Gesamttitel
    fig.suptitle(title, fontsize=40, y=0.985, weight="bold")

    # Layout-Abstaende: Ausreichend Platz nach unten fuer die Legende und vertikal zwischen den Zeilen
    plt.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.07, wspace=0.30, hspace=0.75)

    # Vergroesserte Legende unterhalb der Abbildung
    fig.legend(handles=LEGEND_ELEMS, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=3, fontsize=30, frameon=False)

    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")

    #out_path = OUT_DIR / out_name.replace(".png", ".pdf")
    #fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Gespeichert: {out_path}  ({n} Sätze)")

    plt.show()

    return fig


# Funktionsaufrufe
plot_grid(
    word_level[word_level["sentence_type"] == "Monolingual"],
    "LLM-Surprisal vs. durchschnittliche First-Pass-Lesezeit — monolinguale Sätze",
    "monolingual_sentences_grid.png"
)

plot_grid(
    word_level[word_level["sentence_type"] == "Multilingual"],
    "LLM-Surprisal vs. durchschnittliche First-Pass-Lesezeit — multilinguale (CS) Sätze",
    "multilingual_sentences_grid.png"
)
