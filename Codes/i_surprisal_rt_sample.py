"""
Einzelsatz-Grafik: LLM-Surprisal vs. menschliche Lesezeit
=========================================================

Zeigt für einen Satz (item_order) je Wort das LLM-Surprisal als Balken und die
mittlere First-Pass-Lesezeit der Versuchspersonen als Linie auf zweiter Achse;
CS-Punkte sind farblich hervorgehoben. Dient in der Arbeit als konkretes
Beispiel (Abschnitt 8.3).

Eingabe : Bspr_result/main_analysis_with_surprisal_filtered.csv
          (Versuchsperson x Item x Wort, mit surprisal_raw,
          first_pass_reading_time und cs_point)
Ausgabe : Bspr_result/sentence_item<N>_word_table.csv
          Bspr_result/sentence_item<N>_surprisal_vs_rt.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

IN_PATH = Path("/Bspr_result/main_analysis_with_surprisal_filtered.csv")
OUT_DIR = Path("Bspr_result")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ITEM_ORDER = 2  # <-- hier den gewünschten Satz eintragen

df = pd.read_csv(IN_PATH)
sub = df[df["item_order"] == ITEM_ORDER].copy()

# Wortebene: Surprisal ist für alle Versuchspersonen gleich (erster Wert),
# die Lesezeit wird über die Versuchspersonen gemittelt.
word_summary = (
    sub.groupby(["word_position", "word", "cs_point"], sort=True)
    .agg(surprisal=("surprisal_raw", "first"),
         mean_rt=("first_pass_reading_time", "mean"),
         median_rt=("first_pass_reading_time", "median"),
         n_participants=("first_pass_reading_time", "size"))
    .reset_index()
    .sort_values("word_position")
)

sentence_text = " ".join(word_summary["word"])
print(f"Stimulussatz (item_order={ITEM_ORDER}): {sentence_text}\n")
print(word_summary.to_string(index=False))
word_summary.to_csv(OUT_DIR / f"sentence_item{ITEM_ORDER}_word_table.csv", index=False)

# Grafik: zwei Achsen (Surprisal = Balken, Lesezeit = Linie)
fig, ax1 = plt.subplots(figsize=(max(8, len(word_summary) * 0.9), 5))

x = np.arange(len(word_summary))
colors = ["#d64545" if cs == 1 else "#8aa0c8" for cs in word_summary["cs_point"]]

ax1.bar(x, word_summary["surprisal"], color=colors, alpha=0.85)
ax1.set_ylabel("LLM-Surprisal (bits)", fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(word_summary["word"], rotation=45, ha="right")
ax1.set_xlabel("Wort", fontsize=11)

ax2 = ax1.twinx()
ax2.plot(x, word_summary["mean_rt"], color="black", marker="o", linewidth=2)
ax2.set_ylabel("Durchschnittliche First-Pass-Lesezeit (ms)", fontsize=8)

# Legende unterhalb der Achsen, damit sie die Lesezeitlinie nicht überdeckt.
legend_elems = [
    Patch(facecolor="#d64545", alpha=0.85, label="CS-Punkt (Surprisal)"),
    Patch(facecolor="#8aa0c8", alpha=0.85, label="Nicht-CS (Surprisal)"),
    Line2D([0], [0], color="black", marker="o", label="Durchschnittliche Lesezeit"),
]
ax1.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.60),
           ncol=3, fontsize=11, frameon=False)

plt.title(f"Stimulussatz (item_order={ITEM_ORDER}): \"{sentence_text}\"", fontsize=12)
plt.tight_layout()

out_path = OUT_DIR / f"sentence_item{ITEM_ORDER}_surprisal_vs_rt.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"\nGespeichert: {out_path}")
