"""
LMEM: Zusammenhang zwischen LLM-Surprisal und menschlicher Lesezeit
===================================================================

Geschätzt wird ein Modell:

    log_reading_time ~ surprisal_z * cs_point
                       + (1 + surprisal_z | participant_id) + (1 | item_order)

Da cs_point mit 0/1 kodiert ist, gibt die Zeile "surprisal_z" die
Surprisal-Steigung an Nicht-Wechselpunkten wieder, "cs_point" den
zusätzlichen Aufwand am Sprachwechsel bei mittlerer Surprisal und
"surprisal_z:cs_point" die Differenz beider Steigungen.

Eingabe : main_analysis_dataframe_lmem_ready.csv, small_surprisal_values.csv (XGLM-564M)
Ausgabe : Bspr_result/ (gefilterte Daten, feste Effekte)

Benötigt pymer4 >= 0.9 (conda-Installation, siehe colab_setup_every_session.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from pymer4.models import lmer

# Einstellungen
READING_TIME_FILE = Path("/Bspr_result/main_analysis_dataframe_lmem_ready.csv")
SURPRISAL_FILE = Path("/Bspr_result/small_surprisal_values.csv")
OUTPUT_DIR = Path("Bspr_result")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# REML=True ist lme4-Voreinstellung und für die Berichterstattung der festen
# Effekte üblich.
USE_REML = True

RANDOM_EFFECTS = "+ (1 + surprisal_z | participant_id) + (1 | item_order)"

# Daten laden, zusammenführen und filtern
reading_times = pd.read_csv(READING_TIME_FILE)
surprisal = pd.read_csv(SURPRISAL_FILE)[["item_order", "word_position", "surprisal"]]

data = reading_times.merge(surprisal, on=["item_order", "word_position"], how="left")
data = data.dropna(subset=["surprisal", "first_pass_reading_time"]).reset_index(drop=True)
print(f"Zeilen im Modell (vor Filtern): {len(data)}")

# Erstes Wort je Satz ausschließen: Es war beim BSPR sofort sichtbar (kein
# Tastendruck nötig), die First-Pass-Zeit misst dort also keine reguläre
# Worterkennung, sondern u. a. die Pause vor Satzbeginn.
n_before = len(data)
data = data[data["word_position"] != 1].reset_index(drop=True)
print(f"Erstes Wort je Satz entfernt: {n_before - len(data)} Zeilen ({len(data)} verbleiben)")

# Ausreißerfilter nach Oh & Schuler (2023) und Chen et al. (2026):
# First-Pass-Lesezeiten unter 100 ms oder über 3000 ms werden entfernt.
n_before = len(data)
data = data[(data["first_pass_reading_time"] >= 100)
            & (data["first_pass_reading_time"] <= 3000)].reset_index(drop=True)
print(f"Durch RT-Filter (100–3000 ms) entfernt: {n_before - len(data)} Zeilen "
      f"({len(data)} verbleiben)")

data["surprisal_z"] = (data["surprisal"] - data["surprisal"].mean()) / data["surprisal"].std()
data["log_reading_time"] = np.log(data["first_pass_reading_time"])
data["cs_point"] = data["cs_point"].astype(int)
data["participant_id"] = data["participant_id"].astype(str)
data["item_order"] = data["item_order"].astype(str)

# Rohwert des Surprisal unter eigenem Namen sichern: i_surprisal_rt_sample.py
# und j_surprisal_rt_all_sample.py greifen auf "surprisal_raw" zu.
data["surprisal_raw"] = data["surprisal"]

# Wortzahlen aus den Daten berechnen (nach dem Entfernen des ersten Wortes
# stimmen fest eingetragene Werte nicht mehr).
word_index = data[["item_order", "word_position", "cs_point"]].drop_duplicates()
n_words_all = len(word_index)
n_words_cs = int(word_index["cs_point"].sum())
n_words_non_cs = n_words_all - n_words_cs
print(f"\nWörter im Modell: {n_words_all} gesamt | {n_words_non_cs} Nicht-CS | {n_words_cs} CS")

data_pl = pl.from_pandas(data)  # pymer4 >= 0.9 erwartet polars


# Modell schätzen
def fit_model(formula, data_pl, use_reml=USE_REML):
    """Schätzt ein Modell; ältere pymer4-Versionen kennen REML in fit() nicht."""
    m = lmer(formula, data=data_pl)
    try:
        m.fit(REML=use_reml)
    except TypeError:
        if not use_reml:
            print("Hinweis: Diese pymer4-Version akzeptiert REML nicht in fit(); "
                  "es wird die Voreinstellung (REML=True) verwendet.")
        m.fit()
    return m


def as_pandas(result):
    return result.to_pandas() if hasattr(result, "to_pandas") else pd.DataFrame(result)


model = fit_model(f"log_reading_time ~ surprisal_z * cs_point {RANDOM_EFFECTS}", data_pl)
effects = as_pandas(model.result_fit)

print("\n--- Modell: surprisal_z * cs_point ---")
print(effects.to_string(index=False))


# Effekte auf der Millisekunden-Skala (Rücktransformation aus log)
def get_estimate(term):
    return float(effects.loc[effects["term"] == term, "estimate"].iloc[0])


intercept = get_estimate("(Intercept)")
b_surprisal = get_estimate("surprisal_z")
b_cs = get_estimate("cs_point")
b_interaction = get_estimate("surprisal_z:cs_point")

baseline_ms = np.exp(intercept)
slope_cs = b_surprisal + b_interaction

print("\n--- Interpretationshilfe (Rücktransformation) ---")
print(f"Vorhergesagte Lesezeit an Nicht-Wechselpunkten bei mittlerer Surprisal: "
      f"{baseline_ms:.0f} ms")
print(f"Surprisal-Steigung an Nicht-Wechselpunkten: {b_surprisal:+.4f} log-Einheiten "
      f"= {100 * (np.exp(b_surprisal) - 1):+.1f} % pro SD")
print(f"Surprisal-Steigung an Wechselpunkten (Summe aus Haupteffekt und Interaktion): "
      f"{slope_cs:+.4f} log-Einheiten = {100 * (np.exp(slope_cs) - 1):+.1f} % pro SD")
print(f"Haupteffekt Sprachwechsel: {b_cs:+.4f} log-Einheiten "
      f"= {100 * (np.exp(b_cs) - 1):+.1f} % bzw. "
      f"{baseline_ms * (np.exp(b_cs) - 1):+.0f} ms")


# Surprisal-Steigung an Wechselpunkten als Kontrast
# Die Steigung an Wechselpunkten ist die Summe zweier geschätzter Koeffizienten
# und erscheint deshalb nicht als eigene Zeile in der Ergebnistabelle. Ihr
# Standardfehler folgt aus der Varianz-Kovarianz-Matrix der festen Effekte:
#     SE = sqrt(Var(b1) + Var(b3) + 2 * Cov(b1, b3))
# Das entspricht emmeans::emtrends und erfordert kein zweites Modell.
def fixed_effects_vcov(model):
    """Holt die Varianz-Kovarianz-Matrix der festen Effekte aus dem Modell."""
    for attr in ("vcov", "vcov_", "coef_vcov", "fixef_vcov"):
        value = getattr(model, attr, None)
        if value is not None:
            return np.asarray(value)

    # Rückfallebene über das zugrunde liegende R-Objekt.
    for attr in ("model_obj", "r_model", "_model"):
        r_model = getattr(model, attr, None)
        if r_model is not None:
            from rpy2.robjects import r as _r
            return np.asarray(_r["as.matrix"](_r["vcov"](r_model)))

    raise AttributeError("Varianz-Kovarianz-Matrix in dieser pymer4-Version nicht "
                         "zugänglich; Kontrast kann nicht berechnet werden.")


try:
    from scipy import stats

    V = fixed_effects_vcov(model)
    terms = list(effects["term"])
    i = terms.index("surprisal_z")
    j = terms.index("surprisal_z:cs_point")

    se_cs = float(np.sqrt(V[i, i] + V[j, j] + 2 * V[i, j]))
    t_cs = slope_cs / se_cs
    # Konservative Näherung: kleinere der beiden Satterthwaite-df.
    df_cs = float(min(effects.loc[i, "df"], effects.loc[j, "df"]))
    p_cs = float(2 * stats.t.sf(abs(t_cs), df_cs))
    ci_low, ci_high = (slope_cs - stats.t.ppf(0.975, df_cs) * se_cs,
                       slope_cs + stats.t.ppf(0.975, df_cs) * se_cs)

    print("\n--- Kontrast: Surprisal-Steigung an Wechselpunkten ---")
    print(f"beta = {slope_cs:.4f}, SE = {se_cs:.4f}, "
          f"95%-KI [{ci_low:.4f}; {ci_high:.4f}], "
          f"t({df_cs:.1f}) = {t_cs:.2f}, p = {p_cs:.4f}")
    print("(Freiheitsgrade konservativ genähert; der Wert beruht auf demselben "
          "Modell und nicht auf einer erneuten Schätzung.)")

    pd.DataFrame([{
        "term": "surprisal_z at cs_point = 1", "estimate": slope_cs, "std_error": se_cs,
        "conf_low": ci_low, "conf_high": ci_high, "t_stat": t_cs, "df": df_cs,
        "p_value": p_cs,
    }]).to_csv(OUTPUT_DIR / "contrast_cs_slope.csv", index=False)

except Exception as exc:
    print(f"\nHinweis: Kontrast konnte nicht berechnet werden ({exc}).")

# Ergebnisse speichern
data.to_csv(OUTPUT_DIR / "main_analysis_with_surprisal_filtered.csv", index=False)
effects.to_csv(OUTPUT_DIR / "fixed_effects_breakdown.csv", index=False)

print(f"\nGespeichert in {OUTPUT_DIR}/")
