"""
Modellvergleich: XGLM-564M vs. XGLM-1.7B
========================================

Dieselbe LMEM-Analyse wie in g_Lmem_Surprisal.py, nacheinander für zwei
Surprisal-Quellen. Geschätzt wird jeweils nur das Interaktionsmodell

    log_reading_time ~ surprisal_z * cs_point
                       + (1 + surprisal_z | participant_id) + (1 | item_order)

sowie ein Basismodell ohne Surprisal als Referenz für den
Log-Likelihood-Vergleich.

Beide Sprachmodelle teilen Architektur, Tokenizer und Trainingsdaten und
unterscheiden sich nur in der Kapazität. Nach Oh & Schuler (2023) ist zu
erwarten, dass das kleinere Modell die Lesezeiten besser vorhersagt.

Eingabe : main_analysis_dataframe_lmem_ready.csv
          surprisal_values.csv        (XGLM-1.7B)
          small_surprisal_values.csv  (XGLM-564M)
Ausgabe : Bspr_result/modellvergleich_summary.csv

Beide Analysen laufen auf identisch gefilterten Zeilen, damit der Vergleich
fair ist; das Skript prüft das und bricht bei Abweichung ab.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from pymer4.models import lmer

# Einstellungen
READING_TIME_FILE = Path("/Bspr_result/main_analysis_dataframe_lmem_ready.csv")
OUTPUT_DIR = Path("Bspr_result")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SURPRISAL_SOURCES = {
    "XGLM-564M": Path("small_surprisal_values.csv"),
    "XGLM-1.7B": Path("surprisal_values.csv"),
}

RT_LOWER, RT_UPPER = 100, 3000

# Der Log-Likelihood-Vergleich zwischen Basismodell (ohne Surprisal) und
# Interaktionsmodell erfordert ML statt REML, da sich die festen Effekte
# unterscheiden.
USE_REML = False

RANDOM_EFFECTS = "+ (1 + surprisal_z | participant_id) + (1 | item_order)"


# Datenaufbereitung (identisch zu g_Lmem_Surprisal.py)
def prepare_data(reading_time_file, surprisal_file, verbose=True):
    reading_times = pd.read_csv(reading_time_file)
    surprisal = pd.read_csv(surprisal_file)[["item_order", "word_position", "surprisal"]]

    data = reading_times.merge(surprisal, on=["item_order", "word_position"], how="left")
    data = data.dropna(subset=["surprisal", "first_pass_reading_time"]).reset_index(drop=True)

    # Erstes Wort jedes Satzes ausschließen (beim BSPR sofort sichtbar).
    n_before = len(data)
    data = data[data["word_position"] != 1].reset_index(drop=True)
    n_first_word = n_before - len(data)

    # Ausreißerfilter nach Oh & Schuler (2023) und Chen et al. (2026).
    n_before = len(data)
    data = data[(data["first_pass_reading_time"] >= RT_LOWER)
                & (data["first_pass_reading_time"] <= RT_UPPER)].reset_index(drop=True)
    n_rt = n_before - len(data)

    if verbose:
        print(f"  Erstes Wort entfernt: {n_first_word} | "
              f"RT-Filter entfernt: {n_rt} | verbleiben: {len(data)}")

    data["surprisal_z"] = (data["surprisal"] - data["surprisal"].mean()) / data["surprisal"].std()
    data["log_reading_time"] = np.log(data["first_pass_reading_time"])
    data["cs_point"] = data["cs_point"].astype(int)
    data["participant_id"] = data["participant_id"].astype(str)
    data["item_order"] = data["item_order"].astype(str)
    return data


def as_pandas(result):
    return result.to_pandas() if hasattr(result, "to_pandas") else pd.DataFrame(result)


def get_row(effects_table, term):
    return effects_table[effects_table["term"] == term].iloc[0]


def fit_model(formula, data_pl, use_reml=USE_REML):
    """Schätzt ein Modell; ältere pymer4-Versionen kennen REML in fit() nicht."""
    m = lmer(formula, data=data_pl)
    try:
        m.fit(REML=use_reml)
    except TypeError:
        if not use_reml:
            print("  Hinweis: Diese pymer4-Version akzeptiert REML nicht in fit(); "
                  "der Log-Likelihood-Vergleich beruht daher auf REML-Werten und ist "
                  "nur eingeschränkt interpretierbar.")
        m.fit()
    return m


def try_loglik(model):
    """Liest die Log-Likelihood aus; der Attributname variiert je pymer4-Version."""
    for attr in ("logLike", "log_likelihood", "loglike"):
        val = getattr(model, attr, None)
        if isinstance(val, (int, float)):
            return float(val)

    stats = getattr(model, "result_fit_stats", None)
    if stats is not None:
        try:
            stats_df = as_pandas(stats)
            for col in stats_df.columns:
                if "log" in col.lower() and "lik" in col.lower():
                    return float(stats_df[col].iloc[0])
        except Exception:
            pass
    return np.nan


# Analyse je Surprisal-Quelle
def run_analysis(label, surprisal_file):
    print(f"\n{'=' * 70}\n{label}  ({surprisal_file.name})\n{'=' * 70}")
    data = prepare_data(READING_TIME_FILE, surprisal_file)
    data_pl = pl.from_pandas(data)

    model = fit_model(f"log_reading_time ~ surprisal_z * cs_point {RANDOM_EFFECTS}", data_pl)
    # Basismodell ohne Surprisal: Referenz für den Log-Likelihood-Vergleich.
    baseline = fit_model("log_reading_time ~ 1 + (1 | participant_id) + (1 | item_order)",
                         data_pl)

    effects = as_pandas(model.result_fit)

    print("\n--- Modell: surprisal_z * cs_point ---")
    print(effects.to_string(index=False))

    ll_model = try_loglik(model)
    ll_baseline = try_loglik(baseline)
    delta_ll = (ll_model - ll_baseline
                if not (np.isnan(ll_model) or np.isnan(ll_baseline)) else np.nan)

    rows = []
    for group, term in [
        ("Surprisal (Non-CS slope)", "surprisal_z"),
        ("CS main effect", "cs_point"),
        ("Interaction (surprisal x CS)", "surprisal_z:cs_point"),
    ]:
        r = get_row(effects, term)
        rows.append({
            "model": label, "group": group, "n_obs": len(data),
            "estimate": r["estimate"], "se": r["std_error"],
            "ci_low": r["conf_low"], "ci_high": r["conf_high"],
            "t_stat": r["t_stat"], "p_value": r["p_value"],
            "delta_loglik": delta_ll,
        })

    return pd.DataFrame(rows), len(data)


all_results, n_obs_seen = [], {}
for label, path in SURPRISAL_SOURCES.items():
    res, n_obs = run_analysis(label, path)
    all_results.append(res)
    n_obs_seen[label] = n_obs

if len(set(n_obs_seen.values())) != 1:
    raise ValueError("Die beiden Analysen laufen auf unterschiedlich vielen Zeilen "
                     f"({n_obs_seen}). Ein Vergleich wäre dadurch verzerrt.")

summary = pd.concat(all_results, ignore_index=True)
summary.to_csv(OUTPUT_DIR / "modellvergleich_summary.csv", index=False)

print(f"\n{'=' * 70}\nVERGLEICH\n{'=' * 70}")
print(summary.to_string(index=False))

if summary["delta_loglik"].notna().any():
    print("\nDelta Log-Likelihood gegenüber dem Basismodell ohne Surprisal "
          "(höher = bessere Vorhersage der Lesezeiten):")
    print(summary.groupby("model")["delta_loglik"].first().to_string())
else:
    print("\nHinweis: Log-Likelihood konnte aus dieser pymer4-Version nicht ausgelesen "
          "werden. Der Vergleich stützt sich dann auf Steigungen und t-Werte.")

print(f"\nGespeichert in {OUTPUT_DIR}/")
