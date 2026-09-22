"""
Korpusüberblick TuGeBiC (TR-DE)
===============================

Markiert die Code-Switching-Punkte, gibt Kennzahlen zum Korpus aus
(Tokens, Sätze, Sprachverteilung) und erstellt die POS-Verteilungsgrafiken.
Zusätzlich: durchschnittliche Wortlänge und Silbenzahl je Sprache und POS.

Eingabe : TuGeBiC_Twitter_TRDE.csv (die Spalte CS wird bei Bedarf ergänzt)
Ausgabe : Grafik/TR_DE_POS.png, Grafik/TR_DE_POS_gesamt.png
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from a_cs_features_tr import count_syllables


# Code-Switching-Punkte
def add_cs_columns(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """
    Erzeugt die Spalte CS.

    Ein Wort ist ein CS-Punkt (1), wenn seine Sprache von der des UNMITTELBAR
    vorangehenden Wortes im selben Satz abweicht. OTHER-Token (Satzzeichen,
    Symbole, Usernamen) sind nie CS-Punkte und unterbrechen die Vergleichskette,
    d. h. sie werden nicht übersprungen.
    """
    df_copy = df.copy()
    if force or "CS" not in df_copy.columns:
        df_copy["CS"] = [
            int(
                i > 0
                and df_copy.loc[i, "ID"] == df_copy.loc[i - 1, "ID"]
                and df_copy.loc[i, "LANG"] != df_copy.loc[i - 1, "LANG"]
                and df_copy.loc[i, "LANG"] != "OTHER"
                and df_copy.loc[i - 1, "LANG"] != "OTHER"
            )
            for i in range(len(df_copy))
        ]
    return df_copy


def report_switches_hidden_by_other(df: pd.DataFrame) -> None:
    """
    Nur Diagnose – verändert die Spalte CS nicht.
    Zählt Sprachwechsel, die nicht als CS-Punkt erfasst werden.
    """
    if not {"ID", "LANG"}.issubset(df.columns):
        return

    hidden = 0
    for _, g in df.groupby("ID", sort=False):
        last_real = None        # zuletzt gesehene Sprache ungleich OTHER
        gap_since_last = False  # lag ein OTHER-Token dazwischen?
        for lang in g["LANG"]:
            if lang == "OTHER":
                if last_real is not None:
                    gap_since_last = True
                continue
            if last_real is not None and lang != last_real and gap_since_last:
                hidden += 1
            last_real = lang
            gap_since_last = False

    print(f"Sprachwechsel, die durch dazwischenliegende OTHER-Token nicht als "
          f"CS-Punkt erfasst wurden: {hidden}")
    print(" ")


# Kennzahlen
def compute_and_print_cs_totals(df: pd.DataFrame) -> None:
    print(f"Gesamtanzahl der Code-Switching-Punkte (CS): {int(df.get('CS', pd.Series(0)).sum())}")
    print(" ")


def compute_and_print_sentence_stats(df: pd.DataFrame) -> None:
    """Gibt Token-, Wort-, Zeichen- und Satzzahlen sowie die Sprachverteilung aus."""
    # Eine Zeile = ein Token. Die Zeichenzahl wird getrennt ausgewiesen und darf
    # nicht mit der Wortzahl verwechselt werden.
    total_tokens = len(df)
    total_characters = df["WORD"].astype(str).str.len().sum()
    total_sentences = df["ID"].nunique() if "ID" in df else 0

    # Tokens ohne OTHER – das entspricht "Wörtern" im linguistischen Sinn.
    total_words_without_other = int((df["LANG"] != "OTHER").sum()) if "LANG" in df else total_tokens

    monolingual, bilingual = 0, 0
    if {"ID", "LANG"}.issubset(df.columns):
        for _, g in df.groupby("ID"):
            langs = [l for l in g["LANG"].unique() if l != "OTHER"]
            if len(langs) <= 1:
                monolingual += 1
            else:
                bilingual += 1

    counts = df["LANG"].value_counts()
    print(f"Tokens insgesamt (inkl. OTHER): {total_tokens}")
    print(f"Wörter ohne OTHER: {total_words_without_other}")
    print(f"Zeichen insgesamt (Buchstabenanzahl, NICHT Wortanzahl): {total_characters}")
    print(f"Sätze insgesamt: {total_sentences}")
    print(f"Monolinguale Sätze: {monolingual}")
    print(f"Bilinguale Sätze: {bilingual}")
    for lang in ["DE", "TR", "ENG", "LANG3", "MIXED", "AMBIG", "OTHER"]:
        print(f"{lang}: {int(counts.get(lang, 0))}")
    print(" ")


# Grafiken
def _plot_pos_counts(pos_counts, output_path: str, plot_title: str, color: str) -> None:
    """Balkendiagramm einer POS-Häufigkeitsverteilung."""
    print(pos_counts)
    pos_counts.plot(kind="bar", color=color, edgecolor="black", figsize=(10, 6))
    plt.title(plot_title, fontsize=14)
    plt.xlabel("POS", fontsize=12)
    plt.ylabel("Verteilung", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_pos_distribution(df: pd.DataFrame, output_path: str, plot_title: str) -> None:
    """POS-Verteilung an CS-Punkten (ohne OTHER)."""
    if not {"CS", "LANG", "POS"}.issubset(df.columns):
        print("Warnung: POS-Verteilung wird übersprungen (fehlende Spalten).")
        return
    counts = df[(df["CS"] == 1) & (df["LANG"] != "OTHER")]["POS"].value_counts()
    print(f"POS Verteilung an Code-Switching Punkten (außer OTHER) für {plot_title}:")
    _plot_pos_counts(counts, output_path, plot_title, "skyblue")


def plot_pos_distribution_full_corpus(df: pd.DataFrame, output_path: str, plot_title: str) -> None:
    """POS-Verteilung im gesamten Korpus (ohne OTHER)."""
    if not {"LANG", "POS"}.issubset(df.columns):
        print("Warnung: POS-Verteilung wird übersprungen (fehlende Spalten).")
        return
    counts = df[df["LANG"] != "OTHER"]["POS"].value_counts()
    print(f"POS Verteilung im gesamten TR-DE-Korpus (außer OTHER) für {plot_title}:")
    _plot_pos_counts(counts, output_path, plot_title, "mediumaquamarine")


# Durchschnittswerte je Sprache und POS
def calculate_general_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Durchschnittliche Wortlänge (Buchstaben) je LANG und POS."""
    df_clean = df.copy()
    df_clean["WORD_LEN"] = df_clean["WORD"].astype(str).apply(len)
    stats = df_clean.groupby(["LANG", "POS"])["WORD_LEN"].mean().reset_index()
    stats.rename(columns={"WORD_LEN": "AVG_LENGTH"}, inplace=True)
    return stats.sort_values(by=["POS", "LANG"])


def calculate_general_syllable_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Durchschnittliche Silbenzahl je LANG und POS (nur TR, DE, ENG)."""
    df_clean = df[df["LANG"].str.upper().isin(["TR", "DE", "ENG"])].copy()
    df_clean["SYLLABLE_COUNT"] = df_clean.apply(
        lambda row: count_syllables(str(row["WORD"]), str(row["LANG"])), axis=1
    )
    stats = df_clean.groupby(["LANG", "POS"])["SYLLABLE_COUNT"].mean().reset_index()
    stats.rename(columns={"SYLLABLE_COUNT": "AVG_SYLLABLES"}, inplace=True)
    return stats.sort_values(by=["POS", "LANG"])


# Ablauf
def main(input_csv: str, plot_output_path: str, plot_title: str) -> None:
    df = pd.read_csv(input_csv)
    if "CS" not in df.columns:
        print(f"Spalte CS wird zu {os.path.basename(input_csv)} hinzugefügt ...")
        df = add_cs_columns(df)
        df.to_csv(input_csv, index=False)
        print(f"{os.path.basename(input_csv)} wurde mit der Spalte CS aktualisiert.")
    else:
        print(f"{os.path.basename(input_csv)} enthält bereits die Spalte CS.")
        df = add_cs_columns(df)

    compute_and_print_cs_totals(df)
    compute_and_print_sentence_stats(df)
    report_switches_hidden_by_other(df)

    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
    plot_pos_distribution(df, plot_output_path, plot_title)
    plot_pos_distribution_full_corpus(
        df,
        os.path.join("Grafik", "TR_DE_POS_gesamt.png"),
        "DE/TR POS Verteilung im gesamten TR-DE-Korpus",
    )

    for titel, stats in [("WORTLÄNGEN JE POS", calculate_general_length_stats(df)),
                         ("SILBENZAHLEN JE POS", calculate_general_syllable_stats(df))]:
        print("\n" + "-" * 80)
        print(titel)
        print("-" * 80)
        print(stats.to_string(index=False))
        print("-" * 80)


if __name__ == "__main__":
    input_data = "TuGeBiC_Twitter_TRDE.csv"
    plot_output = os.path.join("Grafik", "TR_DE_POS.png")
    plot_title = "DE/TR POS Verteilung an Code-Switching Punkten"

    print(f"\n{'=' * 80}")
    print(f"Verarbeite {os.path.basename(input_data)}: CS-Analyse und POS-Verteilung")
    print(f"{'-' * 80}")

    main(input_data, plot_output, plot_title)
    print("\n" + "=" * 80)
