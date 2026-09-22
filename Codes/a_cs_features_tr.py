"""
Berechnet Wortmerkmale an Code-Switching-Punkten im Sprachpaar Türkisch-Deutsch.

Ablauf: CSV einlesen -> nur CS-Zeilen (CS == 1) -> jedes Wort in die jeweils andere
Sprache übersetzen -> Zeichenlänge und Silbenzahl bestimmen -> vergleichen, welche
Sprache kürzer ist. Die Zeilen werden parallel verarbeitet (ThreadPoolExecutor).

Eingabe:  CSV mit den Spalten ID, WORD, LANG, CS
Ausgabe:  CSV mit Übersetzungen, Längen, Silbenzahlen sowie KURZ_LEN und
          KURZ_SILBEN ('TR', 'DE' oder 'TR/DE' bei Gleichstand)
"""

import os
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from syllabreak import Syllabreak
from deep_translator import GoogleTranslator

# Spaltenkürzel -> Sprachcode für Übersetzer bzw. Silbentrennung
LANGS = {'DE': 'de', 'TR': 'tr'}
SYLL_CODES = {'DE': 'deu', 'TR': 'tur'}

COLS = ['ID', 'WORD', 'LANG',
        'DE', 'DE_LEN', 'DE_SILBEN',
        'TR', 'TR_LEN', 'TR_SILBEN',
        'KURZ_LEN', 'KURZ_SILBEN']

_syllabreaker = Syllabreak("-")


# Übersetzung
@lru_cache(maxsize=10000)
def _translate_cached(text: str, src: str, dest: str) -> str:
    """Übersetzt einen Text; bei Fehlern bleibt das Original stehen."""
    try:
        return GoogleTranslator(source=src or 'auto', target=dest).translate(text)
    except Exception as e:
        print(f"Übersetzungsfehler: {text[:20]}... -> {e}")
        return text


def translate(text: str, src: str, dest: str) -> str:
    """Prüft die Eingabe und ruft die gecachte Übersetzung auf."""
    if not text:
        return text
    text = str(text).strip()
    return _translate_cached(text, src, dest) if text else text


# Silbenzählung
def count_syllables(word: str, lang: str) -> int:
    """Zählt die Silben eines Wortes; 0 bei leerer Eingabe oder unbekannter Sprache."""
    if not isinstance(word, str) or not word.strip():
        return 0

    code = SYLL_CODES.get(lang.upper())
    if not code:
        print(f"Warnung: Unbekannte Sprache '{lang}' — Silben werden nicht gezählt.")
        return 0

    try:
        return len(_syllabreaker.syllabify(word.strip(), lang=code).split("-"))
    except Exception as e:
        print(f"Warnung: '{word}' ({lang}) nicht trennbar: {e}")
        return 1


# Vergleich
def kuerzeste(tr_val, de_val) -> str:
    """Gibt die Sprache mit dem kleineren Wert zurück: 'TR', 'DE' oder 'TR/DE'."""
    vals = {k: v for k, v in (('TR', tr_val), ('DE', de_val)) if v is not None}
    if not vals:
        return ''
    min_val = min(vals.values())
    return '/'.join(k for k in ('TR', 'DE') if vals.get(k) == min_val)


# Verarbeitung
def process_row(row: pd.Series) -> list:
    """Bildet aus einer Eingabezeile eine Ergebniszeile gemäß COLS."""
    word, lang, word_id = row['WORD'], row['LANG'], row['ID']
    src = LANGS.get(lang.upper(), 'auto')

    # Bei LANG == 'TR' oder 'DE' bleibt das Wort in seiner Ausgangssprache stehen.
    # Alle anderen Werte (z. B. MIXED, LANG3) werden per Autoerkennung des Quelltexts
    # in beide Sprachen übersetzt und in beide Spaltenpaare geschrieben.
    trans = {l: word if l == lang else translate(word, src, code)
             for l, code in LANGS.items()}
    if lang in LANGS:
        trans[lang] = word

    lengths = {l: len(str(trans[l])) for l in LANGS}
    syllables = {l: count_syllables(trans[l], l) for l in LANGS}

    return [word_id, word, lang,
            trans['DE'], lengths['DE'], syllables['DE'],
            trans['TR'], lengths['TR'], syllables['TR'],
            kuerzeste(lengths['TR'], lengths['DE']),
            kuerzeste(syllables['TR'], syllables['DE'])]


def build_dataframe(df: pd.DataFrame, max_workers: int = 10) -> pd.DataFrame:
    """Verarbeitet alle Zeilen parallel und meldet den Fortschritt."""
    data = []
    total = len(df)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                data.append(future.result())
                if idx % 50 == 0 or idx == total:
                    print(f" Verarbeitet: {idx}/{total} ({idx * 100 // total}%)")
            except Exception as e:
                print(f"  ✗ Zeilenfehler: {e}")

    return pd.DataFrame(data, columns=COLS)


def main(input_csv: str, output_csv: str):
    """Liest die CSV, verarbeitet die CS-Zeilen und schreibt das Ergebnis."""
    start = time.time()
    cs_rows = pd.read_csv(input_csv).query("CS == 1")
    print(f"Insgesamt {len(cs_rows)} Zeilen werden verarbeitet\n")

    build_dataframe(cs_rows).to_csv(output_csv, index=False)

    print(f"\n{'=' * 60}")
    print(f"Abgeschlossen! Dauer: {(time.time() - start) / 60:.2f} Minuten")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    input_data = "TuGeBiC_Twitter_TRDE.csv"
    output_data = "CS_Punkte_TR22.csv"

    print(f"Processing {os.path.basename(input_data)} for TR and DE comparison.")
    main(input_data, output_data)
    print("-" * 80)
