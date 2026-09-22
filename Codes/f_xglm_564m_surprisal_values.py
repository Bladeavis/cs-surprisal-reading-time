"""
Surprisal-Werte auf Wortebene mit XGLM-564M
===========================================

Methodisch identisch zu e_XGLM_surprisal_values.py; nur Checkpoint und
Ausgabedatei unterscheiden sich. Genau darin liegt der Zweck: XGLM-564M und
XGLM-1.7B teilen Architektur, Tokenizer und Trainingsdaten, sodass Unterschiede
in der Vorhersage der Lesezeiten allein auf die Modellkapazität zurückgehen
(vgl. Oh & Schuler 2023).

Eingabe : word_cs.xlsx (Spalten: Type, Sentence-Type, Sentence, Word, CS)
Ausgabe : Bspr_result/small_surprisal_values.csv

Methode
-------
Wörter werden über inkrementelle Tokenisierung den Subword-Token zugeordnet:
Der Satz wird Wort für Wort verlängert und jedes Mal neu tokenisiert; die neu
hinzugekommenen Token-IDs gehören zum jeweiligen Wort.

Das Wort-Surprisal ist die Summe von -log P(Token | vorheriger Kontext) über
alle Subword-Token des Wortes (Teacher Forcing, ein Forward-Pass pro Satz).

Hinweis: Der Checkpoint ist mit rund 2,5 GB deutlich kleiner als das
1,7B-Modell und läuft daher auch auf der CPU in vertretbarer Zeit.
"""

import math
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 0) Einstellungen
CS_PATH = Path("word_cs.xlsx")
OUT_DIR = Path("Bspr_result")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nur diese beiden Zeilen unterscheiden sich von e_XGLM_surprisal_values.py.
MODEL_NAME = "facebook/xglm-564M"
OUT_FILE = "small_surprisal_values.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Beginnt bei 2, damit item_order zu main_analysis_dataframe_lmem_ready.csv
# passt (dort lückenlos von 2 bis 53, in der Satzreihenfolge aus word_cs.xlsx).
FIRST_ITEM_ORDER = 2

LN2 = math.log(2)

# 1) Modell und Tokenizer laden
print(f"Lade Modell: {MODEL_NAME} (device={DEVICE}) ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
model.to(DEVICE)
model.eval()


# 2) Zuordnung Wort -> Subword-Token
def align_words_to_tokens(words, tokenizer):
    """
    Ordnet jedem Wort seine Token-IDs zu.

    Rückgabe: die Token-IDs des ganzen Satzes sowie je Wort die halboffene
    Tokenspanne (start, end) – end ist ausgeschlossen.
    """
    prev_ids = tokenizer.encode("", add_special_tokens=True)
    word_token_spans = []
    running_text = ""

    for w in words:
        running_text = w if running_text == "" else running_text + " " + w
        ids = tokenizer.encode(running_text, add_special_tokens=True)

        # Nicht jeder Tokenizer ist präfixstabil; die Länge des gemeinsamen
        # Präfixes ist daher die sicherste Zuordnung.
        common_len = 0
        max_common = min(len(prev_ids), len(ids))
        while common_len < max_common and prev_ids[common_len] == ids[common_len]:
            common_len += 1

        start, end = common_len, len(ids)
        if end <= start:
            # Seltener Fall: Das Wort erzeugt kein neues Token. Es wird
            # trotzdem mindestens ein Token zugewiesen, damit die Spanne gilt.
            start = max(start - 1, len(prev_ids))
            end = max(end, start + 1)

        word_token_spans.append((start, end))
        prev_ids = ids

    return prev_ids, word_token_spans


@torch.no_grad()
def compute_sentence_surprisal(sentence_words, tokenizer, model, device):
    """Berechnet das Surprisal (in nats) je Wort eines Satzes."""
    full_ids, spans = align_words_to_tokens(sentence_words, tokenizer)
    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids).logits[0]                # (seq_len, vocab)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # Surprisal von Token t: -log P(token_t | token_<t), also über logits[t-1].
    token_surprisal_nats = torch.full((len(full_ids),), float("nan"))
    for t in range(1, len(full_ids)):
        token_surprisal_nats[t] = -log_probs[t - 1, full_ids[t]]

    word_surprisal_nats, n_tokens_per_word = [], []
    for start, end in spans:
        vals = token_surprisal_nats[start:end]
        vals = vals[~torch.isnan(vals)]
        word_surprisal_nats.append(float(vals.sum()) if len(vals) else float("nan"))
        n_tokens_per_word.append(end - start)

    return word_surprisal_nats, n_tokens_per_word


# 3) word_cs.xlsx satzweise verarbeiten
df_cs = pd.read_excel(CS_PATH)
sentence_order = df_cs["Sentence"].drop_duplicates().tolist()

all_rows = []
for i, sentence in enumerate(sentence_order):
    item_order = FIRST_ITEM_ORDER + i
    grp = df_cs.loc[df_cs["Sentence"] == sentence].reset_index(drop=True)
    words_with_punct = sentence.split()  # Wörter inkl. Satzzeichen

    if len(words_with_punct) != len(grp):
        print(f"WARNUNG: Wortzahl stimmt nicht für item_order={item_order} "
              f"({len(words_with_punct)} vs. {len(grp)}); Satz wird übersprungen.")
        continue

    word_surprisal_nats, n_tokens = compute_sentence_surprisal(
        words_with_punct, tokenizer, model, DEVICE
    )

    for pos in range(len(grp)):
        nats = word_surprisal_nats[pos]
        all_rows.append({
            "item_order": item_order,
            "word_position": pos + 1,
            "word": grp["Word"][pos],
            "word_with_punct": words_with_punct[pos],
            "cs_point": int(grp["CS"][pos]),
            "type": grp["Type"][pos],
            "sentence_type": grp["Sentence-Type"][pos],
            "sentence": sentence,
            "surprisal_nats": nats,
            "surprisal_bits": nats / LN2 if nats == nats else float("nan"),  # nats == nats: NaN-Test
            "n_subword_tokens": n_tokens[pos],
        })
    print(f"item_order={item_order} fertig ({len(grp)} Wörter).")

df_surprisal = pd.DataFrame(all_rows)

# 4) Speichern
df_surprisal["surprisal"] = df_surprisal["surprisal_bits"]  # Standardeinheit: bits
df_surprisal.to_csv(OUT_DIR / OUT_FILE, index=False)
print(f"\nGespeichert: {OUT_DIR / OUT_FILE}  ({len(df_surprisal)} Zeilen)")

# Deskriptive Statistik
print("\nMittleres Surprisal (bits) nach CS-Status:")
print(df_surprisal.groupby("cs_point")["surprisal_bits"]
      .agg(["count", "mean", "std", "min", "max"]).to_string())

print("\nErste Zeilen:")
print(df_surprisal.head(14)[["item_order", "word_position", "word", "cs_point",
                             "surprisal_bits", "n_subword_tokens"]].to_string(index=False))
