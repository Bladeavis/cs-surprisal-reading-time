"""
Aufbereitung der BSPR-Rohdaten (PCIbex)
=======================================

Eingabe : 010826_results_prod.csv (Rohausgabe von PennController)
          word_cs.xlsx            (CS-Markierung je Wort)
Ausgabe : bereinigte CSV-Dateien im Ordner Bspr_result/

Aufbau der Rohdatei (aus den Daten selbst abgeleitet)
-----------------------------------------------------
Die ersten 7 Felder haben alle Zeilen gemeinsam:
    1 SubmitTime, 2 ParticipantHash, 3 Controller, 4 ItemOrder,
    5 InnerElementNumber, 6 Label (Bedingung bzw. "intro"), 7 LatinSquareGroup

Controller == "Form"      -> 8 FieldName, 9 FieldValue
Controller == "Question"  -> 8 Frage, 9 Antwort, 10 IsCorrect (1/0/NULL),
                             11 Antwortzeit in ms
Controller == "ReversibleDashedSentence2" (Lesezeile):
    8 WordPosition (ab 1), 9 Word, 10 WordIndex (ab 0),
    11..N-1 Ereignisliste variabler Länge, letztes Feld: Newline (true/false)

Ereignisliste:
    [first_pass_RT, erste Bewegungsrichtung,
     (Tastendruck-Nr., RT, Richtung), ...]
  - first_pass_RT: Zeit auf dem Wort beim ersten Erscheinen, bis die Person
    per Vorwärts- oder Rückwärtstaste weitergeht (First-Pass-Lesezeit in ms).
  - Richtung: 1 = vorwärts, -1 = rückwärts.
  - Die weiteren Tripel beschreiben je einen erneuten Besuch; ihre Summe
    ergibt die rereading_time.

Hinweis zu negativen Werten: Lesezeiten sind nie negativ. "-1" steht immer
für eine Rückwärtsbewegung im Richtungsfeld, nie für eine Lesezeit.
"""

from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

RAW_PATH = Path("010826_results_prod.csv")
CS_PATH = Path("word_cs.xlsx")
OUT_DIR = Path("Bspr_result")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_COLS = ["submit_time", "participant_hash", "controller",
               "item_order", "inner_element_number", "label", "latin_square_group"]

# 0) Rohdatei zeilenweise einlesen, Kommentarzeilen überspringen.
with open(RAW_PATH, "r", encoding="utf-8") as f:
    raw_lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#") and ln.strip()]

print(f"Datenzeilen ohne Kommentare: {len(raw_lines)}")

form_rows, question_rows, word_rows = [], [], []
unparsed_rows = []    # nicht lesbare oder unerwartete Zeilen
missing_rt_rows = []  # Wortzeilen ohne Lesezeitdaten

for line_no, line in enumerate(raw_lines, start=1):
    fields = line.split(",")
    if len(fields) < 8:
        unparsed_rows.append({"line_no": line_no, "raw": line, "reason": "Weniger als 8 Felder"})
        continue

    common, rest = fields[:7], fields[7:]
    controller = common[2]

    try:
        if controller == "Form":
            form_rows.append(common + [rest[0], ",".join(rest[1:])])

        elif controller == "Question":
            if len(rest) != 4:
                unparsed_rows.append({"line_no": line_no, "raw": line,
                                      "reason": f"Unerwartete Feldzahl bei Question: {len(rest)}"})
                continue
            question_rows.append(common + list(rest))

        elif controller == "ReversibleDashedSentence2":
            word_position, word_raw = rest[0], rest[1]
            event_fields = rest[2:-1]  # word_index + Ereignisliste
            newline_flag = rest[-1]

            if len(event_fields) == 0 or event_fields[0] == "":
                # Lesezeitdaten fehlen vollständig (seltener Sonderfall).
                missing_rt_rows.append({"line_no": line_no, "raw": line,
                                        "reason": "word_index/RT-Daten leer"})
                word_rows.append(common + [word_position, word_raw, np.nan, [], newline_flag,
                                           "missing_reading_time"])
                continue

            word_index = int(event_fields[0])
            nums = event_fields[1:]

            # Gültig sind: 2 Startwerte + beliebig viele vollständige Tripel.
            if len(nums) < 2 or (len(nums) - 2) % 3 != 0:
                unparsed_rows.append({"line_no": line_no, "raw": line,
                                      "reason": f"Unerwartete Länge der Ereignisliste: {len(nums)}"})
                word_rows.append(common + [word_position, word_raw, word_index, [], newline_flag,
                                           "unparsed_event_list"])
                continue

            nums = [int(n) for n in nums]
            events = [("first_pass", None, nums[0], nums[1])]
            for i in range(2, len(nums), 3):
                kp_no, rt, direction = nums[i:i + 3]
                events.append(("reread", kp_no, rt, direction))

            word_rows.append(common + [word_position, word_raw, word_index, events,
                                       newline_flag, "ok"])

        else:
            unparsed_rows.append({"line_no": line_no, "raw": line,
                                  "reason": f"Unbekannter Controller: {controller}"})
    except Exception as e:
        unparsed_rows.append({"line_no": line_no, "raw": line, "reason": f"Exception: {e}"})

print(f"Form-Zeilen: {len(form_rows)} | Question-Zeilen: {len(question_rows)} | "
      f"Wortzeilen: {len(word_rows)}")
print(f"Nicht lesbare Zeilen: {len(unparsed_rows)} | Zeilen ohne RT: {len(missing_rt_rows)}")

# 1) Form-Daten: Zuordnung Hash <-> ParticipantID.
df_form = pd.DataFrame(form_rows, columns=COMMON_COLS + ["field_name", "field_value"])
df_participants = (
    df_form[df_form["field_name"] == "ParticipantID"]
    [["participant_hash", "field_value"]]
    .rename(columns={"field_value": "participant_id"})
    .drop_duplicates()
    .reset_index(drop=True)
)
print(f"\nAnzahl der Versuchspersonen: {df_participants.shape[0]}")

id_map = dict(zip(df_participants["participant_hash"], df_participants["participant_id"]))

# 2) Verständnisfragen.
df_q = pd.DataFrame(question_rows, columns=COMMON_COLS +
                    ["question_text_raw", "given_answer_raw", "is_correct_raw", "answer_time_ms_raw"])

df_q["participant_id"] = df_q["participant_hash"].map(id_map)
df_q["item_order"] = df_q["item_order"].astype(int)
df_q = df_q.rename(columns={"label": "experimental_condition"})
df_q["question_text"] = df_q["question_text_raw"].apply(unquote)
df_q["given_answer"] = df_q["given_answer_raw"].apply(unquote)
df_q["is_correct"] = df_q["is_correct_raw"].replace({"NULL": np.nan}).astype(float)
df_q["answer_time_ms"] = pd.to_numeric(df_q["answer_time_ms_raw"], errors="coerce")

df_questions = df_q[[
    "participant_id", "participant_hash", "item_order", "experimental_condition",
    "latin_square_group", "question_text", "given_answer", "is_correct", "answer_time_ms",
    "submit_time"
]].sort_values(["participant_id", "item_order"]).reset_index(drop=True)

# 3) Trefferquote je Versuchsperson.
acc = (
    df_questions.groupby("participant_id")
    .agg(total_questions=("is_correct", "size"),
         correct_answers=("is_correct", lambda s: int(s.sum(skipna=True))))
    .reset_index()
)
acc["incorrect_answers"] = acc["total_questions"] - acc["correct_answers"]
acc["accuracy"] = acc["correct_answers"] / acc["total_questions"]
acc["error_rate"] = acc["incorrect_answers"] / acc["total_questions"]
# Fehlerquote über 1/3 gilt als ungeeignet.
acc["suitability"] = np.where(acc["error_rate"] > (1 / 3), "not suitable", "suitable")

df_participant_accuracy = acc.merge(df_participants, on="participant_id", how="left")[
    ["participant_id", "participant_hash", "total_questions", "correct_answers",
     "incorrect_answers", "accuracy", "error_rate", "suitability"]
].sort_values("participant_id").reset_index(drop=True)

print("\nTrefferquote je Versuchsperson:")
print(df_participant_accuracy.to_string(index=False))

# 4) Lesezeiten: Ereignis- und Wortebene.
word_cols = COMMON_COLS + ["word_position", "word_raw", "word_index_0based",
                           "events", "newline_flag", "parse_status"]
df_words_raw = pd.DataFrame(word_rows, columns=word_cols)

df_words_raw["participant_id"] = df_words_raw["participant_hash"].map(id_map)
df_words_raw["item_order"] = df_words_raw["item_order"].astype(int)
df_words_raw = df_words_raw.rename(columns={"label": "experimental_condition"})
df_words_raw["word_position"] = df_words_raw["word_position"].astype(int)
df_words_raw["word"] = df_words_raw["word_raw"].apply(unquote)
df_words_raw["newline_flag"] = df_words_raw["newline_flag"].map({"true": True, "false": False})

# --- Ereignisebene: eine Zeile je Besuch. ---
event_records = []
for _, row in df_words_raw.iterrows():
    for visit_no, (event_type, kp_no, rt, direction) in enumerate(row["events"], start=1):
        event_records.append({
            "participant_id": row["participant_id"],
            "participant_hash": row["participant_hash"],
            "item_order": row["item_order"],
            "experimental_condition": row["experimental_condition"],
            "latin_square_group": row["latin_square_group"],
            "word_position": row["word_position"],
            "word": row["word"],
            "visit_number": visit_no,
            "event_type": event_type,              # first_pass / reread
            "keypress_number_trial_local": kp_no,  # bei first_pass NaN
            "reading_time_ms": rt,
            "move_direction": "forward" if direction == 1 else "backward",
        })

df_reading_events = pd.DataFrame(event_records).sort_values(
    ["participant_id", "item_order", "word_position", "visit_number"]
).reset_index(drop=True)


# --- Wortebene: die eigentliche Analyseeinheit. ---
def summarize_events(events):
    """Fasst die Besuche eines Wortes zu First-Pass-, Rereading- und Gesamtzeit zusammen."""
    if not events:
        return pd.Series({
            "first_pass_reading_time": np.nan,
            "first_pass_move_direction": np.nan,
            "num_reread_events": 0,
            "rereading_time": 0.0,
            "total_reading_time": np.nan,
            "had_regression": False,
        })
    first, rereads = events[0], events[1:]
    first_pass_rt = first[2]
    reread_rt_sum = sum(ev[2] for ev in rereads)
    return pd.Series({
        "first_pass_reading_time": first_pass_rt,
        "first_pass_move_direction": "forward" if first[3] == 1 else "backward",
        "num_reread_events": len(rereads),
        "rereading_time": reread_rt_sum,
        "total_reading_time": first_pass_rt + reread_rt_sum,
        "had_regression": len(rereads) > 0,
    })


summary = df_words_raw["events"].apply(summarize_events)
df_word_level = pd.concat([df_words_raw.drop(columns=["events"]), summary], axis=1)

# 4b) CS-Punkte aus word_cs.xlsx zuordnen.
# word_cs.xlsx enthält die Wörter jedes Satzes in Reihenfolge und ohne
# Satzzeichen. Die 52 Sätze stehen in derselben Reihenfolge und haben dieselbe
# Wortzahl wie die 52 item_order-Werte im Hauptdatensatz.
df_cs = pd.read_excel(CS_PATH)

sentence_order = df_cs["Sentence"].drop_duplicates().tolist()
item_order_sorted = sorted(df_word_level["item_order"].unique())

if len(sentence_order) != len(item_order_sorted):
    raise ValueError(
        f"Die Satzzahl in word_cs.xlsx ({len(sentence_order)}) stimmt nicht mit der "
        f"Itemzahl im Hauptdatensatz ({len(item_order_sorted)}) überein!"
    )

# item_order -> {Wortposition (ab 1): CS-Wert}
cs_lookup = {}
cs_mismatch_report = []
n_participants = df_word_level["participant_id"].nunique()
for item_id, sentence in zip(item_order_sorted, sentence_order):
    cs_values = df_cs.loc[df_cs["Sentence"] == sentence, "CS"].tolist()
    n_main = (df_word_level["item_order"] == item_id).sum() // n_participants
    if len(cs_values) != n_main:
        cs_mismatch_report.append({
            "item_order": item_id, "sentence": sentence,
            "n_words_main": int(n_main), "n_words_cs_file": len(cs_values),
        })
    cs_lookup[item_id] = {pos + 1: cs for pos, cs in enumerate(cs_values)}

df_cs_mismatch_report = pd.DataFrame(cs_mismatch_report)
if not df_cs_mismatch_report.empty:
    print("\nWARNUNG: Für folgende Items stimmt die Wortzahl nicht mit word_cs.xlsx "
          "überein; cs_point wird dort auf NaN gesetzt:")
    print(df_cs_mismatch_report.to_string(index=False))

df_word_level["cs_point"] = df_word_level.apply(
    lambda row: cs_lookup.get(row["item_order"], {}).get(row["word_position"], np.nan), axis=1
)
n_missing_cs = df_word_level["cs_point"].isna().sum()
if n_missing_cs:
    print(f"\nWARNUNG: Für {n_missing_cs} Wortzeilen konnte cs_point nicht zugeordnet werden (NaN).")

# 5) Hauptdatensatz: Versuchsperson x Item x Wort.
df_analysis = df_word_level.merge(
    df_participant_accuracy[["participant_id", "total_questions", "correct_answers",
                             "incorrect_answers", "accuracy", "error_rate", "suitability"]],
    on="participant_id", how="left"
)

df_analysis = df_analysis[[
    "participant_id", "participant_hash",
    "item_order", "latin_square_group", "experimental_condition",
    "word_position", "word_index_0based", "word", "word_raw", "cs_point",
    "first_pass_reading_time", "first_pass_move_direction",
    "num_reread_events", "rereading_time", "total_reading_time", "had_regression",
    "newline_flag", "parse_status",
    "suitability", "error_rate", "accuracy",
]].sort_values(["participant_id", "item_order", "word_position"]).reset_index(drop=True)

# 6) Bericht über fehlende oder nicht lesbare Datensätze.
df_unparsed_report = pd.DataFrame(unparsed_rows + missing_rt_rows)
if df_unparsed_report.empty:
    df_unparsed_report = pd.DataFrame(columns=["line_no", "raw", "reason"])

# 7) Dateien speichern.
df_participant_accuracy.to_csv(OUT_DIR / "participant_accuracy_summary.csv", index=False)
df_questions.to_csv(OUT_DIR / "questions_clean.csv", index=False)
df_reading_events.to_csv(OUT_DIR / "reading_time_events.csv", index=False)
df_analysis.to_csv(OUT_DIR / "main_analysis_dataframe.csv", index=False)
df_unparsed_report.to_csv(OUT_DIR / "unparsed_or_missing_rows_report.csv", index=False)
df_participants.to_csv(OUT_DIR / "participants_id_mapping.csv", index=False)
df_cs_mismatch_report.to_csv(OUT_DIR / "cs_point_mismatch_report.csv", index=False)

# 8) Datei für die LMEM-Analyse (Complete Cases).
# Zeilen ohne aufgezeichnete Lesezeit (parse_status != "ok") werden listenweise ausgeschlossen
excluded_mask = df_analysis["parse_status"] != "ok"
df_excluded_for_lmem = df_analysis.loc[excluded_mask].copy()
df_analysis_lmem_ready = df_analysis.loc[~excluded_mask].reset_index(drop=True)

df_excluded_for_lmem.to_csv(OUT_DIR / "excluded_from_lmem_log.csv", index=False)
df_analysis_lmem_ready.to_csv(OUT_DIR / "main_analysis_dataframe_lmem_ready.csv", index=False)

print(f"\nDaten für die LMEM-Analyse: {df_analysis_lmem_ready.shape[0]} Zeilen "
      f"({excluded_mask.sum()} Zeilen wegen fehlender Lesezeiten ausgeschlossen "
      f"-> excluded_from_lmem_log.csv).")

print("\nGespeicherte Dateien:")
for p in sorted(OUT_DIR.glob("*.csv")):
    print(" -", p.name)

print("\ndf_analysis:", df_analysis.shape)
print("df_reading_events:", df_reading_events.shape)
print("df_questions:", df_questions.shape)
