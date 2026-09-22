# Quantitative Merkmale und kognitiver Aufwand von Code-Switching (Türkisch–Deutsch): LLM-Surprisal vs. Lesezeiten

Dieses Repository enthält den Code und die Daten zur Untersuchung von Türkisch–Deutsch Code-Switching (CS). Im Mittelpunkt steht der Zusammenhang zwischen den LLM-Surprisal-Werten (XGLM) und menschlichen Lesezeiten in mono- und multilingualen Sätzen. Die Analysen basieren auf Korpusdaten (TuGeBiC) und Lesezeitdaten aus Self-Paced-Reading-Experimenten (BSPR / PCIbex), die mit Linear Mixed-Effects Models (LMEM) ausgewertet werden.

---

## Überblick & Forschungsgegenstand

Die Untersuchung gliedert sich in drei zentrale methodische Schritte:

1. **Korpuslinguistische Analyse & Ökonomie (TuGeBiC-Korpus):**
   - Identifikation von Code-Switching-Punkten im Türkisch–Deutschen Korpus.
   - Analyse von Wortarten (POS) sowie Wort- und Silbenlängen (Ökonomiehypothese: Wahl der kürzeren Sprachvariante).
2. **Informationstheoretische Modellierung (Surprisal):**
   - Berechnung des wortweisen Surprisals mittels multilingualer Sprachmodelle (`facebook/xglm-564M` und `facebook/xglm-1.7B`).
3. **Psycholinguistischer Ansatz (LMEM & Lesezeiten):**
   - Untersuchung des Zusammenhangs zwischen LLM-Surprisal-Werten, Code-Switching und First-Pass-Lesezeiten (Self-Paced Reading / PCIbex) mit Linear Mixed-Effects Models.
   - Vergleich der Modelle XGLM-564M und XGLM-1.7B anhand ihrer Modellgüte.

---

## Pipeline & Skripte

Die Skripte sind modular aufgebaut und werden in folgender Reihenfolge ausgeführt:

| Skript | Beschreibung | Eingabe → Ausgabe |
|---|---|---|
| `a_cs_features_tr.py` | Extrahiert Wort-/Silbenmerkmale und übersetzt CS-Wörter. ⚠️ Nicht erneut ausführen, wenn `CS_Punkte_TR22.csv` bereits vorliegt. Online-Übersetzung mit `deep-translator`, kann lange dauern. | `TuGeBiC_Twitter_TRDE.csv` → `CS_Punkte_TR22.csv` |
| `b_cs_corpus_overview_tr.py` | Korpusstatistiken (Tokens, POS, Sprachverteilungen). | `TuGeBiC_Twitter_TRDE.csv` → Statistiken |
| `c_cs_res_tr.py` | Ökonomie- und Korrelationsanalyse. | `TuGeBiC_Twitter_TRDE.csv` + `CS_Punkte_TR22.csv` → Ergebnisse/Grafiken |
| `d_clean_bspr_data.py` | Bereinigung und Vorverarbeitung der PCIbex-Lesezeitdaten. | `010826_results_prod.csv` + `word_cs.xlsx` → LMEM-Daten |
| `f_xglm_564m_surprisal_values.py` | Berechnet Wort-Surprisal mit `facebook/xglm-564M`. | `word_cs.xlsx` → `small_surprisal_values.csv` |
| `e_XGLM_surprisal_values.py` | Berechnet Wort-Surprisal mit `facebook/xglm-1.7B` (GPU empfohlen). | `word_cs.xlsx` → `surprisal_values.csv` |
| `g_Lmem_Surprisal.py` | Hauptanalyse mit Linear Mixed-Effects Models (LMEM). | LMEM-Daten + 564M-Surprisal → Analyseergebnisse |
| `h_lmem_modellvergleich.py` | Vergleich der beiden XGLM-Modelle. | LMEM-Daten + beide Surprisal-Dateien → `modellvergleich_summary.csv` |
| `i_surprisal_rt_sample.py` | Visualisiert Surprisal und Lesezeiten eines Beispielsatzes. | LMEM-Daten + Surprisal → Tabelle/Grafik |
| `j_surprisal_rt_all_sample.py` | Erstellt Übersichten aller Stimulussätze. | Analyse-Daten + `word_cs.xlsx` → `.pdf` / `.png` |
| `BA_ALL_ANALYSE.ipynb` | Interaktives Notebook der gesamten Pipeline (Google Colab). | `TuGeBiC_Twitter_TRDE.csv` + Alle benötigten Dateien → Gesamte Pipeline |

---

## Ausführung & Umgebung

Die gesamte Analyse- und Modellierungspipeline wurde in **Google Colab** über das Notebook `BA_ALL_ANALYSE.ipynb` ausgeführt.

### Ausführung in Google Colab
Das Jupyter-Notebook `BA_ALL_ANALYSE.ipynb` bündelt alle Schritte der Untersuchung (von der Korpus- und Ökonomieanalyse über Surprisal-Berechnungen mit XGLM bis hin zu den LMEM-Modellen) und kann direkt in Google Colab ausgeführt werden. Für die vollständige Ausführung müssen die Dateien `TuGeBiC_Twitter_TRDE.csv`, `CS_Punkte_TR22.csv`, `010826_results_prod.csv` und `word_cs.xlsx` in das Notebook bzw. die Colab-Umgebung eingebunden werden, da diese als notwendige Eingabedateien für die Analysen dienen.

### Lokale Ausführung (Optional)
Alternativ können die einzelnen modularen Python-Skripte lokal ausgeführt werden:
```bash
pip install -r requirements.txt
```

---

## Schnellausführung (Workflow)

```bash
# 1. Korpus- und Ökonomieanalysen
# (a_cs_features_tr.py überspringen, wenn CS_Punkte_TR22.csv bereits existiert)
python b_cs_corpus_overview_tr.py
python c_cs_res_tr.py

# 2. Lesezeit-Rohdaten bereinigen
python d_clean_bspr_data.py

# 3. Surprisal-Werte berechnen
python e_XGLM_surprisal_values.py
python f_xglm_564m_surprisal_values.py

# 4. LMEM-Modelle berechnen
python g_Lmem_Surprisal.py
python h_lmem_modellvergleich.py

# 5. Visualisierungen erstellen
python i_surprisal_rt_sample.py
python j_surprisal_rt_all_sample.py
```

---

## Ordnerstruktur & wichtigste Ausgaben

- `Grafik/`: Enthält generierte Diagramme (z. B. POS-Verteilungen `TR_DE_POS.png`, Regressionskurven `GLM_*.png`).
- `Bspr_result/`: Bereinigte Lesezeitdaten, Surprisal-Tabellen (`surprisal_values.csv`), LMEM-Ergebnisse und Satzraster (`multilingual_sentences_grid.pdf`).
- `TuGeBiC_Twitter_TRDE.csv` & `010826_results_prod.csv`: Primäre Korpus- und Rohdatendateien.

---

## Referenzen

- **TuGeBiC Korpus:** Çetinoğlu, Ö. (2016). *A Turkish-German Code-Switching Corpus*.
- **XGLM:** Lin, X. V. et al. (2021). *Few-shot Learning with Multilingual Language Models*.
- **Surprisal & Lesezeiten:** Oh, B.-D., & Schuler, W. (2023). *Why does surprisal from larger language models provide poorer predictions of reading times?*
- **pymer4:** Jolly, E. (2018). *Pymer4: Connecting R and Python for Linear Mixed Modeling*.
