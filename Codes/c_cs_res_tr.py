"""
Auswertung und Visualisierung der TR-DE-Korpusanalyse
=====================================================

Erstellt die Grafiken und Tabellen zu den Code-Switching-Punkten:
Sprach- und POS-Verteilungen, CS-Effizienz je Wortart sowie die
Korrelationsanalyse (Pearson, Spearman, binomiales GLM) zwischen dem
Längen- bzw. Silbenverhältnis (GLR/GSR) und der CS-Effizienz.

Eingabe : TuGeBiC_Twitter_TRDE.csv, CS_Punkte_TR22.csv
Ausgabe : PNG-Dateien im Ordner Grafik/
"""

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from b_cs_corpus_overview_tr import calculate_general_length_stats, calculate_general_syllable_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

FILE_CS_TR = "CS_Punkte_TR22.csv"
FILE_ORIG_TR = "TuGeBiC_Twitter_TRDE.csv"
OUTPUT_DIR = "Grafik"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_POS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'CONJ', 'DET', 'INTJ',
              'NOUN', 'NUM', 'PRON', 'PROPN', 'SCONJ', 'VERB']

COLOR_MAP = {
    "TR": "#FF4D4D", "DE": "#3498DB", "ENG": "#2ECC71", "TR/DE": "#9B59B6",
    "DE/ENG": "#1ABC9C", "MIXED": "#F39C12", "OTHER": "#95A5A6", "AMBIG": "#F1C40F",
    "LANG3": "#D7BDE2", "NOUN": "#3498DB", "VERB": "#FF6B6B", "ADJ": "#FDCB6E",
    "ADV": "#A29BFE", "PRON": "#00CEC9", "ADP": "#55E6C1", "CONJ": "#E17055",
    "CCONJ": "#E17055", "SCONJ": "#FAB1A0", "DET": "#D6A2E8", "PROPN": "#FD79A8",
    "NUM": "#74B9FF", "PART": "#FFEAA7", "INTJ": "#FF7675", "AUX": "#81ECEC",
    "PUNCT": "#B2BEC3", "X": "#636E72",
}

# Einstellungen je Metrik (Länge bzw. Silben): Farben, Spaltennamen, Beschriftungen
METRIC_STYLES = {
    "length": {
        "color_point": "#8E44AD", "color_line": "#6C3483", "color_text": "#7D3C98",
        "ratio_col": "General_Len_Ratio", "kurz_col": "KURZ_LEN",
        "avg_col": "AVG_LENGTH", "stats_func": calculate_general_length_stats,
        "title_metric": "Wortlängenverhältnis (GLR)",
        "xlabel": "Allgemeines Längenverhältnis (GLR) (max/min)\n(1,0 = identische Wortlänge; höhere Werte = größerer Unterschied)",
        "ylabel": "CS-Effizienz: Anteil der Wahl des kürzeren Worts in Prozent",
        "file_suffix": "LEN",
        # Beschriftungen der Balkendiagramme
        "bar_title": "Durchschnittliche Wortlänge",
        "bar_ylabel": "Durchschnittliche Anzahl von Buchstaben",
        "bar_xlabel": "Wortart (POS)",
        "start_text": "gesamten POS-Längen",
    },
    "syllable": {
        "color_point": "#1976D2", "color_line": "#0D47A1", "color_text": "#1565C0",
        "ratio_col": "General_Syllable_Ratio", "kurz_col": "KURZ_SILBEN",
        "avg_col": "AVG_SYLLABLES", "stats_func": calculate_general_syllable_stats,
        "title_metric": "Silbenverhältnis (GSR)",
        "xlabel": "Allgemeines Silbenverhältnis (GSR) (max/min)\n(1,0 = identische Silbenanzahl; höhere Werte = größerer Unterschied)",
        "ylabel": "CS-Effizienz: Anteil der Wahl des silbenärmeren Worts in Prozent",
        "file_suffix": "SILBEN",
        "bar_title": "Durchschnittliche Silbenlänge",
        "bar_ylabel": "Durchschnittliche Anzahl von Silben",
        "bar_xlabel": "Wortart (POS-Tags)",
        "start_text": "gesamten POS-Silbenlänge",
    },
}


# Hilfsfunktionen
def get_safe_color_list(series):
    """Farbliste zu den Indexwerten einer Series; Unbekanntes wird grau."""
    return [COLOR_MAP.get(str(val).strip(), "#95A5A6") for val in series.index]


def check_match(row, col_target):
    """Prüft, ob die gewählte Sprache der ökonomischeren Variante entspricht."""
    lang = str(row['LANG']).strip().upper()
    target = str(row[col_target]).strip().upper()
    options = target.split('/') if '/' in target else [target]
    return "Match" if lang in options else "Mismatch"


def add_bar_labels_pct(ax):
    """Schreibt Prozentwerte über die Balken."""
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')


def with_pos(cs_df, orig_df, only_target=True):
    """Ergänzt die POS-Spalte aus dem Originalkorpus und filtert auf TARGET_POS."""
    if 'POS' not in cs_df.columns:
        pos_map = orig_df[['ID', 'WORD', 'POS']].drop_duplicates(subset=['ID', 'WORD'])
        cs_df = cs_df.merge(pos_map, on=['ID', 'WORD'], how='left')
    df = cs_df.dropna(subset=['POS']).copy()
    return df[df['POS'].isin(TARGET_POS)] if only_target else df


def save_figure(filename):
    """Speichert die aktuelle Figure und schließt sie."""
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


# Balkendiagramme: Durchschnittswerte je POS
def plot_pos_metric(stats_df, dataset_name, metric, ax=None):
    """Durchschnittliche Wortlänge bzw. Silbenzahl je POS und Sprache."""
    style = METRIC_STYLES[metric]
    if ax is None:
        ax = plt.gca()
    data = stats_df[stats_df['POS'].isin(TARGET_POS)]
    sns.barplot(data=data, x='POS', y=style["avg_col"], hue='LANG', palette=COLOR_MAP, ax=ax)
    ax.set_title(f"{style['bar_title']} - {dataset_name}", fontweight='bold')
    ax.set_ylabel(style["bar_ylabel"])
    ax.set_xlabel(style["bar_xlabel"])
    ax.legend(title="Sprache")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)


def save_general_graph(input_csv, dataset_name, output_filename, metric):
    """Berechnet die Durchschnittswerte im Gesamtkorpus und speichert die Grafik."""
    style = METRIC_STYLES[metric]
    print(f"\nStart der Analyse der {style['start_text']} für: {dataset_name}")
    stats = style["stats_func"](pd.read_csv(input_csv))

    plt.figure(figsize=(12, 6))
    plot_pos_metric(stats, dataset_name, metric, ax=plt.gca())
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Die Grafik wurde erfolgreich gespeichert: {save_path}")


def save_cs_lengths_graphs(cs_df, orig_df, dataset_name):
    """Dieselben Balkendiagramme, aber nur für die CS-Punkte."""
    print(f"\nStart der Analyse der CS-Punkte Längen und Silben für: {dataset_name}")
    df_clean = with_pos(cs_df, orig_df)

    for metric, cols, value_col, filename in [
        ("length", ['TR_LEN', 'DE_LEN'], 'AVG_LENGTH',
         f"POS_Lengths_CS_{dataset_name.replace(' ', '_')}.png"),
        ("syllable", ['TR_SILBEN', 'DE_SILBEN'], 'AVG_SYLLABLES',
         f"POS_SilbenLengths_CS_{dataset_name.replace(' ', '_')}.png"),
    ]:
        stats = df_clean.groupby('POS')[cols].mean().reset_index()
        melted = stats.melt(id_vars='POS', var_name='LANG', value_name=value_col)
        melted['LANG'] = melted['LANG'].apply(lambda x: 'TR' if 'TR' in x else 'DE')

        plt.figure(figsize=(12, 6))
        plot_pos_metric(melted, dataset_name, metric, ax=plt.gca())
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Die Grafik wurde erfolgreich gespeichert: {save_path}")


# Korrelationsanalyse: Pearson, Spearman, binomiales GLM
def _prepare_correlation_data(orig_df, cs_df, lang_pair, metric):
    """Datenaufbereitung: GLR bzw. GSR, CS-Effizienz und N_Efficient/N_Total je POS."""
    style = METRIC_STYLES[metric]
    lang1, lang2 = lang_pair

    if 'POS' not in cs_df.columns and not {'ID', 'WORD'}.issubset(cs_df.columns):
        return None
    cs_df = with_pos(cs_df, orig_df, only_target=False)

    stats = style["stats_func"](orig_df)
    stats = stats[stats['LANG'].isin([lang1, lang2]) & stats['POS'].isin(TARGET_POS)]
    pivot_stats = stats.pivot(index='POS', columns='LANG', values=style["avg_col"])
    if lang1 not in pivot_stats.columns or lang2 not in pivot_stats.columns:
        return None

    # Verhältnis immer als max/min, also stets >= 1
    pivot_stats[style["ratio_col"]] = (
        pivot_stats[[lang1, lang2]].max(axis=1) / pivot_stats[[lang1, lang2]].min(axis=1)
    )

    valid_cs = cs_df[cs_df[style["kurz_col"]].isin([lang1, lang2])].copy()
    if valid_cs.empty:
        return None
    valid_cs['Is_Efficient'] = (valid_cs['LANG'] == valid_cs[style["kurz_col"]]).astype(int)

    cs_counts = valid_cs.groupby('POS')['Is_Efficient'].agg(['sum', 'count']).reset_index()
    cs_counts.rename(columns={'sum': 'N_Efficient', 'count': 'N_Total'}, inplace=True)
    cs_counts['N_Inefficient'] = cs_counts['N_Total'] - cs_counts['N_Efficient']
    cs_counts['CS_Efficiency_Rate'] = (cs_counts['N_Efficient'] / cs_counts['N_Total']) * 100

    final_df = pivot_stats[[style["ratio_col"]]].merge(cs_counts, on='POS', how='inner')
    return final_df.sort_values(by=style["ratio_col"]).reset_index(drop=True)


def _label_points(final_df, ratio_col, offset, color_text, y_values=None):
    """Setzt die POS-Beschriftungen ober- bzw. unterhalb der Regressionslinie."""
    x_max = final_df[ratio_col].max()
    x_span = x_max - final_df[ratio_col].min()

    for i, row in enumerate(final_df.itertuples()):
        ref = y_values[i] if y_values is not None else None
        if ref is not None:
            va = 'bottom' if row.CS_Efficiency_Rate >= ref else 'top'
        else:
            va = 'bottom' if i % 2 == 0 else 'top'
        offset_y = offset if va == 'bottom' else -offset

        # Am rechten Rand leicht nach links rücken, damit das Label nicht überlappt
        x_val = getattr(row, ratio_col)
        ha, x_text = 'center', x_val
        if x_val >= x_max - 0.02 * x_span:
            ha, x_text = 'right', x_val - 0.015 * x_span

        plt.text(x_text, row.CS_Efficiency_Rate + offset_y, row.POS,
                 fontsize=10, fontweight='bold', color=color_text, ha=ha, va=va)


def _start_correlation_figure():
    """Einheitliches Format aller drei Korrelationsgrafiken."""
    plt.figure(figsize=(13, 11))
    sns.set_style("whitegrid", {'grid.linestyle': '--'})


def _finish_correlation_figure(style, plot_title, filename):
    """Titel und Achsenbeschriftungen setzen, Grafik speichern."""
    plt.title(plot_title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(style["xlabel"], fontsize=12, fontweight='bold')
    plt.ylabel(style["ylabel"], fontsize=12, fontweight='bold')
    save_figure(filename)


def _print_significance(name, value, p_value):
    print(f"\n{name} = {value:.3f}, p-value = {p_value:.3f}")
    print("Die Korrelation ist statistisch signifikant." if p_value < 0.05
          else "Die Korrelation ist statistisch nicht signifikant. (p >= 0.05)")


def analyze_correlation(orig_df, cs_df, lang_pair, dataset_name, metric):
    """Pearson, Spearman und binomiales GLM für eine Metrik; speichert drei Grafiken."""
    style = METRIC_STYLES[metric]
    ratio_col = style["ratio_col"]

    print(f"\n{'=' * 60}")
    print(f"--- Korrelationsanalyse ({metric.capitalize()} vs. Efficiency): {dataset_name} ---")
    print(f"{'=' * 60}")

    final_df = _prepare_correlation_data(orig_df, cs_df, lang_pair, metric)
    if final_df is None:
        print("Warnung: Keine gültigen Daten für die Korrelationsanalyse gefunden.")
        return

    print(f"\nDaten für Korrelationsplot ({metric.capitalize()}):")
    print(final_df[['POS', ratio_col, 'CS_Efficiency_Rate', 'N_Efficient', 'N_Total']]
          .round(3).to_string(index=False))

    base_filename = f"Korrelation_{dataset_name.replace(' ', '_')}_{style['file_suffix']}"
    scatter_kws = {'s': 150, 'alpha': 0.9, 'edgecolor': 'white'}
    line_kws = {'color': style["color_line"], 'linewidth': 3}

    # 1) Pearson: lineare Regression mit 95%-Konfidenzband
    r, p_pearson = pearsonr(final_df[ratio_col], final_df['CS_Efficiency_Rate'])
    _print_significance("Pearson r", r, p_pearson)

    _start_correlation_figure()
    sns.regplot(data=final_df, x=ratio_col, y='CS_Efficiency_Rate', color=style["color_point"],
                scatter_kws=scatter_kws, line_kws=line_kws, ci=95)
    _label_points(final_df, ratio_col, offset=2.0, color_text=style["color_text"])
    _finish_correlation_figure(
        style, f"Pearson-Korrelation: {style['title_metric']} vs. CS-Effizienz",
        f"{base_filename}_Pearson.png")

    # 2) Spearman: LOWESS-Glättung statt Gerade
    rho, p_spearman = spearmanr(final_df[ratio_col], final_df['CS_Efficiency_Rate'])
    _print_significance("Spearman rho", rho, p_spearman)

    _start_correlation_figure()
    sns.regplot(data=final_df, x=ratio_col, y='CS_Efficiency_Rate', color=style["color_point"],
                scatter_kws=scatter_kws, line_kws=line_kws, lowess=True)
    lowess_fit = sm_lowess(final_df['CS_Efficiency_Rate'], final_df[ratio_col],
                           xvals=final_df[ratio_col])
    _label_points(final_df, ratio_col, offset=2.0, color_text=style["color_text"],
                  y_values=lowess_fit)
    _finish_correlation_figure(
        style, f"Spearman-Korrelation: {style['title_metric']} vs. CS-Effizienz",
        f"{base_filename}_Spearman.png")

    # 3) Binomiales GLM auf den Zählwerten (effizient vs. nicht effizient)
    glm_model = smf.glm(formula=f'N_Efficient + N_Inefficient ~ {ratio_col}',
                        data=final_df, family=sm.families.Binomial()).fit()
    print(f"\n--- Binomial GLM ({metric.capitalize()}) ---")
    print(glm_model.summary())

    x_min, x_max = final_df[ratio_col].min(), final_df[ratio_col].max()
    x_range = pd.DataFrame({ratio_col: [x_min + i * (x_max - x_min) / 99 for i in range(100)]})
    x_range['Predicted'] = glm_model.predict(x_range) * 100
    predicted_at_points = glm_model.predict(final_df[[ratio_col]]) * 100

    _start_correlation_figure()
    plt.scatter(final_df[ratio_col], final_df['CS_Efficiency_Rate'],
                s=150, alpha=0.9, color=style["color_point"], edgecolor='white', zorder=3)
    plt.plot(x_range[ratio_col], x_range['Predicted'], color=style["color_line"],
             linewidth=3, zorder=2)
    _label_points(final_df, ratio_col, offset=2.0, color_text=style["color_text"],
                  y_values=predicted_at_points.values)
    _finish_correlation_figure(
        style, f"Binomial GLM: {style['title_metric']} vs. CS-Effizienz",
        f"{base_filename}_GLM.png")

    print(f"\nGrafiken gespeichert: {base_filename}_Pearson.png / _Spearman.png / _GLM.png")


# Verteilungs- und Effizienzgrafiken
def _plot_pos_percentages(pos_source, title, filename):
    """POS-Verteilung in Prozent als Balkendiagramm."""
    pos_dist_pct = pos_source['POS'].value_counts(normalize=True) * 100 if not pos_source.empty else pd.Series()
    if pos_dist_pct.empty:
        print("Warnung: Keine POS-Daten gefunden.")
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(x=pos_dist_pct.index, y=pos_dist_pct.values, hue=pos_dist_pct.index,
                palette=get_safe_color_list(pos_dist_pct))
    plt.xticks(rotation=90)
    plt.title(title, fontweight='bold')
    plt.ylabel("Prozent (%)")
    plt.xlabel("Wortart (POS-Tags)")
    add_bar_labels_pct(plt.gca())
    save_figure(filename)


def plot_pos_distribution_full_corpus(df_orig, dataset_name):
    """POS-Verteilung im gesamten Korpus (ohne OTHER, PUNCT und X)."""
    if 'POS' not in df_orig.columns or df_orig.empty:
        print("Warnung: POS-Verteilung (gesamtes Korpus) wird übersprungen (fehlende Daten).")
        return
    mask = (df_orig['POS'] != 'PUNCT') & (df_orig['POS'] != 'X')
    if 'LANG' in df_orig.columns:
        mask &= df_orig['LANG'] != 'OTHER'
    _plot_pos_percentages(df_orig[mask], "POS Verteilung im TR-DE-Korpus",
                          f"{dataset_name}_4b_POS_Verteilung_Gesamt.png")


def analyze_complete(cs_file, orig_file, dataset_name):
    """Sprachverteilung, Effizienz je POS und Verteilung der kürzeren Variante."""
    print(f"Starte Analyse für: {dataset_name} ")
    try:
        df_cs = pd.read_csv(cs_file)
        df_orig = pd.read_csv(orig_file) if os.path.exists(orig_file) else pd.DataFrame(columns=['POS'])
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {e}")
        return

    df_cs = with_pos(df_cs, df_orig)

    # 1) Sprachverteilung an CS-Punkten
    lang_dist = df_cs['LANG'].value_counts()
    plt.figure(figsize=(10, 8))
    plt.pie(lang_dist, labels=lang_dist.index, autopct='%1.1f%%',
            colors=get_safe_color_list(lang_dist), startangle=140)
    plt.title("Sprachverteilung an CS-Punkten", fontweight='bold')
    save_figure(f"{dataset_name}_1_Sprachverteilung.png")

    # 2)/3) Effizienz je POS: Wie oft wird die kürzere bzw. silbenärmere Variante gewählt?
    for kurz_col, palette, title, filename in [
        ('KURZ_LEN', "viridis",
         "Effizienz: Wahl des kürzeren Worts (Length) nach POS an CS-Punkten",
         f"{dataset_name}_2_Effizienz_Laenge_POS.png"),
        ('KURZ_SILBEN', "magma",
         "Effizienz: Wahl des silbenärmeren Worts nach POS an CS-Punkten",
         f"{dataset_name}_3_Effizienz_Silben_POS.png"),
    ]:
        is_match = df_cs.apply(lambda x: 1 if check_match(x, kurz_col) == "Match" else 0, axis=1)
        eff_pos = df_cs.assign(_match=is_match).groupby('POS')['_match'].mean().sort_values(ascending=False) * 100

        plt.figure(figsize=(12, 6))
        sns.barplot(x=eff_pos.index, y=eff_pos.values, hue=eff_pos.index,
                    palette=palette, legend=False)
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        plt.title(title, fontweight='bold')
        plt.ylabel("Anteil der Wahl (%)")
        plt.xlabel("Wortart (POS-Tags)")
        plt.ylim(0, 105)
        for i, v in enumerate(eff_pos.values):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
        save_figure(filename)

    # 4) POS-Verteilung im CS-Punkte-Datenrahmen
    if not df_orig.empty and 'CS' in df_orig.columns:
        pos_source = df_orig[(df_orig['CS'] == 1) & (df_orig['LANG'] != 'OTHER') &
                             (df_orig['POS'] != 'PUNCT') & (df_orig['POS'] != 'X')]
    else:
        pos_source = df_orig
    _plot_pos_percentages(pos_source, "POS Verteilung im CS-Punkte-Datenrahmen",
                          f"{dataset_name}_4_POS_Verteilung.png")

    # 5)/6) Welche Sprache stellt die kürzere bzw. silbenärmere Variante?
    for kurz_col, title, filename in [
        ('KURZ_SILBEN', "Verteilung der Sprachen nach weniger Silben an Code-Switching-Punkten",
         f"{dataset_name}_5_Kurz_Silben_Verteilung.png"),
        ('KURZ_LEN', "Verteilung der Sprachen nach kürzerer Wortlänge an Code-Switching-Punkten",
         f"{dataset_name}_6_Kurz_Laenge_Verteilung.png"),
    ]:
        dist_pct = df_cs[kurz_col].value_counts(normalize=True) * 100
        plt.figure(figsize=(10, 6))
        sns.barplot(x=dist_pct.index, y=dist_pct.values, hue=dist_pct.index,
                    palette=get_safe_color_list(dist_pct))
        plt.title(title, fontweight='bold')
        plt.ylabel("Prozent (%)")
        plt.xlabel("Sprache")
        add_bar_labels_pct(plt.gca())
        save_figure(filename)


# Tabellen
def print_simple_efficiency_table(cs_df, orig_df):
    """CS-Effizienzrate je POS, getrennt für Länge und Silbenzahl."""
    print("\n" + "=" * 60)
    print("--- CS EFFICIENCY RATE (Syllables & Length) ---")
    print("=" * 60)

    df_clean = with_pos(cs_df, orig_df)

    tables = []
    for kurz_col, new_name in [('KURZ_LEN', 'CS_Efficiency_Rate (Length)'),
                               ('KURZ_SILBEN', 'CS_Efficiency_Rate (Syllable)')]:
        valid = df_clean[df_clean[kurz_col].isin(['TR', 'DE'])].copy()
        valid['Is_Efficient'] = (valid['LANG'].astype(str).str.strip()
                                 == valid[kurz_col].astype(str).str.strip()).astype(int)
        eff = valid.groupby('POS')['Is_Efficient'].mean().reset_index()
        tables.append(eff.rename(columns={'Is_Efficient': new_name}))

    final_eff = pd.merge(tables[0], tables[1], on='POS', how='outer')
    print(final_eff.round(3).to_string(index=False))
    print("\n")


def print_pos_averages(orig_df, lang_pair):
    """Durchschnittliche Länge und Silbenzahl je POS im Gesamtkorpus."""
    lang1, lang2 = lang_pair
    print(f"\n{'=' * 60}")
    print(f"--- Durchschnittliche Längen und Silben pro POS ({lang1} vs {lang2}) ---")
    print(f"{'=' * 60}")

    for header, stats_func, value_col, unit, ratio_name in [
        ("WORTLÄNGEN", calculate_general_length_stats, 'AVG_LENGTH', 'LEN', 'General_Length_Ratio'),
        ("SILBENLÄNGEN", calculate_general_syllable_stats, 'AVG_SYLLABLES', 'SILBEN', 'General_Syllable_Ratio'),
    ]:
        stats = stats_func(orig_df)
        stats = stats[stats['LANG'].isin([lang1, lang2]) & stats['POS'].isin(TARGET_POS)]
        pivot = stats.pivot(index='POS', columns='LANG', values=value_col).reset_index()
        col1, col2 = f'{lang1} {unit}', f'{lang2} {unit}'
        pivot.rename(columns={lang1: col1, lang2: col2}, inplace=True)
        pivot[ratio_name] = pivot[col1] / pivot[col2]

        print(f"\n------ DURCHSCHNITTLICHE {header} ------")
        print(pivot[['POS', col1, col2, ratio_name]].round(3).to_string(index=False))
    print("\n")


def print_cs_pos_averages(cs_df, orig_df):
    """Dieselben Durchschnittswerte, aber nur an den CS-Punkten."""
    print(f"\n{'=' * 60}")
    print("--- Durchschnittliche Längen und Silben pro POS (NUR CS-PUNKTE) ---")
    print(f"{'=' * 60}")

    df_clean = with_pos(cs_df, orig_df)

    for header, cols, names, ratio_name in [
        ("WORTLÄNGEN", ['TR_LEN', 'DE_LEN'], ['TR LEN', 'DE LEN'], 'CS_Length_Ratio'),
        ("SILBENLÄNGEN", ['TR_SILBEN', 'DE_SILBEN'], ['TR SILBEN', 'DE SILBEN'], 'CS_Syllable_Ratio'),
    ]:
        stats = df_clean.groupby('POS')[cols].mean().reset_index()
        stats.rename(columns=dict(zip(cols, names)), inplace=True)
        stats[ratio_name] = stats[names[0]] / stats[names[1]]

        print(f"\n------ DURCHSCHNITTLICHE {header} (CS-PUNKTE) ------")
        print(stats[['POS'] + names + [ratio_name]].round(3).to_string(index=False))
    print("\n")


# Ablauf
if __name__ == "__main__":
    orig_dataframe = pd.read_csv(FILE_ORIG_TR)
    cs_dataframe = pd.read_csv(FILE_CS_TR)

    # 1) Grafiken zum Gesamtkorpus
    plot_pos_distribution_full_corpus(orig_dataframe, "TR_DE_Corpus")
    analyze_complete(FILE_CS_TR, FILE_ORIG_TR, "TR_DE_Corpus")
    save_general_graph(FILE_ORIG_TR, "TR-DE Corpus", "POS_Lengths_TR.png", metric="length")
    save_general_graph(FILE_ORIG_TR, "TR-DE Corpus", "POS_SilbenLengths_TR.png", metric="syllable")

    # 2) Grafiken zu den CS-Punkten
    save_cs_lengths_graphs(cs_dataframe, orig_dataframe, "CS-Punkte-Datenrahmen")

    # 3) Tabellen
    print_pos_averages(orig_dataframe, ('TR', 'DE'))
    print_cs_pos_averages(cs_dataframe, orig_dataframe)
    print_simple_efficiency_table(cs_dataframe, orig_dataframe)

    # 4) Korrelationsanalysen (Pearson, Spearman, GLM – für Länge und Silben)
    analyze_correlation(orig_dataframe, cs_dataframe, ('TR', 'DE'), "TR-DE Corpus", metric="length")
    analyze_correlation(orig_dataframe, cs_dataframe, ('TR', 'DE'), "TR-DE Corpus", metric="syllable")

    print(f"\nFertig! Alle Ergebnisse befinden sich im Ordner '{OUTPUT_DIR}'.")
