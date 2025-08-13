# dataset_analysis.py
# Toolkit per profiling dataset: analisi tabellare, grafici e schema YAML.
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml  # richiede PyYAML


# ----------------------------
# Caricamento
# ----------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Carica un dataset CSV in un DataFrame pandas."""
    return pd.read_csv(path)


# ----------------------------
# Analisi tabellare
# ----------------------------
def analyze_dataset(
    df: pd.DataFrame,
    show_report: bool = True,
    freq_top_k: int = 20,
    label_column: Optional[str] = None,
) -> Dict[str, object]:
    """
    Analizza il dataset e (opzionalmente) stampa un report leggibile.

    Ritorna un dict con:
      - column_types_df: DataFrame (colonna -> dtype)
      - numeric_stats_df: DataFrame (describe() numerico)
      - categorical_frequencies: dict[col_name] -> DataFrame [value, count]
      - label_distribution_df: DataFrame con counts e percentages (solo se label_column valido)
    """
    # Tipi di colonna
    column_types_df = (
        pd.DataFrame(df.dtypes, columns=["dtype"])
        .reset_index()
        .rename(columns={"index": "column"})
    )

    # Statistiche numeriche
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        numeric_stats_df = df.describe().T.reset_index().rename(columns={"index": "column"})
    else:
        numeric_stats_df = pd.DataFrame(columns=["column"])

    # Frequenze per categoriali
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_frequencies: Dict[str, pd.DataFrame] = {}
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)
        top = vc.head(freq_top_k)
        freq_df = (
            top.rename("count")
            .to_frame()
            .reset_index()
            .rename(columns={"index": col})
        )
        categorical_frequencies[col] = freq_df

    # Bilanciamento etichette (solo se label_column valido)
    label_distribution_df = None
    if label_column and label_column in df.columns:
        counts = df[label_column].value_counts(dropna=False)
        perc = (counts / len(df) * 100.0)
        label_distribution_df = pd.DataFrame({
            "label": counts.index.astype(object),
            "count": counts.values,
            "percentage": np.round(perc.values, 3),
        })

    report = {
        "column_types_df": column_types_df,
        "numeric_stats_df": numeric_stats_df,
        "categorical_frequencies": categorical_frequencies,
        "label_distribution_df": label_distribution_df,
    }

    if show_report:
        print("\n=== Tipi di colonna ===")
        print(column_types_df.to_string(index=False))

        if not numeric_stats_df.empty:
            print("\n=== Statistiche numeriche ===")
            print(numeric_stats_df.to_string(index=False))

        if categorical_frequencies:
            print("\n=== Frequenze valori (categoriali) ===")
            for col, freq_df in categorical_frequencies.items():
                print(f"\nColonna: {col} (top {len(freq_df)})")
                print(freq_df.to_string(index=False))

        if label_distribution_df is not None:
            print("\n=== Bilanciamento etichette ===")
            print(label_distribution_df.to_string(index=False))

    return report


# ----------------------------
# Visualizzazioni (matplotlib: un grafico per figura, nessun colore specificato)
# ----------------------------
def plot_label_balance(
    df: pd.DataFrame,
    label_column: Optional[str] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> Optional[str]:
    """Grafico delle frequenze della colonna etichette (se specificata e presente)."""
    if not label_column or label_column not in df.columns:
        return None

    counts = df[label_column].value_counts(dropna=False)
    labels = [str(x) for x in counts.index.tolist()]
    values = counts.values

    plt.figure()
    plt.bar(labels, values)
    plt.title(title or f"Bilanciamento etichette ({label_column})")
    plt.xlabel(label_column)
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_numeric_distributions(
    df: pd.DataFrame,
    bins: int = 30,
    max_plots: Optional[int] = None,
    save_dir: Optional[str] = None,
    sample_cols: Optional[List[str]] = None
) -> List[Tuple[str, Optional[str]]]:
    """Istogramma per ogni colonna numerica."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if sample_cols is not None:
        num_cols = [c for c in sample_cols if c in num_cols]
    if max_plots is not None:
        num_cols = num_cols[:max_plots]

    saved_paths = []
    for col in num_cols:
        data = df[col].dropna().values
        if data.size == 0:
            continue

        plt.figure()
        plt.hist(data, bins=bins)
        plt.title(f"Distribuzione: {col}")
        plt.xlabel(col)
        plt.ylabel("frequency")
        plt.tight_layout()

        path = None
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            path = str(Path(save_dir) / f"hist_{col}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append((col, path))

    return saved_paths


def plot_categorical_frequencies(
    df: pd.DataFrame,
    top_k: int = 20,
    max_plots: Optional[int] = None,
    save_dir: Optional[str] = None,
    sample_cols: Optional[List[str]] = None
) -> List[Tuple[str, Optional[str]]]:
    """Bar chart delle TOP-k categorie per colonna categoriale."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if sample_cols is not None:
        cat_cols = [c for c in sample_cols if c in cat_cols]
    if max_plots is not None:
        cat_cols = cat_cols[:max_plots]

    saved_paths = []
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).head(top_k)
        if vc.empty:
            continue
        labels = [str(x) for x in vc.index.tolist()]
        values = vc.values

        plt.figure()
        plt.bar(labels, values)
        plt.title(f"Frequenze top-{top_k}: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        path = None
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            path = str(Path(save_dir) / f"freq_{col}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append((col, path))

    return saved_paths


# ----------------------------
# Schema YAML
# ----------------------------
def save_yaml_schema(
    df: pd.DataFrame,
    output_path: str,
    target_column: Optional[str] = None
) -> str:
    """
    Crea e salva un file YAML con meta-info sul dataset:
    - nome dataset (cartella)
    - target_column se presente (o None)
    - n_rows, n_columns
    - per ogni colonna: tipo, missing, (numeriche: mean/std/min/max), (categoriali: top categories, n unique)
    """
    schema = {
        "dataset_name": Path(output_path).parent.name,
        "target_column": target_column if target_column and target_column in df.columns else None,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": {},
        "exclude_columns": [],
        "notes": "",
    }

    for col in df.columns:
        col_info: Dict[str, object] = {}
        missing = int(df[col].isna().sum())
        col_info["missing_values"] = missing

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["type"] = "numeric"
            if df[col].notna().any():
                col_info["mean"] = float(df[col].mean())
                col_info["std"] = float(df[col].std())
                col_info["min"] = float(df[col].min())
                col_info["max"] = float(df[col].max())
            else:
                col_info["mean"] = col_info["std"] = col_info["min"] = col_info["max"] = None
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
            col_info["type"] = "categorical"
            top_cats = df[col].value_counts(dropna=False).head(10).index.tolist()
            col_info["categories_top"] = [str(c) for c in top_cats]
            col_info["unique_values"] = int(df[col].nunique(dropna=False))
        else:
            # fallback per tipi non standard (datetime64, bool, ecc.)
            col_info["type"] = str(df[col].dtype)

        schema["columns"][col] = col_info

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(schema, f, allow_unicode=True, sort_keys=False)

    return output_path


# ----------------------------
# Pipeline end-to-end
# ----------------------------
def profile_dataset(
    csv_path: str,
    output_base_dir: str = "dataset_analisys",
    label_column: Optional[str] = None,   # <— default None
    freq_top_k: int = 20,
    numeric_bins: int = 30,
    max_num_plots: Optional[int] = None,
    max_cat_plots: Optional[int] = None,
    sample_numeric_cols: Optional[List[str]] = None,
    sample_categorical_cols: Optional[List[str]] = None,
    save_html_overview: bool = True
) -> Dict[str, object]:
    """
    Pipeline completa:
      1) Carica csv
      2) Analizza tipi, statistiche, frequenze, (bilanciamento label solo se label_column valido)
      3) Salva tabelle CSV
      4) Salva grafici PNG
      5) Salva schema YAML (schema.yaml)
      6) (opzionale) overview.html con link rapidi

    Output:
      dataset_analisys/<nome_csv_senza_ext>/
        ├── tables/
        │     ├── column_types.csv
        │     ├── numeric_stats.csv
        │     ├── categorical_<col>.csv
        │     └── label_distribution.csv (solo se label_column valido)
        ├── figures/
        │     ├── label_balance.png (solo se label_column valido)
        │     ├── hist_<numcol>.png
        │     └── freq_<catcol>.png
        └── schema.yaml
    """
    csv_path = str(csv_path)
    csv_name = Path(csv_path).stem
    base_dir = Path(output_base_dir) / csv_name
    tables_dir = base_dir / "tables"
    figs_dir = base_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Carica
    df = load_dataset(csv_path)

    # 2) Analisi (passiamo label_column per includere o meno il bilanciamento)
    report = analyze_dataset(df, show_report=False, freq_top_k=freq_top_k, label_column=label_column)

    # 3) Salva tabelle
    report["column_types_df"].to_csv(tables_dir / "column_types.csv", index=False)
    report["numeric_stats_df"].to_csv(tables_dir / "numeric_stats.csv", index=False)
    for col, freq_df in report["categorical_frequencies"].items():
        safe_col = str(col).replace(os.sep, "_")
        freq_df.to_csv(tables_dir / f"categorical_{safe_col}.csv", index=False)
    if report["label_distribution_df"] is not None:
        report["label_distribution_df"].to_csv(tables_dir / "label_distribution.csv", index=False)

    # 4) Salva grafici
    label_png = plot_label_balance(
        df,
        label_column=label_column,
        save_path=str(figs_dir / "label_balance.png"),
    )
    num_saved = plot_numeric_distributions(
        df,
        bins=numeric_bins,
        max_plots=max_num_plots,
        save_dir=str(figs_dir),
        sample_cols=sample_numeric_cols
    )
    cat_saved = plot_categorical_frequencies(
        df,
        top_k=freq_top_k,
        max_plots=max_cat_plots,
        save_dir=str(figs_dir),
        sample_cols=sample_categorical_cols
    )

    # 5) Schema YAML
    schema_path = base_dir / "schema.yaml"
    save_yaml_schema(df, str(schema_path), target_column=label_column)

    # 6) Overview HTML (opzionale)
    overview_html_path = None
    if save_html_overview:
        overview_html_path = base_dir / "overview.html"
        lines = [
            "<html><head><meta charset='utf-8'><title>Dataset Analysis Overview</title></head><body>",
            f"<h1>Dataset: {csv_name}</h1>",
            "<h2>Tabelle</h2><ul>",
            f"<li><a href='tables/column_types.csv'>column_types.csv</a></li>",
            f"<li><a href='tables/numeric_stats.csv'>numeric_stats.csv</a></li>",
        ]
        for col in report["categorical_frequencies"].keys():
            safe_col = str(col).replace(os.sep, "_")
            lines.append(f"<li><a href='tables/categorical_{safe_col}.csv'>categorical_{safe_col}.csv</a></li>")
        if report["label_distribution_df"] is not None:
            lines.append("<li><a href='tables/label_distribution.csv'>label_distribution.csv</a></li>")
        lines.append("</ul>")

        lines.append("<h2>Figure</h2><ul>")
        if label_png:
            lines.append("<li><a href='figures/label_balance.png'>label_balance.png</a></li>")
        for _, path in num_saved:
            if path:
                lines.append(f"<li><a href='figures/{Path(path).name}'>{Path(path).name}</a></li>")
        for _, path in cat_saved:
            if path:
                lines.append(f"<li><a href='figures/{Path(path).name}'>{Path(path).name}</a></li>")
        lines.append("</ul>")

        lines.append("<h2>Schema</h2><ul>")
        lines.append("<li><a href='schema.yaml'>schema.yaml</a></li>")
        lines.append("</ul>")

        lines.append("</body></html>")
        Path(overview_html_path).write_text("\n".join(lines), encoding="utf-8")

    return {
        "base_dir": str(base_dir),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figs_dir),
        "overview_html": str(overview_html_path) if overview_html_path else None,
        "schema_path": str(schema_path),
        "label_figure": label_png,
        "numeric_figures": [p for _, p in num_saved if p],
        "categorical_figures": [p for _, p in cat_saved if p],
    }
