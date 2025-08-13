# mlframework/__main__.py
import argparse
from dataset_analysis import profile_dataset

def parse_args():
    p = argparse.ArgumentParser(
        prog="mlframework",
        description="Entry point del framework ML: esegui o salta il profiling del dataset."
    )
    p.add_argument("--dataset", required=True, help="Percorso al CSV (es. mio_dataset.csv)")
    p.add_argument("--output-dir", default="dataset_analisys", help="Cartella base per l'output")
    p.add_argument("--label-column", default=None, help="Nome colonna etichette (opzionale)")
    p.add_argument("--analyze", action="store_true", help="Esegui l'analisi del dataset")

    p.add_argument("--freq-top-k", type=int, default=20, help="Top-k categorie per frequenze")
    p.add_argument("--numeric-bins", type=int, default=30, help="Bin per istogrammi numerici")
    p.add_argument("--max-num-plots", type=int, default=None, help="Limite grafici numerici")
    p.add_argument("--max-cat-plots", type=int, default=None, help="Limite grafici categoriali")
    p.add_argument("--no-html", dest="save_html_overview", action="store_false",
                   help="Non salvare l'overview HTML")
    p.set_defaults(save_html_overview=True)
    return p.parse_args()

def main():
    args = parse_args()

    if not args.analyze:
        print("Analisi dataset: SALTATA (usa --analyze per eseguirla).")
        return

    res = profile_dataset(
        csv_path=args.dataset,
        output_base_dir=args.output_dir,
        label_column=args.label_column,
        freq_top_k=args.freq_top_k,
        numeric_bins=args.numeric_bins,
        max_num_plots=args.max_num_plots,
        max_cat_plots=args.max_cat_plots,
        save_html_overview=args.save_html_overview,
    )
    print(f"Analisi salvata in: {res['base_dir']}")
    if res["overview_html"]:
        print(f"Overview HTML: {res['overview_html']}")
    print(f"Schema YAML: {res['schema_path']}")

if __name__ == "__main__":
    main()
