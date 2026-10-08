"""Execute the main notebook and export its computed research tables."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform

import nltk
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook_path = ROOT / "notebook/information_retrieval.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    scope = {"display": lambda value: None}
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), f"notebook:cell-{i}", "exec"), scope)
    finally:
        os.chdir(previous)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    scope["summary"].to_csv(out / "metrics.csv", float_format="%.8f")
    scope["evaluation"].to_csv(out / "per_query_metrics.csv", index=False, float_format="%.8f")
    scope["vocab_documents"].to_csv(out / "vocabulary_documents.csv", index=False)
    scope["vocab_queries"].to_csv(out / "vocabulary_queries.csv", index=False)
    scope["boolean_summary"].to_csv(out / "boolean_metrics.csv", float_format="%.8f")
    pd.DataFrame(scope["index_timings"]).to_csv(out / "index_timings.csv", index=False)
    metadata = {
        "python": platform.python_version(), "nltk": nltk.__version__, "pandas": pd.__version__,
        "documents": len(scope["documents"]), "queries": len(scope["queries"]),
        "judged_queries": scope["judged_queries"], "sample_queries": scope["test_queries"],
        "bm25": {"k1": 1.2, "b": 0.75, "idf": "ln(1+(N-df+0.5)/(df+0.5))"},
        "fallback": False, "lemmatizer_pos": "noun", "query_tf": "linear",
        "aggregation": "macro", "timing_repeats": 1,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [notebook_path] + [ROOT / "notebook" / f"CISI.{suffix}" for suffix in ("ALL", "QRY", "REL")]},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported results for {len(scope['judged_queries'])} judged queries to {out}")


if __name__ == "__main__":
    main()
