import contextlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def execute_notebook(path):
    scope = {"display": lambda value: None}
    notebook = json.loads(path.read_text(encoding="utf-8"))
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), f"{path.name}:cell-{i}", "exec"), scope)
    finally:
        os.chdir(previous)
    return scope


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = execute_notebook(ROOT / "notebook/information_retrieval.ipynb")

    def test_dataset_integrity(self):
        scope = self.scope
        self.assertEqual(len(scope["documents"]), 1460)
        self.assertEqual(len(scope["queries"]), 112)
        self.assertEqual(set(scope["documents"]), set(range(1, 1461)))
        self.assertEqual(set(scope["queries"]), set(range(1, 113)))
        for qid, docs in scope["relevance"].items():
            self.assertIn(qid, scope["queries"])
            self.assertTrue(docs <= scope["documents"].keys())
        self.assertGreater(len(scope["judged_queries"]), 5)
        self.assertLess(len(scope["judged_queries"]), 112)

    def test_real_lemmatization(self):
        self.assertEqual(self.scope["lemmatizer"].lemmatize("libraries"), "library")
        self.assertEqual(self.scope["stemmer"].stem("librarianship"), "librarianship")

    def test_field_search_uses_complete_query(self):
        scope = self.scope
        query = {"stemmed_words": scope["documents"][1]["title_stemmed"], "title_stemmed": []}
        results = scope["bm25_search"](query, scope["query_field_for"]("title_stemmed"), scope["rich_indexes"]["title_stemmed"])
        self.assertIn(1, dict(results))

    def test_scores_against_independent_formulas(self):
        scope = self.scope
        docs = {1: {"tokens": ["a", "a", "b"]}, 2: {"tokens": ["b"]}, 3: {"tokens": []}}
        idx = scope["build_rich_inverted_index"](docs, "tokens")
        query = {"tokens": ["a", "a", "b"]}
        tfidf = dict(scope["tfidf_search"](query, "tokens", idx))
        self.assertAlmostEqual(tfidf[1], (4 * math.log(3) + math.log(1.5)) / 3)
        self.assertAlmostEqual(tfidf[2], math.log(1.5))
        bm25 = dict(scope["bm25_search"](query, "tokens", idx))
        expected = 0.0
        for df, tf, qtf in ((1, 2, 2), (2, 1, 1)):
            expected += qtf * math.log(1 + (3 - df + .5) / (df + .5)) * tf * 2.2 / (tf + 1.2 * (.25 + .75 * 3 / (4 / 3)))
        self.assertAlmostEqual(bm25[1], expected)
        self.assertNotIn(3, bm25)

    def test_empty_indexes_and_queries(self):
        scope = self.scope
        for docs in ({}, {1: {"tokens": []}}):
            idx = scope["build_rich_inverted_index"](docs, "tokens")
            for search in (scope["tfidf_search"], scope["bm25_search"]):
                self.assertEqual(search({"tokens": ["unknown"]}, "tokens", idx), [])
        self.assertEqual(scope["boolean_search"]({"tokens": []}, "tokens", "tokens"), set())

    def test_fallback_on_empty_query(self):
        scope = self.scope
        query = {"title_stemmed": [], "tokens": ["known"], "stemmed_words": ["wrong"]}
        empty = scope["build_rich_inverted_index"]({1: {"title_stemmed": []}}, "title_stemmed")
        full = scope["build_rich_inverted_index"]({1: {"tokens": ["known"]}, 2: {"tokens": []}}, "tokens")
        for search in (scope["tfidf_search"], scope["bm25_search"]):
            expected = search(query, "tokens", full)
            self.assertEqual(search(query, "title_stemmed", empty, fallback_idx=full), [(i, v * .5) for i, v in expected])

    def test_boolean_fallback_and_unknown_operator(self):
        scope = self.scope
        query = {"title_tokens": [], "tokens": ["library"]}
        self.assertEqual(scope["boolean_search"](query, "title_tokens", "title_tokens", fallback_field="tokens"), set(scope["simple_indexes"]["tokens"]["library"]))
        with self.assertRaises(ValueError):
            scope["boolean_search"](query, "tokens", "tokens", operator="XOR")

    def test_metrics(self):
        scope = self.scope
        result = [(1, 3.0), (2, 2.0), (3, 1.0)]
        self.assertAlmostEqual(scope["precision_at_k"](result, {1, 3, 4}, 5), 2 / 5)
        self.assertAlmostEqual(scope["recall_at_k"](result, {1, 3, 4}, 5), 2 / 3)
        self.assertAlmostEqual(scope["average_precision"](result, {1, 3, 4}), (1 + 2 / 3) / 3)
        self.assertEqual(scope["precision_at_k"]([], {1}, 0), 0)
        self.assertEqual(scope["recall_at_k"]([], set(), 5), 0)
        for metric in (scope["precision_at_k"], scope["recall_at_k"]):
            with self.assertRaises(ValueError):
                metric(result, {1}, -1)

    def test_bm25_parameters(self):
        for k1, b in ((0, .75), (-1, .75), (1.2, -1), (1.2, 2), (float("nan"), .75)):
            with self.assertRaises(ValueError):
                self.scope["bm25_search"]({"tokens": []}, "tokens", self.scope["rich_indexes"]["tokens"], k1=k1, b=b)

    def test_parser_unknown_sections_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.ALL"
            path.write_text(".I 1\n.W\nvalid text\n.Z\nignored metadata\n.X\n1 5 1\n", encoding="utf-8")
            record = self.scope["parse"](path)[1]
            self.assertEqual(record["text"], "valid text")
            self.assertEqual(record["x_ref"], [(1, 5, 1)])
            path.write_text(".I 1\n.W\nfirst\n.I 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.scope["parse"](path)

    def test_rankings_and_evaluation(self):
        scope = self.scope
        for search in (scope["tfidf_search"], scope["bm25_search"]):
            for qid in scope["judged_queries"]:
                results = search(scope["queries"][qid], "stemmed_words", scope["rich_indexes"]["stemmed_words"])
                self.assertEqual(results, sorted(results, key=lambda pair: (-pair[1], pair[0])))
                self.assertEqual(len(results), len(dict(results)))
                self.assertTrue(all(math.isfinite(v) and v > 0 for _, v in results))
        expected = 2 * len(scope["fields"]) * len(scope["judged_queries"])
        self.assertEqual(len(scope["evaluation"]), expected)
        self.assertTrue(scope["evaluation"].drop(columns=["model", "field", "query"]).map(lambda x: 0 <= x <= 1).all().all())

    def test_notebook_copy_consistency(self):
        self.assertEqual((ROOT / "notebook/information_retrieval.ipynb").read_bytes(), (ROOT / "notebook/information_retrieval copy.ipynb").read_bytes())


if __name__ == "__main__":
    unittest.main()
