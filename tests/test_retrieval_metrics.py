import unittest

from evaluation import metrics


class QuestionMetricsTest(unittest.TestCase):
    def test_partial_recall_and_reciprocal_rank(self):
        result = metrics.question_metrics(["x", "a", "y", "b"], ["a", "b"], [1, 2, 4])
        self.assertEqual(result["recall@1"], 0.0)
        self.assertEqual(result["recall@2"], 0.5)
        self.assertEqual(result["recall@4"], 1.0)
        self.assertEqual(result["hit@1"], 0.0)
        self.assertEqual(result["hit@2"], 1.0)
        self.assertEqual(result["full@2"], 0.0)
        self.assertEqual(result["full@4"], 1.0)
        self.assertEqual(result["mrr"], 0.5)

    def test_duplicate_sources_count_once_and_do_not_consume_rank(self):
        result = metrics.question_metrics(["x", "x", "x", "a"], ["a"], [2])
        # 去重后排名为 x, a，所以 a 在第 2 位。
        self.assertEqual(result["recall@2"], 1.0)
        self.assertEqual(result["mrr"], 0.5)

    def test_no_gold_found(self):
        result = metrics.question_metrics(["x"], ["a"], [5])
        self.assertEqual(result["recall@5"], 0.0)
        self.assertEqual(result["mrr"], 0.0)

    def test_empty_gold_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.question_metrics(["a"], [], [1])


class SummaryTest(unittest.TestCase):
    def test_summarize_averages_each_metric(self):
        rows = [{"mrr": 1.0, "hit@1": 1.0}, {"mrr": 0.0, "hit@1": 0.0}]
        self.assertEqual(metrics.summarize(rows), {"mrr": 0.5, "hit@1": 0.5})
        self.assertEqual(metrics.summarize([]), {})

    def test_percentile_uses_nearest_rank(self):
        values = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(metrics.percentile(values, 0.5), 20.0)
        self.assertEqual(metrics.percentile(values, 0.95), 40.0)
        self.assertEqual(metrics.percentile([], 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
