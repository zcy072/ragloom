import unittest

from evaluation import metrics
from evaluation.run_eval import diagnose_rerank_loss


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


class RerankDiagnosisTest(unittest.TestCase):
    def test_separates_crowded_results_from_gold_pushed_out(self):
        many = [f"d{i}" for i in range(12)]
        traces = {
            # 候选前 10 含全部标准文档，重排只返回 3 个不同文档：名额被同一文档占满。
            "crowded": {"gold": {"a", "b"}, "candidates": ["a", "b", "c"], "reranked": ["a", "a", "c"]},
            # 重排返回了足够多文档，但标准文档 b 被排到第 10 名之后。
            "pushed": {"gold": {"a", "b"}, "candidates": ["a", "b"] + many[:8],
                       "reranked": ["a"] + many[:10] + ["b"]},
            # 重排后仍然完整覆盖，不算损失。
            "kept": {"gold": {"a"}, "candidates": ["a"], "reranked": ["a"]},
            # 候选阶段就没有覆盖全部标准文档，损失不归因于重排。
            "never": {"gold": {"a", "z"}, "candidates": ["a"], "reranked": ["a"]},
        }
        self.assertEqual(
            diagnose_rerank_loss(traces, metrics.unique_in_order),
            {
                "full10_lost_by_rerank": 2,
                "of_which_fewer_than_10_documents_returned": 1,
                "of_which_gold_ranked_below_10": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
