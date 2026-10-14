from __future__ import annotations

import copy
import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentsearch.agent import Agent, AgentFailure
from agentsearch.cli import run_batch
from agentsearch.config import ROOT, load_config
from agentsearch.data import chunks, questions
from agentsearch.evaluation import evaluate, normalize_answer, validate_file, validate_record
from agentsearch.llm import APIClient, Reply, ProtocolError
from agentsearch.retrieval import Retrieval, build_corpus, build_dense
from agentsearch.skills import Skills
from agentsearch.state import SearchState


class ScriptedClient:
    """Test-only fixture. Never exposed in submission CLI or UI."""
    is_real = False
    model = "offline-test"

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages):
        self.requests.append(messages)
        if not self.replies:
            raise RuntimeError("No more test replies")
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        content = item if isinstance(item, str) else json.dumps(item)
        return Reply(content, 13, {"usage": {"total_tokens": 13},
                                  "choices": [{"message": {"content": content}}]})


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT / "configs/sparse.json")
        self.config["index_dir"] = str(self.root / "index")
        self.config["chunk_chars"] = 90
        self.config["chunk_overlap"] = 20
        self.source = self.root / "docs.jsonl"
        docs = [
            {"doc_id": "d1", "text": "Mira Vale founded the Lantern Archive in 2007. " * 4, "source_url": "https://example.org/one"},
            {"doc_id": "d2", "text": "Mira Vale received the Silver Quill Award in 2018.", "source_url": "https://example.org/two"},
            {"doc_id": "d3", "text": "Jonas Reed founded the Mountain Library in 2009.", "source_url": "https://example.org/three"}]
        self.source.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
        build_corpus(str(self.source), self.config)
        self.retrieval = Retrieval(self.config)

    def tearDown(self):
        self.retrieval.close()
        self.temp.cleanup()

    def agent(self, replies, **changes):
        config = copy.deepcopy(self.config)
        config.update(changes)
        return Agent(self.retrieval, ScriptedClient(replies), config)

    @staticmethod
    def retrieve(query="Lantern Archive"):
        return {"action": "retrieve", "args": {"query": query, "top_k": 2, "method": "sparse"}}

    @staticmethod
    def final(ids=None):
        return {"action": "final", "answer": "Mira Vale", "evidence_ids": ids or ["e1"]}

    def test_chunk_offsets_cover_tail_without_infinite_loop(self):
        text = "x" * 251
        pieces = list(chunks(text, 90, 20))
        self.assertEqual(pieces[-1][1], len(text))
        for start, end, part in pieces:
            self.assertEqual(part, text[start:end])
            self.assertLessEqual(len(part), 90)
        with self.assertRaises(ValueError):
            list(chunks(text, 20, 20))

    def test_sparse_search_deduplicates_documents_and_keeps_provenance(self):
        result = self.retrieval.retrieve("Lantern Archive", 8, "sparse")["results"]
        self.assertEqual(result[0]["doc_id"], "d1")
        self.assertEqual(len({x["doc_id"] for x in result}), len(result))
        hit = result[0]
        original = self.retrieval.db.execute("SELECT text FROM documents WHERE doc_id=?", (hit["doc_id"],)).fetchone()[0]
        self.assertEqual(original[hit["start"]:hit["end"]], hit["snippet"])

    def test_open_and_find_return_bounded_real_passages(self):
        page = self.retrieval.open_document("d1", 10, 40)
        self.assertEqual(len(page["text"]), 40)
        found = self.retrieval.find_in_document("d2", "silver")["matches"]
        self.assertIn("Silver Quill", found[0]["text"])
        with self.assertRaises(ValueError):
            self.retrieval.open_document("unknown")

    def test_unbuilt_retriever_does_not_silently_fall_back(self):
        config = dict(self.config, methods=["sparse", "dense"])
        with self.assertRaises(FileNotFoundError):
            Retrieval(config)
        with self.assertRaises(ValueError):
            self.retrieval.retrieve("foo", method="instruction")

    def test_skills_load_only_after_selection_and_path_is_confined(self):
        skills = Skills(self.config["skills_dir"])
        self.assertEqual(len(skills.metadata()), 2)
        self.assertNotIn("instructions", skills.metadata()[0])
        self.assertIn("instructions", skills.load("search-recovery"))
        with self.assertRaises(ValueError):
            skills.load("../config.json")

    def test_skill_agent_loop_and_exact_usage_accounting(self):
        a = self.agent([{"action": "load_skill", "args": {"name": "multi-clue-search"}},
                        self.retrieve(), self.final(), {"supported": True, "issues": []}])
        row = a.run("Who founded Lantern Archive?", "q1")
        validate_record(row)
        self.assertEqual(row["tool_calls"], 2)
        self.assertEqual(row["total_tokens"], 52)
        self.assertEqual(row["trajectory"][-1]["selected_skills"], ["multi-clue-search"])
        first = json.loads(a.client.requests[0][1]["content"])
        self.assertEqual(first["loaded_skills"], {})
        second = json.loads(a.client.requests[1][1]["content"])
        self.assertIn("multi-clue-search", second["loaded_skills"])
        with self.assertRaises(ValueError):
            validate_record(row, course=True)

    def test_verification_failure_triggers_further_search(self):
        a = self.agent([self.retrieve(), self.final(), {"supported": False, "issues": ["Award not checked"]},
                        self.retrieve("Silver Quill Award"), self.final(["e1", "e2"]), {"supported": True}])
        row = a.run("Who founded the archive and received the award?")
        self.assertEqual(row["tool_calls"], 2)
        self.assertEqual(row["total_tokens"], 78)
        self.assertEqual(len([e for e in row["trajectory"] if e["type"] == "answer_verification"]), 2)

    def test_malformed_json_repair_is_counted(self):
        a = self.agent(["not json", self.retrieve(), self.final(), {"supported": True}])
        row = a.run("Who founded Lantern Archive?")
        self.assertEqual(row["total_tokens"], 52)
        self.assertTrue(any(e["type"] == "format_error" for e in row["trajectory"]))

    def test_unknown_citation_cannot_be_submitted(self):
        a = self.agent([self.retrieve(), self.final(["e999"])], max_rounds=2)
        with self.assertRaises(AgentFailure) as caught:
            a.run("Who founded Lantern Archive?")
        self.assertTrue(any(e["type"] == "candidate_rejected" for e in caught.exception.state.trajectory))

    def test_duplicate_calls_count_but_do_not_execute_again(self):
        a = self.agent([self.retrieve(), self.retrieve(), self.final(), {"supported": True}])
        row = a.run("Who founded Lantern Archive?")
        calls = [e for e in row["trajectory"] if e["type"] == "tool_call"]
        self.assertIn("Duplicate", calls[1]["result"]["error"])
        self.assertEqual(row["tool_calls"], 2)

    def test_context_is_bounded_and_trace_is_retained(self):
        state = SearchState("Who?")
        for i in range(30):
            state.collect({"doc_id": f"d{i}", "text": "x" * 5000})
            state.event("tool_call", result={"text": "x" * 10000})
        config = dict(self.config, context_chars=5000)
        view = state.context(config)
        self.assertLessEqual(len(json.dumps(view, ensure_ascii=False)), 5000)
        self.assertEqual(len(state.trajectory), 30)
        self.assertEqual(len(state.evidence), 30)
        self.assertLess(len(view["evidence"]), 30)

    def test_single_pass_baseline_has_one_retrieval(self):
        a = self.agent([self.final()])
        row = a.run("Who founded Lantern Archive?", mode="single")
        self.assertEqual(row["tool_calls"], 1)
        self.assertEqual(row["total_tokens"], 13)

    def test_failed_api_retains_trace_and_is_not_valid_prediction(self):
        a = self.agent([self.retrieve(), RuntimeError("quota exhausted")])
        with self.assertRaises(AgentFailure) as caught:
            a.run("Who founded Lantern Archive?")
        self.assertEqual(caught.exception.state.total_tokens, 13)
        self.assertEqual(caught.exception.state.trajectory[-1]["type"], "llm_error")

    def test_protocol_failure_keeps_known_billed_usage(self):
        error = ProtocolError("Missing content", {"usage": {"total_tokens": 77}})
        a = self.agent([self.retrieve(), error])
        with self.assertRaises(AgentFailure) as caught:
            a.run("Who?")
        self.assertEqual(caught.exception.state.total_tokens, 90)

    def test_questions_do_not_pass_gold_to_agent(self):
        file = self.root / "questions.jsonl"
        file.write_text(json.dumps({"id": 1, "question": "Who?", "answer": "SECRET GOLD"}), encoding="utf-8")
        self.assertEqual(list(questions(file)), [{"id": "1", "question": "Who?"}])

    def test_batch_evaluate_and_submission_validation(self):
        file = self.root / "questions.jsonl"
        file.write_text(json.dumps({"id": "q1", "question": "Who founded Lantern Archive?", "answer": "Mira Vale"}), encoding="utf-8")
        output = self.root / "pred.jsonl"
        a = self.agent([self.retrieve(), self.final(), {"supported": True}])
        run_batch(a, file, output, expected_count=1)
        self.assertEqual(validate_file(output, file, expected_count=1)["records"], 1)
        self.assertEqual(evaluate(output, file)["normalized_em"], 1)
        with self.assertRaises(ValueError):
            validate_file(output, file, expected_count=100)
        with self.assertRaises(ValueError):
            validate_file(output, file, expected_count=1, course=True)

    def test_normalized_em_is_exact_not_keyword_hit_rate(self):
        self.assertEqual(normalize_answer("The Mira, Vale!"), "mira vale")
        self.assertNotEqual(normalize_answer("It is Mira Vale"), normalize_answer("Mira Vale"))

    def test_tampering_with_usage_is_detected(self):
        row = self.agent([self.retrieve(), self.final(), {"supported": True}]).run("Who?")
        row["total_tokens"] += 1
        with self.assertRaises(ValueError):
            validate_record(row)

    def test_api_response_usage_is_required_and_preserved(self):
        env = {"LLM_API_KEY": "test-not-a-real-key", "LLM_MODEL": "course-approved-test-id"}
        raw = {"choices": [{"message": {"content": '{"action":"final"}'}}], "usage": {"total_tokens": 123}}
        with patch.dict("os.environ", env), patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())):
            reply = APIClient().complete([{"role": "user", "content": "Test"}])
            self.assertEqual(reply.total_tokens, 123)
        del raw["usage"]
        with patch.dict("os.environ", env), patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())):
            with self.assertRaises(ProtocolError):
                APIClient().complete([])

    def test_resume_blocks_inflight_question_with_unknown_usage(self):
        file = self.root / "questions.jsonl"
        file.write_text(json.dumps({"id": "q1", "question": "Who?"}), encoding="utf-8")
        output = self.root / "interrupted.jsonl"
        Path(str(output) + ".inflight.json").write_text('{"id":"q1"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            run_batch(self.agent([]), file, output, resume=True, expected_count=1)

    def test_dense_and_instruction_faiss_pipeline_and_rrf(self):
        try:
            import faiss
            import numpy as np
        except ImportError:
            self.skipTest("Optional FAISS dependency is not installed")
        recorded = []

        class Encoder:
            def encode(self, texts, **options):
                recorded.append((texts, options))
                vectors = []
                for text in texts:
                    lower = text.lower()
                    vector = np.array([1.0 if "lantern" in lower else 0.1,
                                       1.0 if "silver" in lower else 0.1,
                                       1.0 if "mountain" in lower else 0.1], dtype="float32")
                    vectors.append(vector / np.linalg.norm(vector))
                return np.array(vectors)

        config = dict(self.config, methods=["sparse", "dense", "instruction"])
        with patch("agentsearch.retrieval.load_encoder", return_value=Encoder()):
            build_dense(config, "dense")
            build_dense(config, "instruction")
            retrieval = Retrieval(config)
            try:
                dense = retrieval.retrieve("Lantern", method="dense")["results"]
                instruction = retrieval.retrieve("Silver", method="instruction")["results"]
                hybrid = retrieval.retrieve("Silver", method="hybrid")["results"]
                self.assertEqual(dense[0]["doc_id"], "d1")
                self.assertEqual(instruction[0]["doc_id"], "d2")
                self.assertEqual(hybrid[0]["doc_id"], "d2")
                self.assertEqual(set(hybrid[0]["retrievers"]), {"sparse", "dense", "instruction"})
            finally:
                retrieval.close()
        self.assertTrue(any(texts[0].startswith("passage: ") for texts, _ in recorded))
        self.assertTrue(any(texts[0].startswith("query: ") for texts, _ in recorded))
        self.assertTrue(any(options.get("prompt", "").startswith("Instruct: ") for _, options in recorded))


if __name__ == "__main__":
    unittest.main()
