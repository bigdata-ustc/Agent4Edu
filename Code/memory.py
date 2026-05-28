import math
import random
from pathlib import Path
from typing import Any

from config import SIM_PARAMS
from tools.knowledge_proficiency import KnowledgeProficiency
from tools.relation_graph import RelationGraph
from utils import normalize_concept


class Memory:
    """Memory module aligned with the paper pipeline.

    It contains factual memory, short-term memory, long-term memory, KCG-based
    memory reinforcement, forgetting, and reflection support.
    """

    def __init__(self, student_id: int, data_path: str | Path):
        self.student_id = int(student_id)
        self.factual: list[list[Any]] = []
        self.short: list[list[Any]] = []
        self.long: dict[str, list[Any]] = {
            "significant_facts": [],
            "learning_status": [],
            "knowledge_proficiency": [],
            "practiced_knowledge": [],
        }
        self.threshold = int(SIM_PARAMS["long_term_thresh"])
        self.short_size = int(SIM_PARAMS["short_term_size"])
        self.forget_lambda = float(SIM_PARAMS["forget_lambda"])
        self.relation_graph = RelationGraph(data_path)
        self.knowledge_proficiency = KnowledgeProficiency(data_path)

    def retrieve_short(self) -> list[list[Any]]:
        self.short = self.factual[-self.short_size:]
        return self.short

    def retrieve_long(self, current_concept: str | None = None, time_step: int = 1) -> dict[str, Any]:
        concepts = list(self.long["practiced_knowledge"])
        if current_concept:
            concepts.append(normalize_concept(current_concept))
        kp = self._proficiency_context(concepts, time_step)
        self.long["knowledge_proficiency"] = kp
        return {
            "significant_facts": self.long["significant_facts"],
            "learning_status": self.long["learning_status"],
            "knowledge_proficiency": kp,
            "practiced_knowledge": self.long["practiced_knowledge"],
        }

    def _proficiency_context(self, concepts: list[str], time_step: int) -> list[dict[str, Any]]:
        seen: set[str] = set()
        context: list[dict[str, Any]] = []
        for concept in concepts:
            norm = normalize_concept(concept)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            value = self.knowledge_proficiency.value(self.student_id, norm, max(0, time_step - 2))
            context.append({"concept": norm, "value": value, "level": self.knowledge_proficiency.tier(value)})
        return context

    def similarity_by_kcg(self, record: list[Any]) -> list[int]:
        sim: list[int] = []
        current = normalize_concept(record[1])
        for memory_element in self.factual:
            previous = normalize_concept(memory_element[1])
            if self.relation_graph.is_related(current, previous):
                sim.append(1)
            elif self.relation_graph.same_course(current, previous) and random.random() > 0.8:
                sim.append(1)
            else:
                sim.append(0)
        return sim

    def reinforce(self, record: list[Any]) -> list[int]:
        sim_list = self.similarity_by_kcg(record)
        for i, sim in enumerate(sim_list):
            self.factual[i][3] += sim

        max_count = max([r[3] for r in self.factual] + [1])
        self.factual.append([record[0], normalize_concept(record[1]), int(record[2]), max_count])
        self._promote_to_long_term()
        return sim_list

    def write_without_reinforcement(self, record: list[Any]) -> None:
        self.factual.append([record[0], normalize_concept(record[1]), int(record[2]), 1])
        self._promote_to_long_term()

    def _promote_to_long_term(self) -> None:
        existing = {fact[-1] for fact in self.long["significant_facts"] if fact}
        for idx, rec in enumerate(self.factual):
            fact_id = idx + 1
            if rec[3] >= self.threshold and fact_id not in existing:
                promoted = list(rec) + [fact_id]
                self.long["significant_facts"].append(promoted)
                rec[3] = 1

    def forget(self, time_step: int) -> None:
        kept: list[Any] = []
        for fact in self.long["significant_facts"]:
            fact_id = int(fact[-1])
            if 1 / (1 + math.exp(-(time_step - fact_id))) <= self.forget_lambda:
                kept.append(fact)
            else:
                factual_index = fact_id - 1
                if 0 <= factual_index < len(self.factual):
                    self.factual[factual_index][3] = 1
        self.long["significant_facts"] = kept

    def update_practiced_knowledge(self, concept: str) -> None:
        concept = normalize_concept(concept)
        if concept and concept not in self.long["practiced_knowledge"]:
            self.long["practiced_knowledge"].append(concept)

    def write_long_summary(self, summary: str) -> None:
        if summary:
            self.long["learning_status"].append(summary)

    def reflect_corrective(self, practice: dict[str, Any], ans: dict[str, str]) -> str:
        fb = ""
        true_concept = normalize_concept(practice["know_name"])
        pred_concept = normalize_concept(ans.get("task2", ""))
        if true_concept != pred_concept:
            fb += (
                "The knowledge tested by this question is "
                f"{true_concept} but you wrongly think the knowledge is "
                f"{pred_concept or 'unknown knowledge'}.\n"
            )
        pred_correct = normalize_concept(ans.get("task4", ""))
        true_score = int(practice.get("score", 0))
        if pred_correct != "yes" and true_score == 1:
            fb += "You thought you could not solve this problem correctly, but in fact, you will solve it correctly.\n"
        if pred_correct != "no" and true_score == 0:
            fb += "You thought you could solve this problem correctly, but in fact, you do not answer it correctly.\n"
        return fb

    @staticmethod
    def reflection_instruction(corrective: str) -> str:
        return (
            (corrective or "")
            + "\nYou should directly output your reflection and summarize your # Learning Status # within 500 words "
            + "based on your # profile #, # short-term memory #, # long-term memory # and previous # Learning Status #. "
            + "Do not output any other information."
        )
