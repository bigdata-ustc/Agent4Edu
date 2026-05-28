import re
from typing import Any

from llm_client import LLMClient
from memory import Memory
from profile import Profile
from utils import normalize_concept


class Action:
    """Action module: generate and parse four learner-simulation tasks."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def simulate_practice(self, profile: Profile, memory: Memory, practice: dict[str, Any], step: int) -> dict[str, Any]:
        true_concept = normalize_concept(practice["know_name"])
        short_memory = memory.retrieve_short()
        long_memory = memory.retrieve_long(current_concept=true_concept, time_step=step)
        options = self._concept_options(memory, true_concept, seed=step + memory.student_id)

        system_prompt = profile.build_prompt() + "\nThe information above is your # profile #."
        user_prompt = self._build_action_prompt(practice, short_memory, long_memory, options)
        response = self.llm.call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        parsed = self._parse_tasks(response)

        task_scores = self._score_tasks(practice, parsed)
        corrective = memory.reflect_corrective(practice, parsed)
        reflection = self.llm.call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": Memory.reflection_instruction(corrective)},
        ])

        record = [practice.get("exer_content", ""), true_concept, int(practice.get("score", 0)), 1]
        if memory is not None:
            memory.reinforce(record)
            memory.update_practiced_knowledge(true_concept)
            memory.write_long_summary(reflection)
            memory.forget(step)

        return {
            "exercise_id": practice.get("exer_id"),
            "true_score": int(practice.get("score", 0)),
            "true_concept": true_concept,
            "concept_options": options,
            "prompt": user_prompt,
            "raw_response": response,
            "parsed_response": parsed,
            "task_scores": task_scores,
            "reflection": reflection,
        }

    def _concept_options(self, memory: Memory, true_concept: str, seed: int) -> list[str]:
        distractors = memory.relation_graph.sample_distractors(true_concept, k=2, seed=seed)
        return [true_concept] + distractors

    def _build_action_prompt(
        self,
        practice: dict[str, Any],
        short_memory: list[list[Any]],
        long_memory: dict[str, Any],
        concept_options: list[str],
    ) -> str:
        chunks: list[str] = []
        if short_memory:
            chunks.append("I will give you recent practice records as # Recent Facts # below.")
            for idx, item in enumerate(short_memory, start=1):
                result = "rightly" if int(item[2]) == 1 else "wrongly"
                chunks.append(
                    f"Record{idx}: You {result} answered an exercise.\n"
                    f"- # Textual Content #: {item[0]}\n"
                    f"- # Knowledge Concept #: {item[1]}"
                )
            chunks.append("The information above is your # short-term memory #.")

        sig_facts = long_memory.get("significant_facts", [])
        if sig_facts:
            chunks.append("I will give you important reinforced records as # Reinforced Facts # below.")
            for idx, item in enumerate(sig_facts, start=1):
                result = "rightly" if int(item[2]) == 1 else "wrongly"
                chunks.append(
                    f"Record{idx}: You {result} answered an exercise.\n"
                    f"- # Textual Content #: {item[0]}\n"
                    f"- # Knowledge Concept #: {item[1]}"
                )

        kp = long_memory.get("knowledge_proficiency", [])
        if kp:
            chunks.append("Your current # Knowledge Proficiency # is:")
            for row in kp:
                if row.get("value") is None:
                    chunks.append(f"- {row['concept']}: unknown")
                else:
                    chunks.append(f"- {row['concept']}: {row['level']} ({row['value']:.4f})")

        status = long_memory.get("learning_status", [])
        if status:
            chunks.append("Your current # Learning Status # is summarized as:")
            chunks.append(str(status[-1]))
        if sig_facts or kp or status:
            chunks.append("The information above is your # long-term memory #.")

        question = (
            "Currently, you start to answer the recommended exercise. Its content information is as follows:\n\n"
            f"# Textual Content #: {practice.get('exer_content', '')}\n\n"
            f"# Options #: {practice.get('exer_option', '')}\n"
            f"# Reference Answer #: {practice.get('exer_answer', '')}\n"
            f"# Analysis #: {practice.get('exer_analysis', '')}\n"
        )
        chunks.append(question)
        chunks.append("To answer this exercise, please complete the following four tasks in sequence:")
        chunks.append(
            "Task 1 is to decide whether to attempt the recommended problem based on your ability in Profile "
            "and knowledge proficiency in Long-term Memory. If you consider the problem too difficult, output \"No\"; otherwise output \"Yes\". "
            "Regardless of your choice, the subsequent tasks will still be executed."
        )
        chunks.append("Task 2 is to choose one knowledge concept tested by this exercise from the following three options:")
        chunks.extend([f"- {c}" for c in concept_options])
        chunks.append("Only output the knowledge concept and do not output any other information for Task 2.")
        chunks.append(
            "Task 3 is to design a short problem-solving idea for this question based on your profile, memory and learning status, "
            "and then give a final answer. Your response should align with your profile, memory, and past performance."
        )
        chunks.append(
            "Task 4 is to estimate whether you can correctly solve this problem based on your profile, learning records, "
            "learning status, and problem-solving idea. If you can correctly solve it, answer \"Yes\"; otherwise answer \"No\"."
        )
        chunks.append(
            "Output exactly in this format:\n"
            "Task1: <answer for task1>\n"
            "Task2: <answer for task2>\n"
            "Task3: <answer for task3>\n"
            "Task4: <answer for task4>"
        )
        return "\n\n".join(chunks)

    @staticmethod
    def _parse_tasks(text: str) -> dict[str, str]:
        normalized = text
        normalized = re.sub(r"(?i)task\s*1\s*:", "Task1:", normalized)
        normalized = re.sub(r"(?i)task\s*2\s*:", "Task2:", normalized)
        normalized = re.sub(r"(?i)task\s*3\s*:", "Task3:", normalized)
        normalized = re.sub(r"(?i)task\s*4\s*:", "Task4:", normalized)
        out: dict[str, str] = {}
        for name, next_name in [("task1", "Task2:"), ("task2", "Task3:"), ("task3", "Task4:"), ("task4", None)]:
            label = name.capitalize().replace("Task", "Task")
            start_token = label + ":"
            if start_token not in normalized:
                out[name] = ""
                continue
            part = normalized.split(start_token, 1)[1]
            if next_name and next_name in part:
                part = part.split(next_name, 1)[0]
            out[name] = part.strip().strip('"').strip()
        return out

    @staticmethod
    def _score_tasks(practice: dict[str, Any], ans: dict[str, str]) -> dict[str, int]:
        score = int(practice.get("score", 0))
        true_concept = normalize_concept(practice.get("know_name", ""))
        task1 = normalize_concept(ans.get("task1", ""))
        task2 = normalize_concept(ans.get("task2", ""))
        task4 = normalize_concept(ans.get("task4", ""))
        flag1 = int((task1 == "yes" and score == 1) or (task1 == "no" and score == 0))
        flag2 = int(task2 == true_concept)
        flag4 = int(((task4 == "yes" and score == 1) or (task4 == "no" and score == 0)) and flag1 == 1)
        return {"task1": flag1, "task2": flag2, "task4": flag4}
