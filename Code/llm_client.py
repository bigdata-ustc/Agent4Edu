import os
import re
from typing import Any

from config import OPENAI_API_KEY, OPENAI_BASE_URL, SIM_PARAMS


class LLMClient:
    """Small LLM wrapper used by the Action module.

    The default provider is ``mock`` so the repository can be smoke-tested without
    network access or API keys. Set ``AGENT4EDU_LLM_PROVIDER=openai`` for real
    simulation.
    """

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or SIM_PARAMS.get("llm_provider", "mock")
        self.model = model or self._default_model()
        self.client: Any | None = None
        if self.provider == "openai":
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            self.client = OpenAI(**kwargs)

    def _default_model(self) -> str:
        mtype = SIM_PARAMS.get("gpt_type", 0)
        if mtype == 0:
            return os.getenv("AGENT4EDU_OPENAI_MODEL", "gpt-3.5-turbo-1106")
        if mtype == 1:
            return os.getenv("AGENT4EDU_OPENAI_MODEL", "gpt-4-1106-preview")
        return os.getenv("AGENT4EDU_OPENAI_MODEL", "gpt-4o-mini")

    def call(self, messages: list[dict[str, str]]) -> str:
        if self.provider == "mock":
            return self._mock_call(messages)
        if self.provider != "openai":
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is empty. Set it or use AGENT4EDU_LLM_PROVIDER=mock.")
        assert self.client is not None
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            timeout=120,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    def _mock_call(self, messages: list[dict[str, str]]) -> str:
        user = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        if "reflection" in user.lower() and "learning status" in user.lower():
            return (
                "The recent practice suggests a stable learning state. The learner should keep reinforcing "
                "concepts that appeared in recent mistakes and connect them with previously practiced concepts."
            )
        concept = self._extract_first_option(user) or "unknown concept"
        return (
            "Task1: Yes\n"
            f"Task2: {concept}\n"
            "Task3: I will use the relevant concept and previous practice experience to identify the key condition, "
            "then derive the answer step by step.\n"
            "Task4: Yes"
        )

    @staticmethod
    def _extract_first_option(text: str) -> str | None:
        # The prompt prints Task-2 concept options as '- concept'. Use the first Task-2 option
        # for deterministic smoke tests, but ignore earlier memory/proficiency bullets.
        segment = text
        marker = "Task 2 is to choose one knowledge concept tested by this exercise from the following three options:"
        if marker in text:
            segment = text.split(marker, 1)[1].split("Only output", 1)[0]
        for line in segment.splitlines():
            line = line.strip()
            if line.startswith("-") and len(line) > 1:
                return line[1:].strip()
        match = re.search(r"Knowledge Concept.*?:\s*(.+)", text)
        if match:
            return match.group(1).strip()
        return None
