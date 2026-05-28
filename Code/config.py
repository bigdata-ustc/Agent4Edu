import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = Path(os.getenv("AGENT4EDU_DATA_PATH", PROJECT_ROOT / "data" / "demo"))
RESULT_PATH = Path(os.getenv("AGENT4EDU_RESULT_PATH", BASE_DIR / "simulation"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

SIM_PARAMS = {
    "memory_source": os.getenv("AGENT4EDU_MEMORY_SOURCE", "real"),
    "learning_effect": os.getenv("AGENT4EDU_LEARNING_EFFECT", "yes"),
    "forgetting_effect": os.getenv("AGENT4EDU_FORGETTING_EFFECT", "yes"),
    "reflection_choice": os.getenv("AGENT4EDU_REFLECTION", "yes"),
    "sim_strategy": os.getenv("AGENT4EDU_SIM_STRATEGY", "performance"),
    "gpt_type": int(os.getenv("AGENT4EDU_GPT_TYPE", "0")),
    "short_term_size": int(os.getenv("AGENT4EDU_SHORT_TERM_SIZE", "5")),
    "long_term_thresh": int(os.getenv("AGENT4EDU_LONG_TERM_THRESH", "5")),
    "forget_lambda": float(os.getenv("AGENT4EDU_FORGET_LAMBDA", "0.99")),
    "llm_provider": os.getenv("AGENT4EDU_LLM_PROVIDER", "mock"),
}
