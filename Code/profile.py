from dataclasses import dataclass
from pathlib import Path

from config import DATA_PATH
from utils import load_json, normalize_concept


ACTIVITY_MEAN = 0.039885658914728686
DIVERSITY_MEAN = 0.06271615720524018


@dataclass(frozen=True)
class LearnerProfileValues:
    user_id: int
    activity_ratio: float
    diversity_ratio: float
    preference: str
    success_rate_value: float
    ability_value: float


class Profile:
    """Learner Profile module.

    It consumes the profile file produced by ``Code/prepare/build_profile.py``.
    The six fields follow the paper implementation:
    ``student_id, activity, diversity, preference, success_rate, IRT ability``.
    """

    def __init__(self, agent_id: int, data_path: str | Path | None = None):
        path = Path(data_path or DATA_PATH) / "profile.json"
        data = load_json(path)
        raw = data.get(str(agent_id))
        if raw is None:
            raise KeyError(f"No profile for agent/student id {agent_id} in {path}")
        parts = str(raw).strip().split("\t")
        if len(parts) < 6:
            raise ValueError(f"Profile row must contain six tab-separated fields: {raw!r}")
        self.values = LearnerProfileValues(
            user_id=int(float(parts[0])),
            activity_ratio=float(parts[1]),
            diversity_ratio=float(parts[2]),
            preference=normalize_concept(parts[3]),
            success_rate_value=float(parts[4]),
            ability_value=float(parts[5]),
        )

    def activity(self) -> str:
        return "high" if self.values.activity_ratio > ACTIVITY_MEAN else "low"

    def diversity(self) -> str:
        return "high" if self.values.diversity_ratio > DIVERSITY_MEAN else "low"

    def preference(self) -> str:
        return self.values.preference

    def success_rate(self) -> str:
        ar = self.values.success_rate_value
        if ar > 0.6:
            return "high"
        if ar > 0.3:
            return "medium"
        return "low"

    def ability(self) -> str:
        ab = self.values.ability_value
        if ab > 0.5:
            return "good"
        if ab > 0.4:
            return "common"
        return "poor"

    def build_prompt(self) -> str:
        tips_act = {
            "high": "you maintain a high level of online exercise activity and practice frequently",
            "low": "you practice less regularly and with lower enthusiasm",
        }
        tips_div = {
            "high": "you explore diverse knowledge categories",
            "low": "you focus on limited knowledge categories",
        }
        act = self.activity()
        div = self.diversity()
        return (
            "You are a high school student engaging in self-directed exercising on an online learning platform. "
            f"During online study, you exhibit {act} activity, which means {tips_act[act]}. "
            f"You have {div} knowledge diversity, which means {tips_div[div]}. "
            f"The knowledge concept you practice most often is: {self.preference()}. "
            f"Your success rate is {self.success_rate()}. "
            f"You possess {self.ability()} analytical and problem-solving skills."
        )
