import json
from config import DATA_PATH

class Profile:
    """
    Learner Profile Module (§4.1.4.2)
    提取 explicit practice styles 和 implicit ability
    """
    def __init__(self, agent_id):
        path = f"{DATA_PATH}/profile.json"
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        # raw: "id\tactivity\tdiversity\tpreference\tsuccess_rate\tabs"
        self.values = data[str(agent_id)].split('\t')

    def activity(self):
        mean = 0.0398856589
        return 'high' if float(self.values[1]) > mean else 'low'

    def diversity(self):
        mean = 0.0627161572
        return 'high' if float(self.values[2]) > mean else 'low'

    def preference(self):
        return self.values[3]

    def success_rate(self):
        ar = float(self.values[4])
        if ar > 0.6: return 'high'
        if ar > 0.3: return 'medium'
        return 'low'

    def ability(self):
        ab = float(self.values[5])
        if ab > 0.5: return 'good'
        if ab > 0.4: return 'common'
        return 'poor'

    def build_prompt(self):
        tips_act = {
            'high': 'you maintain a high level of online exercise activity and practice frequently',
            'low':  'you practice less regularly and with lower enthusiasm'
        }
        tips_div = {
            'high': 'you explore diverse knowledge categories',
            'low':  'you focus on limited knowledge categories'
        }
        return (
            f"You are a student with {self.activity()} activity, "
            f"{tips_act[self.activity()]}. "
            f"You have {self.diversity()} diversity, "
            f"{tips_div[self.diversity()]}. "
            f"Most practiced concept: {self.preference()}. "
            f"Success rate: {self.success_rate()}. "
            f"Problem-solving ability: {self.ability()}."
        )
        