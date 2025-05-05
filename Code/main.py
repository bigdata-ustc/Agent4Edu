import asyncio
from config import DATA_PATH, RESULT_PATH
from profile import Profile
from memory import Memory
from action import AgentAction
from utils import load_json, save_json

async def run_for_student(student_id):
    logs = load_json(f"{DATA_PATH}/stu_logs.json")[student_id]['logs']
    KCG = load_json(f"{DATA_PATH}/kcg.json")
    know_course_list = load_json(f"{DATA_PATH}/know_course_list.json")
    know_name = load_json(f"{DATA_PATH}/know_name_list.json")
    profile = Profile(student_id)
    memory = Memory(KCG, know_course_list, know_name)
    action = AgentAction(profile, memory)
    results = []

    for rec_id in range(len(logs)):
        rec = logs[rec_id]
        ans, raw, corr, summ = action.simulate_step(rec, rec_id+1, similarity_fn=memory.reinforce)
        results.append({'ans':ans,'raw':raw,'corr':corr,'summ':summ})

    save_json(f"{RESULT_PATH}/{student_id}_results.json", results)

async def main():
    # students = range(1)
    user_idx_start = 0
    user_idx_end = 1
    agent_id_list = load_json(f"{DATA_PATH}/agent_id_list.json")
    students = agent_id_list[user_idx_start:user_idx_end]
    
    await asyncio.gather(*[run_for_student(s) for s in students])

if __name__ == '__main__':
    asyncio.run(main())
