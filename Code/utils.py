import json, os

def load_json(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# 以下为论文中 Memory Module 所需的相似度判定函数示例
def compute_similarity(record, memory_list, kcg, know_course_list):
    '''
    计算 record 与 memory_list 中每条事实的知识点相似度。
    kcg: 知识点关联图（元组对集合）
    know_course_list: 知识点->课程 映射
    record: [text, concept, correct_flag, counter]
    memory_list: [[text, concept, correct_flag, counter], ...]
    返回: [0/1 列表]
    '''
    sim = []
    rec = record[1].lower().strip().replace('\"','')
    rec_id = know_course_list.get(rec)
    for m in memory_list:
        mem = m[1].lower().strip().replace('\"','')
        mem_id = know_course_list.get(mem)
        if (rec_id, mem_id) in kcg or (mem_id, rec_id) in kcg:
            sim.append(1)
        else:
            # 同一课程内，有 20% 概率认为相似
            if rec_id == mem_id and random.random() > 0.8:
                sim.append(1)
            else:
                sim.append(0)
    return sim

# 可选：基于 LLM 的相似度判断（如需更精确的文本相似度，可调用 _response_llm_gpt）
# def llm_compute_similarity(record, memory_list, llm_client):
#     pass