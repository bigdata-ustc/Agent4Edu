import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据集路径

DATA_PATH = os.path.join(BASE_DIR, '../data/demo')
# 结果路径
RESULT_PATH = os.path.join(BASE_DIR, 'simulation')

# OpenAI 配置
OPENAI_API_KEY = ''

# 模拟参数
SIM_PARAMS = {
    'memory_source':    'real',
    'learning_effect':  'yes',
    'forgetting_effect':'yes',
    'reflection_choice':'yes',
    'sim_strategy':     'performance',
    'gpt_type':         0,  # 0: GPT-3.5, 1: GPT-4
    'short_term_size':  5,
    'long_term_thresh': 5,
    'forget_lambda':    0.99
}
