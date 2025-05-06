import openai
from openai import OpenAI
from config import OPENAI_API_KEY, SIM_PARAMS

class LLMClient:
    """
    LLM 接口调用封装模块
    用于统一处理 GPT / DeepSeek 等模型的调用
    """
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        mtype = SIM_PARAMS['gpt_type']
        if mtype == 0:
            self.model = 'gpt-3.5-turbo-1106'
        elif mtype == 1:
            self.model = 'gpt-4-1106-preview'
        # elif mtype == 2:
        #     self.model = 'deepseek-chat'
        else:
            raise ValueError(f"Unsupported gpt_type: {mtype}")

    def call(self, messages):
        print(messages)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            timeout=120,
            max_tokens=2048
        )
        return resp.choices[0].message.content
