# 安装：pip install transformers torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class FreeLocalChatbot:
    def __init__(self):
        # 使用较小的中文模型
        model_name = "blanchefort/rubert-base-cased-sentiment"
        
        # 或者用这些免费模型：
        # - "bert-base-chinese"
        # - "uer/gpt2-chinese-cluecorpussmall"
        # - "IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese"
        
        self.model = pipeline("text-generation", model=model_name)
    
    def respond(self, question):
        # 本地运行，无需联网
        response = self.model(question, max_length=100)
        return response[0]["generated_text"]

# 缺点：需要下载模型（几GB），速度较慢