from typing import Dict, List
import re

class IntentClassifier:
    def __init__(self):
        # 意图和对应的模式
        self.intent_patterns = {
            "greeting": [
                r"你好", r"您好", r"hi", r"hello", r"早上好", r"下午好"
            ],
            "ask_name": [
                r"叫什么", r"名字", r"你是谁", r"你是哪个"
            ],
            "ask_weather": [
                r"天气", r"下雨", r"晴天", r"温度", r"天气预报"
            ],
            "ask_price": [
                r"价格", r"多少钱", r"价钱", r"cost", r"price", r"贵不贵"
            ],
            "goodbye": [
                r"再见", r"拜拜", r"88", r"下次聊", r"不说了"
            ]
        }
        
        self.intent_responses = {
            "greeting": "您好！我是AI助手，很高兴为您服务！",
            "ask_name": "我是智能客服助手，您可以叫我小助手。",
            "ask_weather": "请告诉我您想查询哪个城市的天气？",
            "ask_price": "请问您想了解哪个产品的价格呢？",
            "goodbye": "感谢咨询，再见！欢迎下次再来！",
            "default": "这个问题我需要学习一下，您可以换个方式问问吗？"
        }
    
    def classify_intent(self, text: str) -> str:
        text = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        
        return "default"
    
    def respond(self, user_input: str) -> str:
        intent = self.classify_intent(user_input)
        return self.intent_responses[intent]


# ==================== 以下是启动代码 ====================

def main():
    # 创建意图分类器实例
    chatbot = IntentClassifier()
    
    print("=" * 50)
    print("智能客服系统已启动")
    print("支持的话题：打招呼、问名字、问天气、问价格、告别")
    print("输入 '退出' 或 'quit' 结束对话")
    print("=" * 50)
    
    while True:
        # 获取用户输入
        user_input = input("\n👤 用户: ").strip()
        
        # 检查是否退出
        if user_input.lower() in ['退出', 'quit', 'exit', 'q']:
            print("\n🤖 AI: 再见！感谢使用智能客服系统！")
            break
        
        # 获取并显示回复
        response = chatbot.respond(user_input)
        print(f"🤖 AI: {response}")
        
        # 显示识别出的意图（调试信息）
        intent = chatbot.classify_intent(user_input)
        print(f"   [识别意图: {intent}]")


# 如果直接运行这个文件，启动对话系统
if __name__ == "__main__":
    main()