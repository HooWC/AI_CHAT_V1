from typing import Dict, List, Optional, Tuple
import re
import json
from datetime import datetime
import random

class MultiTurnChatbot:
    def __init__(self):
        # 意图和对应的模式
        self.intent_patterns = {
            "greeting": [r"你好", r"您好", r"hi", r"hello"],
            "ask_name": [r"叫什么", r"名字", r"你是谁"],
            "ask_weather": [r"天气", r"下雨", r"晴天", r"温度", r"天气预报"],
            "ask_price": [r"价格", r"多少钱", r"价钱", r"cost"],
            "goodbye": [r"再见", r"拜拜", r"下次聊"]
        }
        
        # 基本回复模板
        self.base_responses = {
            "greeting": "您好！我是智能天气助手，可以帮您查询全球天气！",
            "ask_name": "我是天气小助手，随时为您提供天气查询服务。",
            "ask_weather": "请告诉我您想查询哪个城市的天气？",
            "ask_price": "我的天气服务是完全免费的！",
            "goodbye": "感谢使用，再见！",
            "default": "这个问题我需要学习一下，您可以换个方式问问吗？"
        }
        
        # 对话状态：记录当前正在进行的任务
        self.conversation_state = {
            "current_intent": None,      # 当前意图
            "waiting_for": None,         # 等待什么信息
            "collected_info": {},        # 已收集的信息
            "last_intent": None,         # 上一个意图
            "user_name": None,           # 用户名（可扩展）
            "conversation_history": []   # 对话历史
        }
        
        # 天气数据库（模拟）
        self.weather_data = {
            "马来西亚": {"temp": "28-32°C", "condition": "多云转雷阵雨", "humidity": "85%"},
            "北京": {"temp": "5-12°C", "condition": "晴", "humidity": "45%"},
            "上海": {"temp": "10-18°C", "condition": "阴转小雨", "humidity": "75%"},
            "纽约": {"temp": "8-15°C", "condition": "多云", "humidity": "60%"},
            "东京": {"temp": "12-20°C", "condition": "晴", "humidity": "55%"}
        }
        
        # 实体识别关键词
        self.location_keywords = ["马来西亚", "北京", "上海", "纽约", "东京", "伦敦", "巴黎"]
    
    def classify_intent(self, text: str) -> str:
        """识别用户意图"""
        text = text.lower()
        
        # 首先检查是否在回答之前的问题
        if self.conversation_state["waiting_for"] == "city":
            if self.extract_location(text):
                return "provide_city"
        
        # 正常意图识别
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        
        return "default"
    
    def extract_location(self, text: str) -> Optional[str]:
        """从文本中提取地点"""
        text_lower = text.lower()
        
        # 简单关键词匹配
        for location in self.location_keywords:
            if location.lower() in text_lower:
                return location
        
        # 如果包含"天气在"或"的天气"
        if "天气" in text:
            # 提取"天气"前面的内容作为可能的地点
            parts = text.split("天气")
            if len(parts) > 0 and parts[0].strip():
                potential_loc = parts[0].strip()
                if len(potential_loc) < 10:  # 避免太长的内容
                    return potential_loc
        
        return None
    
    def get_weather_info(self, city: str) -> str:
        """获取天气信息"""
        if city in self.weather_data:
            data = self.weather_data[city]
            return f"{city}的天气：{data['condition']}，温度{data['temp']}，湿度{data['humidity']}"
        else:
            return f"抱歉，我还没有{city}的天气数据。目前支持查询：{', '.join(list(self.weather_data.keys())[:5])}"
    
    def handle_weather_flow(self, user_input: str) -> Tuple[str, bool]:
        """处理天气查询的多轮对话"""
        # 检查是否已经有城市信息
        if "city" in self.conversation_state["collected_info"]:
            city = self.conversation_state["collected_info"]["city"]
            weather_info = self.get_weather_info(city)
            
            # 重置状态
            self.reset_conversation_state()
            
            # 添加后续问题
            follow_up = random.choice([
                "\n还需要查询其他城市的天气吗？",
                "\n还有其他天气问题吗？",
                "\n想了解其他城市的天气吗？"
            ])
            
            return weather_info + follow_up, False
        
        # 如果没有城市信息，询问城市
        else:
            location = self.extract_location(user_input)
            if location:
                self.conversation_state["collected_info"]["city"] = location
                weather_info = self.get_weather_info(location)
                
                # 重置状态
                self.reset_conversation_state()
                
                follow_up = random.choice([
                    "\n还想知道其他城市的天气吗？",
                    "\n还有什么可以帮您的？"
                ])
                
                return weather_info + follow_up, False
            else:
                # 需要用户提供城市
                self.conversation_state["waiting_for"] = "city"
                return self.base_responses["ask_weather"], True
    
    def reset_conversation_state(self):
        """重置对话状态（一轮对话结束）"""
        self.conversation_state["current_intent"] = None
        self.conversation_state["waiting_for"] = None
        self.conversation_state["collected_info"] = {}
    
    def save_conversation(self, user_input: str, ai_response: str):
        """保存对话历史"""
        self.conversation_state["conversation_history"].append({
            "user": user_input,
            "ai": ai_response,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 保持最近10轮对话
        if len(self.conversation_state["conversation_history"]) > 10:
            self.conversation_state["conversation_history"] = self.conversation_state["conversation_history"][-10:]
    
    def respond(self, user_input: str) -> str:
        """生成回复（核心方法）"""
        # 识别意图
        intent = self.classify_intent(user_input)
        
        # 更新对话状态
        self.conversation_state["last_intent"] = self.conversation_state["current_intent"]
        self.conversation_state["current_intent"] = intent
        
        # 保存用户输入
        self.save_conversation(user_input, "")
        
        # 根据意图和状态生成回复
        response = ""
        
        # 特殊处理：用户提供了城市信息
        if intent == "provide_city":
            location = self.extract_location(user_input)
            if location:
                self.conversation_state["collected_info"]["city"] = location
                weather_info = self.get_weather_info(location)
                response = weather_info
                
                # 添加后续问题
                follow_up = random.choice([
                    "\n还需要查询其他天气信息吗？",
                    "\n还有其他问题吗？",
                    "\n想了解其他城市的天气吗？"
                ])
                response += follow_up
                
                # 重置状态
                self.reset_conversation_state()
            else:
                response = "抱歉，我没听清楚是哪个城市，请再说一遍城市名称。"
        
        # 查询天气（开始多轮对话）
        elif intent == "ask_weather":
            response, is_waiting = self.handle_weather_flow(user_input)
        
        # 其他意图
        elif intent in self.base_responses:
            response = self.base_responses[intent]
            
            # 如果是问天气，设置等待状态
            if intent == "ask_weather":
                self.conversation_state["waiting_for"] = "city"
        
        else:
            # 检查是否在等待信息
            if self.conversation_state["waiting_for"] == "city":
                location = self.extract_location(user_input)
                if location:
                    weather_info = self.get_weather_info(location)
                    response = weather_info
                    self.reset_conversation_state()
                    
                    # 添加后续问题
                    follow_up = random.choice([
                        "\n还想知道其他城市的天气吗？",
                        "\n还有什么可以帮您的？"
                    ])
                    response += follow_up
                else:
                    response = "您说的是哪个城市呢？请告诉我具体的城市名称。"
            else:
                response = self.base_responses["default"]
        
        # 保存AI回复
        self.save_conversation("", response)
        
        return response
    
    def show_conversation_history(self):
        """显示对话历史"""
        print("\n" + "="*60)
        print("对话历史记录：")
        print("="*60)
        for i, turn in enumerate(self.conversation_state["conversation_history"], 1):
            if turn["user"]:
                print(f"[{turn['time']}]")
                print(f"👤 您: {turn['user']}")
            if turn["ai"]:
                print(f"🤖 AI: {turn['ai']}")
                print("-"*40)
    
    def get_conversation_status(self):
        """获取当前对话状态"""
        status = f"""
当前对话状态：
- 当前意图: {self.conversation_state['current_intent']}
- 等待信息: {self.conversation_state['waiting_for']}
- 已收集: {json.dumps(self.conversation_state['collected_info'], ensure_ascii=False)}
- 历史记录数: {len(self.conversation_state['conversation_history'])}
        """
        return status


# ==================== 主程序 ====================

def main():
    chatbot = MultiTurnChatbot()
    
    print("=" * 60)
    print("🤖 多轮对话天气助手")
    print("=" * 60)
    print("功能：")
    print("1. 支持连续对话（比如：问天气 -> 告诉城市 -> 得到结果）")
    print("2. 支持查询：马来西亚、北京、上海、纽约、东京")
    print("3. 输入 '历史' 查看对话记录")
    print("4. 输入 '状态' 查看对话状态")
    print("5. 输入 '退出' 结束对话")
    print("=" * 60)
    
    print("\n🤖 AI: 您好！我是天气助手，可以帮您查询全球天气信息！")
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if not user_input:
                continue
            
            # 特殊命令
            if user_input.lower() in ['退出', 'quit', 'exit', 'q']:
                print("\n🤖 AI: 再见！欢迎下次查询天气！")
                chatbot.show_conversation_history()
                break
            
            elif user_input == '历史':
                chatbot.show_conversation_history()
                continue
            
            elif user_input == '状态':
                print(chatbot.get_conversation_status())
                continue
            
            # 正常对话
            response = chatbot.respond(user_input)
            print(f"🤖 AI: {response}")
            
        except KeyboardInterrupt:
            print("\n🤖 AI: 对话已结束")
            break
        except Exception as e:
            print(f"🤖 AI: 抱歉，出错了: {e}")

if __name__ == "__main__":
    main()