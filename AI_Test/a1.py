# 规则匹配聊天机器人
responses = {
    "你好": "你好！我是AI助手。",
    "你叫什么名字": "我是Python AI助手。",
    "今天天气怎么样": "我无法获取实时天气，建议查看天气预报。",
    "再见": "再见！祝你有个美好的一天！"
}

def simple_chatbot():
    print("AI助手已启动！输入'再见'结束对话。")
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input == "再见":
            print("AI: " + responses["再见"])
            break
        
        # 查找匹配的回复
        reply = responses.get(user_input, "我不太明白，请换种方式问问看。")
        print("AI: " + reply)

if __name__ == "__main__":
    simple_chatbot()