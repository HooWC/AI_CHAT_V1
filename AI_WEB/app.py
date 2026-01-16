from flask import Flask, render_template, request, jsonify
from chatbot_logic import SuperChatbot

app = Flask(__name__)

# 全局初始化机器人，防止重复加载模型
bot = SuperChatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_query = data.get('message')
    history = data.get('history', [])
    
    # 调用你的机器人逻辑
    ai_response = bot.chat(user_query, history)
    
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)