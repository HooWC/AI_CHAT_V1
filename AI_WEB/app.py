from flask import Flask, render_template, request, Response, stream_with_context
from chatbot_logic import SuperChatbot
import json

app = Flask(__name__)

# 全局初始化 AI 引擎 (0.5B 版本)
print("正在初始化 AI，请稍候...")
bot = SuperChatbot()

@app.route('/')
def index():
    # 确保你的 HTML 文件放在 templates 文件夹下
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('message', '')
    history = data.get('history', [])

    def generate():
        try:
            # 调用 chatbot_logic 中的流式生成
            for token in bot.chat_stream(user_query, history):
                # 按照 SSE 协议格式发送数据
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            print(f"生成出错: {e}")
            yield f"data: {json.dumps({'token': '[发生错误]'})}\n\n"
            
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    # host='0.0.0.0' 允许局域网访问
    # debug=False 非常关键！可以节省一半的内存占用
    app.run(host='0.0.0.0', port=5000, debug=False)