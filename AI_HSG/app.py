import json
import os
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from hsg_engine import HSGEngine

app = Flask(__name__)
hsg_ai = HSGEngine()
HISTORY_FILE = "chat_history.json"

# 加载历史记录
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# 保存历史记录
def save_history(history_data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_sessions', methods=['GET'])
def get_sessions():
    history = load_history()
    # 返回给前端：ID 和 标题
    sessions = [{"id": k, "title": v['title']} for k, v in history.items()]
    return jsonify(sessions[::-1]) # 最新的在上面

@app.route('/get_messages/<session_id>', methods=['GET'])
def get_messages(session_id):
    history = load_history()
    return jsonify(history.get(session_id, {}).get("messages", []))

@app.route('/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    history = load_history()
    if session_id in history:
        del history[session_id]
        save_history(history)
        return jsonify({"success": True, "message": "对话已删除"})
    return jsonify({"success": False, "message": "对话不存在"}), 404

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    session_id = data.get('session_id')
    
    history = load_history()
    
    # 如果是新会话
    if not session_id or session_id not in history:
        import time
        session_id = str(int(time.time()))
        history[session_id] = {
            "title": user_input[:15] + "...",
            "messages": []
        }
    
    history[session_id]["messages"].append({"role": "user", "content": user_input})
    save_history(history)

    def generate():
        full_reply = ""
        # 传入历史消息给 AI 引擎以维持上下文
        context = history[session_id]["messages"][-5:] 
        for token in hsg_ai.chat_stream(user_input, context[:-1]):
            full_reply += token
            yield f"data: {json.dumps({'token': token, 'session_id': session_id})}\n\n"
        
        # 保存 AI 的回答
        new_history = load_history()
        new_history[session_id]["messages"].append({"role": "assistant", "content": full_reply})
        save_history(new_history)

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    # 使用 waitress 作为生产级 WSGI 服务器
    try:
        from waitress import serve
        print("🚀 启动生产服务器 (Waitress)...")
        print("📍 访问地址: http://localhost:5000")
        serve(app, host='0.0.0.0', port=5000, threads=4)
    except ImportError:
        print("⚠️ Waitress 未安装，使用开发服务器")
        print("💡 建议安装: pip install waitress")
        app.run(debug=True, port=5000)