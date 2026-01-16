import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# === 1. 页面配置 ===
st.set_page_config(page_title="SuperChatbot Web", page_icon="🤖", layout="wide")
st.title("🤖 SuperChatbot Pro (Web版)")

# === 2. 加载模型 (使用缓存，只加载一次) ===
@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" # 显存够可换 1.5B
    status_text = st.empty()
    status_text.info(f"正在加载模型 {model_name}... 请稍候")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    status_text.empty() # 加载完清空提示
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# === 3. 侧边栏：设置面板 ===
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 模式选择
    mode = st.radio("对话模式", ["助手模式", "小说模式"])
    
    # 动态参数
    temperature = st.slider("温度 (创造力)", 0.1, 1.5, 0.7 if mode == "助手模式" else 0.95)
    max_tokens = st.slider("最大回复长度", 128, 2048, 1024)
    
    # 清空按钮
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# === 4. 初始化对话历史 (Session State) ===
# Streamlit 每次交互都会重跑代码，所以要存在 session_state 里
if "messages" not in st.session_state:
    st.session_state.messages = []

# 根据模式设置 System Prompt
if not st.session_state.messages:
    if mode == "小说模式":
        sys_prompt = "你是一位获得诺贝尔文学奖的小说家。请根据用户的要求创作情节跌宕起伏、描写细腻的小说。"
    else:
        sys_prompt = "你是一个通晓百科、乐于助人的中文 AI 助手。"
    st.session_state.messages.append({"role": "system", "content": sys_prompt})

# === 5. 渲染历史对话 ===
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])

# === 6. 处理用户输入 ===
if user_input := st.chat_input("输入你的问题..."):
    # 显示用户消息
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # === 生成回复 (流式) ===
    with st.chat_message("assistant"):
        # 构建输入
        text = tokenizer.apply_chat_template(
            st.session_state.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 设置流式输出器 (这是 Web 版流式的关键)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 参数设置
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1
        )

        # 在新线程中运行生成，防止阻塞主线程
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # st.write_stream 会自动从 streamer 读取 tokens 并打字机显示
        response = st.write_stream(streamer)
        
        # 将完整的回复存入历史
        st.session_state.messages.append({"role": "assistant", "content": response})