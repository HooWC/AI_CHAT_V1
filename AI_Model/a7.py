import torch
import json
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

# 初始化 Rich 控制台
console = Console()

class SuperChatbot:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_history_len = 10  # 限制保留最近的 N 轮对话，防止爆显存
        
        console.print(f"[bold green]正在加载引擎: {self.model_name} (设备: {self.device})...[/bold green]")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        except Exception as e:
            console.print(f"[bold red]模型加载失败: {e}[/bold red]")
            console.print("请检查网络或显存。建议先使用较小的模型如 'Qwen/Qwen2.5-0.5B-Instruct'")
            exit()

        # 默认生成参数
        self.gen_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        
        self.mode = "assistant"
        self.messages = []
        self.reset_history()

    def reset_history(self):
        """重置对话"""
        if self.mode == "novel":
            sys_prompt = "你是一位获得诺贝尔文学奖的小说家。请根据用户的要求创作情节跌宕起伏、描写细腻、人物性格鲜明的小说。请使用生动的修辞手法。"
            self.gen_kwargs["temperature"] = 0.95  # 写小说更发散
        else:
            sys_prompt = "你是一个精通编程、科学与人文的 AI 助手。回答要条理清晰，准确无误。可以使用 Markdown 格式优化排版。"
            self.gen_kwargs["temperature"] = 0.7   # 问答更严谨
        
        self.messages = [{"role": "system", "content": sys_prompt}]
        console.print(f"[dim]已重置上下文，当前模式: {self.mode}[/dim]")

    def trim_history(self):
        """滑动窗口：当对话过长时，移除最早的对话（保留 System Prompt）"""
        # System prompt 是 index 0，所以我们检查长度是否超过 limit + 1
        if len(self.messages) > (self.max_history_len * 2) + 1:
            # 保留 system prompt (index 0)，切掉中间旧的，保留最近的
            removed_count = len(self.messages) - ((self.max_history_len * 2) + 1)
            self.messages = [self.messages[0]] + self.messages[-(self.max_history_len * 2):]
            console.print(f"[dim yellow]⚠️为了保持思维清晰，遗忘了 {removed_count} 条旧消息...[/dim yellow]")

    def save_chat(self, filename="chat_history.json"):
        """保存对话到本地"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            console.print(f"[green]✅ 对话已保存至 {filename}[/green]")
        except Exception as e:
            console.print(f"[red]❌ 保存失败: {e}[/red]")

    def load_chat(self, filename="chat_history.json"):
        """加载本地对话"""
        if not os.path.exists(filename):
            console.print(f"[red]❌ 找不到文件: {filename}[/red]")
            return
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.messages = json.load(f)
            console.print(f"[green]✅ 已加载历史对话 ({len(self.messages)} 条消息)[/green]")
        except Exception as e:
            console.print(f"[red]❌ 加载失败: {e}[/red]")

    def chat(self, user_input):
        self.trim_history() # 检查是否需要遗忘旧消息
        self.messages.append({"role": "user", "content": user_input})
        
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 打印 AI 思考中的提示
        console.print(Text("🤖 AI 正在思考...", style="bold cyan"), end="\r")

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 换行开始输出
        print("\n" + "-"*30) 
        
        # 调整显存
        if self.device == "cuda":
            torch.cuda.empty_cache()

        generated_ids = self.model.generate(
            **model_inputs,
            streamer=streamer,
            **self.gen_kwargs
        )
        print("-" * 30 + "\n")

        # 保存回复
        response = self.tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": response})

def print_menu():
    menu_text = """
    [bold cyan]🎮 指令菜单:[/bold cyan]
    [green]/novel[/green] - 切换小说模式 (高创造力)
    [green]/chat[/green]  - 切换助手模式 (高严谨度)
    [green]/save[/green]  - 保存当前对话
    [green]/load[/green]  - 读取历史对话
    [green]/temp X[/green]- 设置温度 (0.1-1.0)，例如 /temp 0.9
    [green]/clear[/green] - 清空记忆
    [red]/exit[/red]  - 退出程序
    """
    console.print(Panel(menu_text, title="SuperChatbot Pro", subtitle="基于 Qwen2.5", border_style="blue"))

def main():
    # 建议根据显存大小修改此处，8G显存推荐 1.5B 或 3B
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" 
    
    bot = SuperChatbot(model_name)
    print_menu()
    
    while True:
        try:
            mode_icon = "📝" if bot.mode=='novel' else "🧠"
            user_input = Prompt.ask(f"\n[bold]{mode_icon} 你[/bold]")
            
            if not user_input.strip(): continue

            # 指令处理
            if user_input.startswith("/"):
                cmd_parts = user_input.lower().split()
                cmd = cmd_parts[0]
                
                if cmd == '/exit': break
                elif cmd == '/clear': bot.reset_history()
                elif cmd == '/novel': 
                    bot.mode = "novel"
                    bot.reset_history()
                elif cmd == '/chat': 
                    bot.mode = "assistant"
                    bot.reset_history()
                elif cmd == '/save': bot.save_chat()
                elif cmd == '/load': bot.load_chat()
                elif cmd == '/temp':
                    if len(cmd_parts) > 1:
                        try:
                            val = float(cmd_parts[1])
                            bot.gen_kwargs["temperature"] = max(0.1, min(1.5, val))
                            console.print(f"[dim]🌡️ 温度已设置为: {bot.gen_kwargs['temperature']}[/dim]")
                        except: console.print("[red]❌ 请输入数字，例如 /temp 0.8[/red]")
                    else:
                        console.print(f"[dim]当前温度: {bot.gen_kwargs['temperature']}[/dim]")
                else:
                    console.print("[red]❌ 未知指令[/red]")
                continue

            # 正常对话
            bot.chat(user_input)

        except KeyboardInterrupt:
            console.print("\n[yellow]检测到中断，正在退出...[/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]发生错误: {e}[/bold red]")

if __name__ == "__main__":
    main()