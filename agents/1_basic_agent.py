# 一个基础的agent：LLM+Tools

# pip install llama-index-core llama-index-llms-openai llama-index-llms-openai-like python-dotenv

from dotenv import load_dotenv

load_dotenv()  # 用于从.env文件中加载环境变量到当前的运行环境中

# from llama_index.llms.openai import OpenAI

# 如果使用国内模型，需要用不同的库。比如百炼使用的是OpenAILike
# https://docs.llamaindex.org.cn/en/stable/api_reference/llms/openai_like/
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import AgentWorkflow
import os

# 定义工具
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

# llm = OpenAI(model="gpt-4o-mini")

llm = OpenAILike(
    model="qwen-flash",
    api_base=os.getenv("BAILIAN_API_BASE"),
    api_key=os.getenv("BAILIAN_API_KEY"),
    is_chat_model=True,  # 必须设置为True
    is_function_calling_model=True,  # 启用工具调用功能
    temperature=0.1,
    max_tokens=2048,  # 要生成的最大 Token 数
    context_window=30720,  # 上下文窗口
)

workflow = AgentWorkflow.from_tools_or_functions(
    [multiply, add],
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools."
)

async def main():
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
