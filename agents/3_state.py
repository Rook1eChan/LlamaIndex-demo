# 默认情况下， AgentWorkflow是无状态的，即agent没有记忆
# 可以使用Context类来维护状态

from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.workflow import JsonPickleSerializer, JsonSerializer
import os

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

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

ctx = Context(workflow)

async def main():
    response = await workflow.run(user_msg="Hi, my name is Laurie!",ctx=ctx)
    print(response)

    response2 = await workflow.run(user_msg="What's my name?",ctx=ctx)
    print(response2)

    response3 = await workflow.run(user_msg="What is 20+(2*4)?",ctx=ctx)
    print(response3)

    # 将ctx保存为字典
    ctx_dict = ctx.to_dict(serializer=JsonSerializer())

    # print(ctx_dict)
    
    # 加载ctx
    restored_ctx = Context.from_dict(
        workflow, ctx_dict, serializer=JsonSerializer()
    )

    response4 = await workflow.run(user_msg="What's my name?",ctx=restored_ctx)
    print(response4)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
