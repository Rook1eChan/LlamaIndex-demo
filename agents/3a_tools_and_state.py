# 使用tool来修改context

from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.workflow import JsonPickleSerializer, JsonSerializer
import os

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

async def set_name(ctx: Context, name: str) -> str:
    """set user's name"""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["name"] = name

    return f"Name set to {name}"

workflow = AgentWorkflow.from_tools_or_functions(
    [set_name],
    llm=llm,
    system_prompt="You are a helpful assistant that can set a name.",
    initial_state={"name": "unset"},  # 初始化ctx中的元数据，比如"sex": "girl"
    # 如果不初始化name，大模型会说I don't have the ability to set a name right now，这是为什么捏   
)

async def main():
    ctx = Context(workflow)

    response = await workflow.run(user_msg="What's my name?", ctx=ctx)
    print(str(response))
    
    response2 = await workflow.run(user_msg="My name is Laurie", ctx=ctx)
    print(str(response2))

    ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    print(ctx_dict)
    
    state = await ctx.store.get("state")
    print("Name as stored in state: ", state["name"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
