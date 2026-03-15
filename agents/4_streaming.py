# 将agent的处理过程流式输出

# pip install llama-index-tools-tavily-research

from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.tools.tavily_research import TavilyToolSpec
import os
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)

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

tavily_tool = TavilyToolSpec( api_key=os.getenv("TAVILY_API_KEY") )  # 一个网络搜索工具

workflow = AgentWorkflow.from_tools_or_functions(
    tavily_tool.to_tool_list(),
    llm=llm,
    system_prompt="You're a helpful assistant that can search the web for information."
)

async def main():
    handler = workflow.run(user_msg="What did Energy Secretary Chris Wright say about the war with Iran on Sunday?")

    # 追踪agent的动态
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)
            print("----------\n\n")
        elif isinstance(event, AgentInput):
            print("Agent input: ", event.input)  # the current input messages
            print("Agent name:", event.current_agent_name)  # the current agent name
            print("----------\n\n")
        elif isinstance(event, AgentOutput):
            print("Agent output: ", event.response)  # the current full response
            print("Tool calls made: ", event.tool_calls)  # the selected tool calls, if any
            print("Raw LLM response: ", event.raw)  # the raw llm api response
            print("----------\n\n")    
        elif isinstance(event, ToolCallResult):
            print("Tool called: ", event.tool_name)  # the tool name
            print("Arguments to the tool: ", event.tool_kwargs)  # the tool kwargs
            print("Tool output: ", event.tool_output)  # the tool output            
            print("----------\n\n")
    # print final output
    print(str(await handler))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
