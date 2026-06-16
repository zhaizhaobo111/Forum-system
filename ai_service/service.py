import operator
import os
from typing import TypedDict, Annotated, List
from enum import Enum

from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

# 导入AI发帖助手模块
from agent_write import execute_agent_write

load_dotenv()

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化大模型
model = ChatOpenAI(
    model="mimo-v2.5",
    openai_api_key=os.getenv("MIMO_API_KEY"),
    openai_api_base="https://token-plan-cn.xiaomimimo.com/v1",
    temperature=0
)

# ==================== 摘要功能 ====================

# 状态定义
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str


# 理解用户意图
def understand_intent(state: AgentState):
    message = state["messages"][-1]
    if "摘要" in message or "总结" in message:
        return {
            "messages": ["检测到摘要需求"],
            "next_step": "generate_summary"
        }
    elif "翻译" in message:
        return {
            "messages": ["检测到翻译需求"],
            "next_step": "translate"
        }
    else:
        return {
            "messages": ["一般对话"],
            "next_step": "general_chat"
        }


# 生成摘要（使用大模型）
def generate_summary(state: AgentState):
    content = state["messages"][-1]

    # 使用大模型生成摘要
    prompt = f"""请为以下内容生成一个简洁的摘要，要求：
    1. 摘要长度在50-100字之间
    2. 保留关键信息
    3. 语言简洁明了

    原始内容：
    {content}

    请直接输出摘要，不要添加任何前缀。"""

    try:
        response = model.invoke(prompt)
        summary = response.content
        return {"messages": [summary]}
    except Exception as e:
        # 如果大模型调用失败，回退到简单截取
        summary = content[:100] + "..." if len(content) > 100 else content
        return {
            "messages": [summary]
        }


# 翻译
def translate(state: AgentState):
    return {
        "messages": ["翻译功能待实现"]
    }


# 一般对话
def general_chat(state: AgentState):
    return {
        "messages": ["收到，有什么可以帮你的吗？"]
    }


# 创建工作流
builder = StateGraph(AgentState)

# 添加节点
builder.add_node("understand_intent", understand_intent)
builder.add_node("generate_summary", generate_summary)
builder.add_node("translate", translate)
builder.add_node("general_chat", general_chat)


# 路由函数
def router_intent(state: AgentState) -> str:
    user_intent = state["next_step"]
    if user_intent == "generate_summary":
        return "generate_summary"
    elif user_intent == "translate":
        return "translate"
    else:
        return "general_chat"


# 添加边
builder.add_edge(START, "understand_intent")

# 条件边
builder.add_conditional_edges(
    "understand_intent",
    router_intent,
    ["generate_summary", "translate", "general_chat"]
)

# 结束边
builder.add_edge("generate_summary", END)
builder.add_edge("translate", END)
builder.add_edge("general_chat", END)

# 编译工作流
app_graph = builder.compile()


class SummaryRequest(BaseModel):
    content: str


@app.post("/api/ai/summary")
async def generate_summary_api(request: SummaryRequest):
    # 直接调用大模型生成摘要，不经过意图判断
    prompt = f"""请为以下内容生成一个极简的摘要，要求：
    1. 摘要长度在10字以内（非常重要！）
    2. 提取最核心的关键词或短语
    3. 语言简洁明了

    原始内容：
    {request.content}

    请直接输出摘要，不要添加任何前缀。"""

    try:
        response = model.invoke(prompt)
        summary = response.content
        # 确保摘要不超过10字
        if len(summary) > 10:
            summary = summary[:10]
        return {"summary": summary}
    except Exception as e:
        # 如果大模型调用失败，回退到简单截取
        summary = request.content[:10] + "..." if len(request.content) > 10 else request.content
        return {"summary": summary}

@app.get("/api/ai/health")
async def health():
    return {"status": "ok"}


# ==================== AI发帖助手 ====================

# 请求模型
class WriteRequest(BaseModel):
    topic: str


# 响应模型
class WriteResponse(BaseModel):
    title: str
    summary: str
    content: str
    tags: List[str]


@app.post("/api/ai/agent/write", response_model=WriteResponse)
async def agent_write(request: WriteRequest):
    """AI发帖助手接口"""
    if not request.topic or request.topic.strip() == "":
        return {"error": "请输入文章主题"}

    # 调用AI发帖助手模块
    result = execute_agent_write(request.topic)

    if "error" in result:
        return result

    return WriteResponse(
        title=result.get("title", ""),
        summary=result.get("summary", ""),
        content=result.get("content", ""),
        tags=result.get("tags", [])
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


