import operator
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

load_dotenv()
app=FastAPI()
# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 状态定义
class AgentState(TypedDict):
    messages:Annotated[list,operator.add]
    next_step:str
# 理解用户意图
def understand_intent(state:AgentState):
    message=state["messages"][-1]
    if "摘要"in message or "总结" in message:
        return{
            "messages":["检测到摘要需求"],
            "next_step":"generate_summary"
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
# 生成摘要
def generate_summary(state:AgentState):
    message=state["messages"][-1]
    summary=message[:100]+"..." if len(message)>100 else message
    return {
        "messages":[f"摘要:{summary}"]
    }

# 翻译
def translate(state: AgentState):
      # 这里可以调用大模型API进行翻译
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

# 添加边
builder.add_edge(START, "understand_intent")
# 路由函数
def router_intent(state: AgentState) -> str:
    user_intent = state["next_step"]
    if user_intent == "generate_summary":
        return "generate_summary"
    elif user_intent == "translate":
        return "translate"
    else:
        return "general_chat"
# 条件边
builder.add_conditional_edges(
    "understand_intent",
    router_intent,
    ["generate_summary", "translate", "general_chat"]
)
builder.add_edge("generate_summary", END)
builder.add_edge("translate", END)
builder.add_edge("general_chat", END)

# 编译工作流
app_graph = builder.compile()

class SummaryRequest(BaseModel):
      content: str

@app.post("/api/ai/summary")
async def generate_summary(request: SummaryRequest):
      result = app_graph.invoke({"messages": [request.content], "next_step": ""})
      return {"summary": result["messages"][-1]}

@app.get("/api/ai/health")
async def health():
      return {"status": "ok"}

if __name__ == "__main__":
      import uvicorn
      uvicorn.run(app, host="0.0.0.0", port=8000)
