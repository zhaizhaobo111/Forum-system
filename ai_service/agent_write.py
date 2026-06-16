"""
AI发帖助手模块
使用LangGraph实现多Agent协同工作流
"""
import os
import json
import hashlib
from typing import TypedDict, List

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# Redis缓存（可选）
try:
    import redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False


# 初始化大模型
def get_model():
    return ChatOpenAI(
        model="mimo-v2.5",
        openai_api_key=os.getenv("MIMO_API_KEY"),
        openai_api_base="https://token-plan-cn.xiaomimimo.com/v1",
        temperature=0
    )


def get_cache_key(topic: str) -> str:
    """生成缓存key"""
    return f"ai:write:{hashlib.md5(topic.encode()).hexdigest()}"


def get_from_cache(topic: str):
    """从缓存获取结果"""
    if not REDIS_AVAILABLE:
        return None
    try:
        cache_key = get_cache_key(topic)
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except:
        pass
    return None


def save_to_cache(topic: str, result: dict, expire: int = 3600):
    """保存到缓存"""
    if not REDIS_AVAILABLE:
        return
    try:
        cache_key = get_cache_key(topic)
        redis_client.setex(cache_key, expire, json.dumps(result, ensure_ascii=False))
    except:
        pass


# 发帖助手状态定义
class WriteState(TypedDict):
    topic: str                    # 用户输入的主题
    plan: str                     # 文章计划
    outline: str                  # 文章大纲
    content: str                  # 文章正文
    title: str                    # 文章标题
    summary: str                  # 文章摘要
    tags: List[str]               # 文章标签
    current_step: str             # 当前步骤


# Planner Agent：规划文章结构
def planner_agent(state: WriteState):
    """规划Agent：根据用户主题，规划文章结构"""
    model = get_model()
    topic = state["topic"]

    prompt = f"""你是一个专业的文章规划师。请根据用户主题，快速规划文章结构。

用户主题：{topic}

请用一句话概括文章核心观点（不超过20字）。"""

    try:
        response = model.invoke(prompt)
        plan = response.content
        return {"plan": plan, "current_step": "outline"}
    except Exception as e:
        return {"plan": f"介绍{topic}的基本概念和应用", "current_step": "outline"}


# Outline Agent：生成大纲
def outline_agent(state: WriteState):
    """大纲Agent：根据规划，生成文章大纲"""
    model = get_model()
    topic = state["topic"]
    plan = state["plan"]

    prompt = f"""根据主题和规划，生成3个小标题。

主题：{topic}
规划：{plan}

格式：
一、xxx
二、xxx
三、xxx

直接输出，不要解释。"""

    try:
        response = model.invoke(prompt)
        outline = response.content
        return {"outline": outline, "current_step": "writer"}
    except Exception as e:
        return {"outline": f"一、{topic}简介\n二、核心内容\n三、总结", "current_step": "writer"}


# Writer Agent：撰写正文
def writer_agent(state: WriteState):
    """写作Agent：根据大纲，撰写文章正文（限制50字以内）"""
    model = get_model()
    topic = state["topic"]
    outline = state["outline"]

    prompt = f"""根据主题和大纲，写一段简短的文章开头（不超过50字）。

主题：{topic}
大纲：{outline}

直接输出内容，不要标题，不要解释。"""

    try:
        response = model.invoke(prompt)
        content = response.content
        # 确保内容不超过50字
        if len(content) > 50:
            content = content[:50]
        return {"content": content, "current_step": "title"}
    except Exception as e:
        return {"content": f"{topic}是一个重要的技术话题...", "current_step": "title"}


# Title Agent：生成标题
def title_agent(state: WriteState):
    """标题Agent：根据文章内容，生成吸引人的标题"""
    model = get_model()
    topic = state["topic"]

    prompt = f"""根据主题生成一个标题（不超过20字）。

主题：{topic}

直接输出标题，不要解释。"""

    try:
        response = model.invoke(prompt)
        title = response.content
        # 确保标题不超过20字
        if len(title) > 20:
            title = title[:20]
        return {"title": title, "current_step": "summary"}
    except Exception as e:
        return {"title": f"{topic}入门指南", "current_step": "summary"}


# Summary Agent：生成摘要
def summary_agent(state: WriteState):
    """摘要Agent：根据文章内容，生成摘要"""
    model = get_model()
    topic = state["topic"]

    prompt = f"""根据主题生成一句话摘要（不超过30字）。

主题：{topic}

直接输出摘要，不要解释。"""

    try:
        response = model.invoke(prompt)
        summary = response.content
        # 确保摘要不超过30字
        if len(summary) > 30:
            summary = summary[:30]
        return {"summary": summary, "current_step": "tags"}
    except Exception as e:
        return {"summary": f"介绍{topic}的基本概念和应用", "current_step": "tags"}


# Tags Agent：生成标签
def tags_agent(state: WriteState):
    """标签Agent：根据文章内容，生成标签"""
    model = get_model()
    topic = state["topic"]

    prompt = f"""根据主题生成3个标签，用逗号分隔。

主题：{topic}

直接输出标签，如：标签1,标签2,标签3"""

    try:
        response = model.invoke(prompt)
        tags_str = response.content
        # 解析标签
        tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
        # 最多保留3个标签
        tags = tags[:3]
        return {"tags": tags, "current_step": "done"}
    except Exception as e:
        return {"tags": [topic, "技术", "教程"], "current_step": "done"}


# 创建AI发帖助手工作流
def create_write_graph():
    """创建并编译AI发帖助手工作流"""
    builder = StateGraph(WriteState)

    # 添加节点
    builder.add_node("planner", planner_agent)
    builder.add_node("outline", outline_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("title", title_agent)
    builder.add_node("summary", summary_agent)
    builder.add_node("tags", tags_agent)

    # 添加边
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "outline")
    builder.add_edge("outline", "writer")
    builder.add_edge("writer", "title")
    builder.add_edge("title", "summary")
    builder.add_edge("summary", "tags")
    builder.add_edge("tags", END)

    # 编译工作流
    return builder.compile()


# 编译工作流（全局变量）
write_graph = create_write_graph()


def execute_agent_write(topic: str) -> dict:
    """
    执行AI发帖助手工作流

    Args:
        topic: 用户输入的文章主题

    Returns:
        包含title、summary、content、tags的字典
    """
    if not topic or topic.strip() == "":
        return {"error": "请输入文章主题"}

    # 检查缓存
    cached_result = get_from_cache(topic)
    if cached_result:
        return cached_result

    # 执行Agent工作流
    result = write_graph.invoke({
        "topic": topic,
        "plan": "",
        "outline": "",
        "content": "",
        "title": "",
        "summary": "",
        "tags": [],
        "current_step": "planner"
    })

    final_result = {
        "title": result.get("title", ""),
        "summary": result.get("summary", ""),
        "content": result.get("content", ""),
        "tags": result.get("tags", [])
    }

    # 保存到缓存
    save_to_cache(topic, final_result)

    return final_result
