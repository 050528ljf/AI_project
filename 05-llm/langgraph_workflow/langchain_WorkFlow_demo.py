from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import operator
import os

# 定义状态结构（在节点间传递的数据）---------------------------------
class ResearchState(TypedDict):
    topic : str                 # 研究主题
    research_notes :str         # 研究笔记
    draft : str                 # 草稿
    review_feedback : str       # 审阅意见
    final_report : str          # 最终报告
    revision_count : Annotated[int, operator.add]
        # 修改次数（累加）

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("GLM_model_1"),
    base_url=os.getenv("GLM_base_url"),
    api_key=os.getenv("GLM_API_KEY"),
    temperature=0.9
)   

# 定义节点函数-------------------------------------------------------
def research_node(state:ResearchState) -> dict:
    """节点一：调研阶段"""
    response = llm.invoke(
        f"请对以下主题进行简要调研，列出5个关键要点：{state['topic']}"
    )
    return {"research_notes": response.content}

def write_node(state: ResearchState) -> dict:
    """节点二：撰写草稿"""
    prompt = f""""
    主题：{state["topic"]}
    调研笔记：{state['research_notes']}
    {'上次审阅意见：'+ state.get('review_feedback', '')
    if state.get('review_feedback')
    else ''}

    请根据以上内容撰写一篇300字分析报告草稿。
    """
    response = llm.invoke(prompt)
    return {"draft": response.content, "revision_count": 1}

def review_node(state: ResearchState) -> dict:
    """节点三：审阅草稿"""
    response = llm.invoke(
        f"审阅以下报告，如果质量达标，回复'APPROVED',否则给出具体修改意见：\n\n{state['draft']}"
    )
    return {"review_feedback": response.content}

def finalize_node(state: ResearchState) -> dict:
    """节点四：最终定稿"""
    return {"final_report": state["draft"]}

# 路由函数：决定审阅后走哪条路
def should_revise(state: ResearchState) -> str:
    if "APPROVED" in state["review_feedback"]:
        return "finalize"     # 质量过关后给review_feedback赋值为了APPROVED
    elif state["revision_count"] >= 3:
        return "finalize"     # 超过最大循环次数，防止死循环
    else:
        return "revise"       # 返回写作节点重写

# -构建工作流-------------------------------------------------------
workflow = StateGraph(ResearchState)

# 添加节点
workflow.add_node("research", research_node)
workflow.add_node("write", write_node)
workflow.add_node("review", review_node)
workflow.add_node("finalize", finalize_node)

# 设置入口
workflow.set_entry_point("research")

# 添加边（定义逻辑流转）
workflow.add_edge("research", "write") # 调研 -> 写作
workflow.add_edge("write"  , "review") # 写作 -> 审阅

# 条件边：审阅后根据结果选择路径
workflow.add_conditional_edges(
    "review",
    should_revise,
    {
        "revise"  :  "write",    # 需要修改 -> 回到写作
        "finalize":  "finalize"  # 通过审核 -> 最终定稿
    }
)

workflow.add_edge("finalize", END)

# 编译并运行
app = workflow.compile()

result = app.invoke(
    {
        "topic": "生成式AI对软件开发行业的影响",
        "revision_count": 0
    }
    )
print(result["final_report"])

# for chunk in llm.stream("写一篇100字左右的短文"):
#     print(chunk.content, end="", flush=True)