from langchain_openai import ChatOpenAI
# from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents import create_agent
from langchain_core.tools import tool
# from langsmith import hub
import requests, datetime, os
from dotenv import load_dotenv

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市当前天气"""
    # 实际项目中替换为真实天气 API
    mock_data = {
        "北京": "晴天，气温 22°C，微风",
        "上海": "多云，气温 26°C，湿度 75%",
        "广州": "小雨，气温 30°C，建议带伞",
    }
    return mock_data.get(city, f"暂无{city}的天气数据")

@tool
def search_web(query: str) -> str:
    """在网上搜索相关信息，返回相关信息内容摘要"""
    # 实际项目中接入Tavily / Serper API
    return f"搜索'{query}'的结果：这是一个模拟的搜索结果"

@tool 
def calculate(expression: str) -> str:
    """计算数学表达式，例如'2 + 3 * 4'"""
    try:
        result = eval(expression, {"__builtins__":{}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误:{e}"

@tool
def get_date() -> str:
    """获取今天日期"""
    return datetime.date.today().strftime("%Y年%m月%d日")

# 创建agent
tools = [get_weather, search_web, calculate, get_date]
# llm   = ChatOpenAI(model="gpt-4o", temperature=0)
load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("GLM_model_1"),
    base_url=os.getenv("GLM_base_url"),
    api_key=os.getenv("GLM_API_KEY"),
    temperature=0.9
)   
# # 使用Langchain Hub的标准ReAct prompt,
# # 用在创建 agent 中，也就是说ReAct模式主要由prompt
# # 实现，而不是代码逻辑
# prompt = hub.pull("hwchase17/react")

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant with access to tools. "
                  "Use the tools to answer the user's questions."
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "今天是几号？北京天气怎么样？如果出门步行 5km 消耗约 300 卡路里，"
             "跑步同样距离消耗大约是步行的 1.6 倍，请计算跑步消耗的卡路里。"}
        ]
    }
)
print(result["messages"][-1].content)
# Agent 会自动决策：先调用 get_date → get_weather("北京") → calculate("300*1.6")
# 最后综合所有信息给出完整回答