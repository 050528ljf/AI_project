from crewai import Agent, Task, Crew, Process, LLM  # 导入 CrewAI 的 LLM
from dotenv import load_dotenv
import os

load_dotenv()

# 使用 CrewAI 的 LLM 类，而非 LangChain 的 ChatOpenAI
llm = LLM(
    model="openai/GLM-5.3",   # 注意 openai/ 前缀，CrewAI 靠它路由到 OpenAI 兼容接口
    temperature=0.9,
    api_key=os.getenv("GLM_API_KEY"),
    base_url=os.getenv("GLM_base_url")  # 你的 GLM base_url
)

# 定义agent角色
researcher = Agent(
    role="市场研究员",
    goal="收集并整理关于目标主题的全面、准确市场信息",
    backstory="你是一个经验丰富的市场分析师，擅长从海量信息中提炼关键",
    llm=llm
)

analyst = Agent(
    role="数据分析师",
    goal="基于研究员提供的信息，进行深度分析并得出有价值的结论",
    backstory="你擅长用数据说话，能发现隐藏在信息背后的趋势和机会",
    llm=llm
)

writer = Agent(
    role="报告撰写专家",
    goal="将分析结论撰写成清晰、专业、有说服力的报告",
    backstory="你有丰富的商业写作经验，能让复杂的分析变得易于理解",
    llm=llm
)

# 定义Task（任务）
research_task = Task(
    description="调研中国新能源汽车市场的现状，包括主要品牌、市场份额、增长趋势",
    expected_output="一份包含 5 个关键数据点的市场调研报告，含数字和具体事实",
    agent=researcher
)

analysis_task = Task(
    description="基于调研报告，分析未来 3 年的机会与风险，给出投资评级建议",
    expected_output="SWOT 分析表格 + 投资评级（强烈推荐/推荐/中性/谨慎）+ 理由",
    agent=analyst,
    context=[research_task]    # 依赖调研任务的输出，即上下文
)

writing_task = Task(
    description="将调研和分析整合成一份 500 字的专业投资简报，格式清晰",
    expected_output="包含执行摘要、市场现状、机会风险、投资建议四个部分的简报",
    agent=writer,
    context=[research_task, analysis_task]
)

# 组建团队并执行
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,    # 顺序执行（也可改为 hierarchical）
    verbose=True
)

result = crew.kickoff()
print(result)
