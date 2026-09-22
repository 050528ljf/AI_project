import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
llm = ChatOpenAI(
    model="deepseek-flash",
    temperature=0.9,
    api_key=os.getenv("DeepSeek_API_KEY"),
    base_url="https://api.deepseek.com"
)

parser = StrOutputParser()


# 1. 翻译
translate_prompt = ChatPromptTemplate.from_template(
    "将以下英文翻译成中文，保持原意：\n\n{article}"
)

translate_chain = translate_prompt | llm | parser


# 2. 摘要
summarize_prompt = ChatPromptTemplate.from_template(
    "请将以下文章提炼三个要点，每点只用一句话：\n\n{translate}"
)

summarize_chain = summarize_prompt | llm | parser


# 3. 标题
title_prompt = ChatPromptTemplate.from_template(
    "请为以下文章生成一个简短的标题：\n\n{summary}"
)

title_chain = title_prompt | llm | parser


chain = (
    {"translate": translate_chain}
    | RunnablePassthrough.assign(summary=summarize_chain)
    | RunnablePassthrough.assign(title=title_chain)
)


article = """
Artificial intelligence is transforming how we work and live.
From automating repetitive tasks to assisting in creative work,
AI tools are becoming indispensable in modern workflows...
"""


result = chain.invoke({"article": article})

print("翻译：")
print(result["translate"])

print("\n摘要：")
print(result["summary"])

print("\n标题：")
print(result["title"])

# from dotenv import load_dotenv
# import os

# from langchain_openai import ChatOpenAI

# load_dotenv()

# api_key = os.getenv("DEEPSEEK_API_KEY")

# print("API Key 是否读取成功：", api_key is not None)

# llm = ChatOpenAI(
#     model="deepseek-flash",
#     temperature=0.9,
#     api_key=api_key,
#     base_url="https://api.deepseek.com",
# )