import os
from urllib import response

from openai import OpenAI
from dotenv import load_dotenv

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


load_dotenv()

model = init_chat_model(model="deepseek-v4-flash",temperature=0)
agent = create_agent(model)

response = agent.invoke({
    "messages":[{"role":"user","content":"你是谁"}]
})
print(response)
