from langchain_community.chat_models import ChatZhipuAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModel


def llm_model_selector(model_name):
    if model_name == "glm4":
        ZHIPUAI_API_KEY = "your_key"
        llm = ChatZhipuAI(
            temperature=0.5,
            api_key=ZHIPUAI_API_KEY,
            model_name="glm-4",
        )
    elif model_name == "ollama3.1":
        llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0
        )
    elif model_name == "chatgpt":
        llm = ChatOpenAI(
            api_key="your_key",
            base_url="https://api",
            temperature=0,
            model="gpt-3.5-turbo"
        )
    elif model_name == "oglm4":
        llm = ChatOllama(
            model="glm4",
            temperature=0
        )
    elif model_name == "odeepseek":
        llm = ChatOllama(
            model="deepseek-llm",
            temperature=0
        )
    else:
        raise ValueError("LLMTypeError: unKnown LLM model. Please check the model name '"+model_name+"'")

    return llm


