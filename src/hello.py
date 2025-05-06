import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

template = "Write a story about a brave astronaut exploring {theme}."
prompt = PromptTemplate(input_variables=["theme"], template=template)

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

chain = prompt | llm
response = chain.invoke({"theme": "a distant planet"})
print(response.content)
