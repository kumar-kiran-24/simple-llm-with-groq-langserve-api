from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROK_API_KEY")

model = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

generic_template = "Translate the following into {language}:"
prompt = ChatPromptTemplate.from_messages([
    ("system", generic_template),
    ("user", "{text}")
])

parser = StrOutputParser()
chain = prompt | model | parser

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Simple API using LangChain + Groq"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
