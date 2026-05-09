# Day 31: LangChain Basics
# Prompt → LLM → Output Chain

# Install required libraries
# pip install langchain openai langchain-openai

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(
    api_key="YOUR_API_KEY",   # Replace with your API key
    model="gpt-3.5-turbo"
)

# Create Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words."
)

# Create Chain
chain = prompt | llm

# Run chain
response = chain.invoke({
    "topic": "Machine Learning"
})

# Print output
print("🔹 AI Response:\n")
print(response.content)