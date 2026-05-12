# Day 34: Using Tools with LangChain
# Enable LLM to call Python functions

# Install required libraries
# pip install langchain langchain-openai openai

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType

# Initialize LLM
llm = ChatOpenAI(
    api_key="YOUR_API_KEY",   # Replace with your API key
    model="gpt-3.5-turbo"
)

# Create Python tools

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def square_number(n: int) -> int:
    """Return square of a number."""
    return n * n

# List of tools
tools = [multiply_numbers, square_number]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
response = agent.run(
    "What is the square of 8 and multiply it by 5?"
)

print("\n🔹 Final Answer:\n")
print(response)