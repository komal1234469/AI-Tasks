# Day 33: Chatbot with Memory using LangChain

# Install required libraries
# pip install langchain langchain-openai openai

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize LLM
llm = ChatOpenAI(
    api_key="YOUR_API_KEY",   # Replace with your API key
    model="gpt-3.5-turbo"
)

# Create memory
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Chat interactions
response1 = conversation.predict(
    input="Hi, my name is Komal."
)

print("Bot:", response1)

response2 = conversation.predict(
    input="What is my name?"
)

print("\nBot:", response2)

response3 = conversation.predict(
    input="Explain Machine Learning in simple words."
)

print("\nBot:", response3)