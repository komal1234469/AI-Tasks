# Day 35: Conversational Chatbot with Memory

# Install required libraries
# pip install langchain langchain-openai openai

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize OpenAI model
llm = ChatOpenAI(
    api_key="YOUR_API_KEY",   # Replace with your API key
    model="gpt-3.5-turbo"
)

# Create memory
memory = ConversationBufferMemory()

# Create conversation chain
chatbot = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

print("🤖 AI Chatbot Started (type 'exit' to stop)\n")

# Chat loop
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye 👋")
        break

    response = chatbot.predict(input=user_input)

    print("Bot:", response)