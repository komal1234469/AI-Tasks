# Day 32: Summarization Chain using LangChain

# Install required libraries
# pip install langchain langchain-openai openai

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Initialize OpenAI model
llm = ChatOpenAI(
    api_key="YOUR_API_KEY",   # Replace with your API key
    model="gpt-3.5-turbo"
)

# Sample document
document = """
Artificial Intelligence is transforming industries worldwide.
AI systems are used in healthcare, education, finance,
automation, and customer support. Machine learning enables
systems to learn patterns from data and improve predictions
without explicit programming.
"""

# Create summarization prompt
prompt = ChatPromptTemplate.from_template(
    "Summarize the following document in 3 simple sentences:\n\n{document}"
)

# Output parser
parser = StrOutputParser()

# Build chain
chain = prompt | llm | parser

# Run chain
result = chain.invoke({
    "document": document
})

# Print summary
print("🔹 Document Summary:\n")
print(result)