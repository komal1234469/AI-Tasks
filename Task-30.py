# Day 30: Prompt Engineering - Summarization & Q&A

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key="YOUR_API_KEY"   # Replace with your API key
)

# Sample text
text = """
Artificial Intelligence is transforming industries by enabling machines
to perform tasks that normally require human intelligence. It is widely
used in healthcare, finance, education, and automation systems.
"""

# -------------------------------
# 1. Summarization Prompt
# -------------------------------

summary_prompt = f"""
Summarize the following text in 2 simple sentences:

{text}
"""

summary_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": summary_prompt}
    ]
)

print("🔹 Summary:\n")
print(summary_response.choices[0].message.content)

# -------------------------------
# 2. Question Answering Prompt
# -------------------------------

qa_prompt = f"""
Answer the following question based on the text.

Text:
{text}

Question:
Which industries are using Artificial Intelligence?
"""

qa_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": qa_prompt}
    ]
)

print("\n🔹 Question Answering:\n")
print(qa_response.choices[0].message.content)