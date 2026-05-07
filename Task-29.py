# Day 29: Calling OpenAI API and Generating Responses

# Install OpenAI library
# pip install openai

from openai import OpenAI

# Initialize client
client = OpenAI(
    api_key="YOUR_API_KEY"   # Replace with your OpenAI API key
)

# User prompt
prompt = "Explain Machine Learning in simple words."

# Generate response
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Print AI response
print("Prompt:", prompt)
print("\nAI Response:\n")
print(response.choices[0].message.content)