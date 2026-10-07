from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b"
)

print("1. Normal mode")
print("2. Expert mode")
print("3. Quick mode")

choice = int(input("Enter your choice: "))

if choice == 1:
    mode = "You are a helpful AI assistant. Give clear, natural, and easy-to-understand answers."

elif choice == 2:
    mode = """You are an expert AI assistant.
    Give accurate, technical, and detailed answers using professional terminology and strong reasoning.

    Structure your answers clearly:
    - Use headings and subheadings.
    - Use bullet points for key points.
    - Use numbered lists for step-by-step instructions.
    - Use tables when comparing information or presenting structured data.
    - Include examples when useful.
    - Keep the response organized, clear, and easy to understand."""
    
elif choice == 3:
    mode = "You are a quick AI assistant. Give short, direct, and to-the-point answers. Avoid unnecessary explanations."

# creating messages history

messages=[
    SystemMessage(content=mode)  # sets the behaviour of chat bot 
]

print("============== Welcome ===============")
print("Type 0 to exit the application")
print("======================================")
while True:
    prompt = input("you :")
    messages.append(HumanMessage(content=prompt))
    if prompt =="0":
        print("Thank you for using!")
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot :",response.content)

print(messages)
