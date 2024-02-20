'''
This module is used to chat with the chatbot using the prompt method.
'''

import os
from langchain.llms import OpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

def load_llm():

    # models llama2-uncensored, mistral, codellama, wizard-math, falcon

    llm = Ollama(
    model="mistral",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

def chat_with_prompt(query):
    # llm = ChatOpenAI(model=MODEL_NAME, temperature=0.8)
    llm = load_llm()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_template(f"You are an intelligent AI bot. Explain each concept {query} in simple terms.")

    chatbot = LLMChain(llm=llm, prompt=prompt, memory=memory)

    response = chatbot({"query": query})
    return response["text"]

if __name__ == "__main__":
    while True:
        user_query = input("Enter your query: ")
        if user_query == "exit":
            break
        result = chat_with_prompt(user_query)
        print(result)
        print("-"*100)

