import boto3
import logging
import os
import streamlit as st
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrockConverse
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter
from langchain_aws import ChatBedrock
import uuid

os.environ['AWS_PROFILE']='aws-personal'
client = boto3.client('sts')
logging.basicConfig(level=logging.CRITICAL)
session_store = {}

def get_session_id():
    # Check if session ID already exists in session state
    if 'session_id' not in st.session_state:
        # Generate a new UUID if not exists
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

model = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)
# Amazon Bedrock - KnowledgeBase Retriever 
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="ARVWHIZYMJ", # 👈 Set your Knowledge base ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)


# Initialize the DynamoDB chat message history
table_name = "SessionTable"
session_id = get_session_id() # You can make this dynamic based on the user session
history = DynamoDBChatMessageHistory(table_name=table_name, session_id=session_id)

# Create the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."
         "Answer the question based only on the following context:\n {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


output_parser = StrOutputParser()

# Combine the prompt with the Bedrock LLM
#chain = prompt_template | model | output_parser
chain = (
    RunnableParallel({
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    })
    .assign(response = prompt | model | StrOutputParser())
    .pick(["response", "context"])
)


# Integrate with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: DynamoDBChatMessageHistory(
        table_name=table_name, session_id=session_id
    ),
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="response",
)

st.title("LangChain DynamoDB Bot")

# Load messages from DynamoDB and populate chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # Load the stored messages from DynamoDB
    stored_messages = history.messages  # Retrieve all stored messages
    
    # Populate the session state with the retrieved messages
    for msg in stored_messages:
        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
        st.session_state.messages.append({"role": role, "content": msg.content})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate assistant response using Bedrock LLM and LangChain
    config = {"configurable": {"session_id": session_id}}
    result = chain_with_history.invoke({"question": prompt}, config=config)
    response=result['response']

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})