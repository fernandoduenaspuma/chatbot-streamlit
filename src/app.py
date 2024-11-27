import boto3
import os
import logging
import streamlit as st
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrockConverse

os.environ['AWS_PROFILE']='aws-personal'

client = boto3.client('sts')
logging.basicConfig(level=logging.CRITICAL)

def get_iam_user_id():    
    # Get the caller identity
    identity = client.get_caller_identity()
    
    # Extract and return the User ID
    user_id = identity['UserId']
    return user_id

model = ChatBedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    max_tokens=2048,
    temperature=0.0,
    top_p=1,
    stop_sequences=["\n\nHuman"],
    verbose=True
)

# Initialize the DynamoDB chat message history
table_name = "SessionTable"
session_id = get_iam_user_id()  # You can make this dynamic based on the user session
history = DynamoDBChatMessageHistory(table_name=table_name, session_id=session_id)

# Create the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

output_parser = StrOutputParser()

# Combine the prompt with the Bedrock LLM
chain = prompt_template | model | output_parser

# Integrate with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: DynamoDBChatMessageHistory(
        table_name=table_name, session_id=session_id
    ),
    input_messages_key="question",
    history_messages_key="history",
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
    response = chain_with_history.invoke({"question": prompt}, config=config)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})