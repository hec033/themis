import os
import time
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


def instantiate_pinecone(pc_api_key: str, pc_index_name: str):
    # Load pinecone index
    pc = Pinecone(api_key=pc_api_key)

    # check if index already exists (themis-gpt-4-turbo-preview)
    if pc_index_name not in pc.list_indexes().names():
        # if it does not create, create index
        pc.create_index(
            pc_index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )
        # wait for index to be initialized
        time.sleep(1)

    # connect to index
    index = pc.Index(pc_index_name)

    return index


def pc_retrieve(openai_client, pinecone_index, text, model="text-embedding-ada-002"):
    # convert to embedding for query vector
    xq = openai_client.embeddings.create(input=[text], model=model).data[0].embedding

    # get relevant text
    res = pinecone_index.query(vector=xq, top_k=5, include_values=True, include_metadata=True)

    return res


def themis_ai():
    st.title("Themis")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-turbo-preview"
        st.session_state["pc-index-name"] = "themis-gpt-4-turbo-preview"

    # Load your API key from an environment variable or secret management service
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_THEMIS_KEY"),
    )
    # Load Pinecone index
    pc_index = instantiate_pinecone(pc_api_key=os.environ.get("PINECONE_API_KEY"), pc_index_name=st.session_state["pc-index-name"])

    # create a list to store all messages for context
    primer = f"""You are Themis. A highly intelligent system that answers legal-based user questions based on your 
    knowledge."""

    primer_aug = f"""You are Themis-Augmented. A highly intelligent system that answers legal-based user questions based on 
    contextual information retrieved from specialized legal sources. The contextual information will be provided as 
    "Context" by the user. The question will be provided as "Query" below "Context" by the user. Use the 
    information present in "Context" along with your knowledge to answer each question. If the information necessary to answer the question in 
    the "Query" cannot be found in the information present in "Context", you truthfully say "I don't know"."""

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": primer}
        ]

    if "messages_aug" not in st.session_state:
        st.session_state.messages_aug = [
            {"role": "system", "content": primer_aug}
        ]

    themis, themis_aug = st.columns(2)

    with themis:
        st.header("Themis")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # keep repeating the following
    while True:
        # prompt user
        message = input("User: ")

        # Exit program if user inputs "quit"
        if message.lower() in ["quit", "quit()", "stop", "stop()"]:
            break

        # Retrieve similar vectors from Pinecone
        retrieval_matches = pc_retrieve(openai_client=client, pinecone_index=pc_index, text=message, model="text-embedding-ada-002")

        # Augment query
        contexts = [item['metadata']['section_content'] for item in retrieval_matches['matches']]
        contexts = "\n\n---\n\n".join(contexts)
        augmented_message = f"""Context: \"{contexts}\"\n\nQuery: {message}"""
        #print('AUG START====================================================')
        #print()
        #print(augmented_message)
        #print()
        #print('AUG END====================================================')
        #print()

        # add each message to the list
        messages.append({"role": "user", "content": message})
        messages_aug.append({"role": "user", "content": augmented_message})

        # request gpt-4-turbo-preview for chat completion
        stream_completion = client.chat.completions.create(
            messages=messages,
            model=st.session_state["openai_model"],
            stream=True
        )

        # print response and add it to the message list
        print("Themis: ", end="")
        entire_message = ""
        for chunk in stream_completion:
            msg = chunk.choices[0].delta.content or ""
            print(chunk.choices[0].delta.content or "", end="")
            entire_message += msg

        # add the bot response to messages
        messages.append({"role": "assistant", "content": entire_message})

        #####################################
        # AUGMENTED
        #####################################
        # request gpt-4-turbo-preview for chat completion
        stream_completion_aug = client.chat.completions.create(
            messages=messages_aug,
            model="gpt-4-turbo-preview",
            stream=True
        )
        print()
        print()

        # print response and add it to the message list
        print("Themis Augmented: ", end="")
        entire_message_aug = ""
        for chunk_aug in stream_completion_aug:
            msg_aug = chunk_aug.choices[0].delta.content or ""
            print(chunk_aug.choices[0].delta.content or "", end="")
            entire_message_aug += msg_aug

        # add the bot response to messages
        messages_aug.append({"role": "assistant", "content": entire_message_aug})
        print()
        print()
        print()
        print()


if __name__ == "__main__":
    print("Starting chatting with Themis (type 'quit' to stop)!")
    themis_ai()
