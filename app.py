import bs4
import os

import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider

from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import CohereEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer, pipeline

@cl.cache
def load_llm():
    model = "mistralai/Mistral-7B-Instruct-v0.2"

    quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16
                            )
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, 
                                                 quantization_config=quantization_config)
    streamer = TextStreamer(tokenizer, skip_prompt=False)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        streamer=streamer,
        max_new_tokens=150
    )
    # model = AutoModelForCausalLM.from_pretrained(model,
    #                                              quantization_config=quantization_config)
    
    #pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

llm = load_llm()

add_llm_provider(
    LangchainGenericProvider(
        id=llm._llm_type, name="Mistral-chat", llm=llm, is_chat=False
    )
)


def create_chain(docs, llm):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embedding = CohereEmbeddings(cohere_api_key = os.getenv("COHERE_API"))
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    prompt = hub.pull("rlm/rag-prompt")

    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\n
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n
    Question: {question}\n
    Context: {context}\n
    Answer: """
    
    prompt = PromptTemplate(template=template, input_variables=['question', 'context'])

    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
        
    return rag_chain

@cl.on_chat_start
async def init():

    elements = [
        cl.Image(name="logo", display="inline",
                 path="static/Logo.png")
    ]
    await cl.Message(content="Hello! Welcome to Data Helper Chatbot!",
                     elements=elements).send()


    link = None

    while link == None:
        link = await cl.AskUserMessage(
            content = "Please input a web link to begin!"
        ).send()

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=([link['output']]),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("mw-parser-output")
            )
        )
    )
    docs = loader.load()
    rag_chain = create_chain(docs, llm)

    # Create user session to store data
    cl.user_session.set("rag_chain", rag_chain)

    # Send response back to user
    await cl.Message(
        content = f"Content parsed! Ask me anything related to the weblink!"
    ).send()
    

@cl.on_message # this function will be called every time a user inputs a message in the UI
async def main(message: str):

   
    rag_chain = cl.user_session.get("rag_chain")
    res = rag_chain.invoke(message.content)
    await cl.Message(content=res).send()

