# Retrieval Augmented Generation using Langchain and Chainlit

This project is a simple implementation of Retrieval Augmented Generation (RAG) using Langchain and Chainlit framework.
Data Helper is a bot that enables users to inquire about the content of a webpage.

![Logo](https://github.com/Joanna-Khek/chainlit-rag/blob/main/static/Logo.png)

## Tools
| Tools Used  | Usage |
| ------------- | ------------- |
| Mistal 7B Instruct V2 | LLM for inference  |
| Langchain | To chain various preprocessing steps |
| Cohere | Text embedding |
| ChromaDB | Vector store |
| Chainlit | Front-end interface |

## Steps
1. Extract web content from a URL using Langchain's WebBaseLoader
2. Split the web content using Langchain's RecursiveCharacterTextSplitter function
3. Obtain a vector representation for each chunk using Cohere's embedding model.
4. Store these vectors into a vector store (e.g Chroma DB)
5. Compare the vector representation of the user's input (query) with all vectors in the vector store. Retrieve the top few similar vectors.
6. Feed these similar vectors into the large language model's prompt template as additional context, along withh the user's query.

## Demo

![Demo](https://github.com/Joanna-Khek/chainlit-rag/blob/main/static/demo_gif.gif)

In this demo, I supplied the last of us fandom wiki page to the bot. ``https://thelastofus.fandom.com/wiki/The_Last_of_Us_Part_II``. I then asked the question ``How did Joel die?``. I asked it to return 6 most similar chunks and fed it to the prompt template as additional context.

The prompt fed to the LLM is shown below

![context](https://github.com/Joanna-Khek/chainlit-rag/blob/main/static/context_example.png)

The Chainlit framework was employed to create a user interface, allowing users to input their queries.

## Future Work
I've noticed that parameters like chunk size, chunk overlap, and the prompt template significantly impact the generated output. I intend to conduct experiments with various parameters to improve the output further.
