"""
Will be a RAG Implementation for the Bible. Created by Hand as opposed to using Agents.

Source:
https://medium.com/@o39joey/introduction-to-rag-with-python-langchain-62beeb5719ad


"""

import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader



embedding_model = "text-embedding-3-small"
chat_model = "gpt-4o-mini"
ALINE = "*" * 50
persist_dir = Path(__file__).resolve().parent / "data/chroma"


def main():
    
    print("Starting main function")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

    bibleChunks = BuildBibleChunks()
    print(f"Bible Chunks: {len(bibleChunks)}")
    print(ALINE)
    print()
    print()


    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=api_key,
        max_retries=3,
    )

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

        

    db = Chroma.from_documents(
        documents=bibleChunks,
        embedding=embeddings,
        collection_name="NIV2",
        persist_directory=str(persist_dir),
    )
    print("Documents added to Chroma vector store.")
    
    print(ALINE) 
    print(ALINE)
    print("Vector store setup complete.")



    return 

    results = vectorStore.similarity_search("Where was Jesus born?", k=8)
    print("Similarity Search Results:\n", results)
    print(ALINE)
    for r in results:
        print(r.page_content)
        print(ALINE)    

    print("DONE")



    retriever = vectorStore.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(model=chat_model, openai_api_key=api_key, temperature=0)

    promptTemplate = getPromptTemplate()

   


    customRagPrompt = PromptTemplate.from_template(promptTemplate)

    print("Prompt Template\n",promptTemplate)
    print(ALINE)

    ragChain = (
        {
            "context": retriever | formatDocs, "query": RunnablePassthrough()
        }
        | customRagPrompt
        | llm
        | StrOutputParser()
    )

    print("RagChain\n",ragChain)
    print(ALINE)

    s = ragChain.invoke("Where was Jesus born?")

    print("Answer:\n")
    print(s)
    print(ALINE)





def getPromptTemplate():
    retVal = """use the context provided to answer the question below.  If you do not know the answer based on the 
    context provided, say 'I don't know' based on the context and suggest that they
    reword the question.
    
    context: {context}
    question: {query}
    answer: """


    return retVal







def formatDocs(docs):
    retVal = "\n\n".join(doc.page_content for doc in docs)
    return retVal





def BuildBibleChunks():
    """
    BuildBibleChunks reads all markdown files from the "data/kjv-markdown" directory,
    splits them into smaller chunks using RecursiveCharacterTextSplitter, and returns
    a list of these chunks.
    """
    from pathlib import Path

    source_dir = Path(__file__).resolve().parent / "data" / "kjv-markdown"
    chunks = []

    loader = DirectoryLoader(
        str(source_dir),
        glob="**/*.md",
        show_progress=True,
        use_multithreading=True,
    ) 

    

    documents = loader.load()

    if not documents:
        raise ValueError(f"No markdown files found under {source_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    for document in documents:
        source = Path(document.metadata.get("source", ""))
        document.metadata["book_file"] = source.name
        document.metadata["relative_path"] = source.as_posix()

    chunks.extend(splitter.split_documents(documents))

    return chunks




def BuildDataSource():
    with open(f"data/2025_StateOfTheUnionAddress.txt", "r", encoding="utf-8") as file:
        text = file.read()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.create_documents([text])

    return chunks







def BasicPrompt():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

    prompt = ChatPromptTemplate.from_messages(
        [("system", "you are an expert in the medical field specializing in ObGyn Leadership. Give a detailed and comprehensive answer to the question provided."),
        ("user", "{question}")
        ]
    )


    questionChain = prompt|model|StrOutputParser()

    response = questionChain.invoke({"question": "What leadership roles can an ObGyn Move into that will allow them to work remotely?"})

    print("test")
    print(response)





if __name__ == "__main__":
    main()