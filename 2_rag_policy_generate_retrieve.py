# # RAG Generator
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS

# import os
# from dotenv import load_dotenv
# load_dotenv()

# PDF_PATH = r"2_policies_small.pdf"
# VECTOR_DB_PATH = "vector_db"

# os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# def create_vector_database():
#     print("Loading PDF...")

#     loader = PyPDFLoader(PDF_PATH)
#     documents = loader.load()
#     print(f"Loaded {len(documents)} pages")

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100
#     )
#     chunks = splitter.split_documents(documents)
#     print(f"Created {len(chunks)} chunks")

#     embeddings = OpenAIEmbeddings()
#     print("Generating embeddings...")

#     vectorstore = FAISS.from_documents(
#         documents=chunks,
#         embedding=embeddings
#     )
#     print("Saving vector database...")
#     vectorstore.save_local(VECTOR_DB_PATH)
#     print("Done!")

# if __name__ == "__main__":
#     create_vector_database()











#Rag Retriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

VECTOR_DB_PATH = "vector_db"

def load_vector_db():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def ask_question(question="hi"):
    vectorstore = load_vector_db()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    result = qa.invoke({"query": question})
    print("\nAnswer\n")
    print(result["result"])
    print("\nSources\n")

    for doc in result["source_documents"]:
        print("=" * 60)
        print(doc.metadata)
        print(doc.page_content[:300])


# if __name__ == "__main__":
#     while True:
#         question = input("\nAsk Question (type exit to quit): ")
#         if question.lower() == "exit":
#             break
#         ask_question(question)

if __name__ == "__main__":
    ask_question("Tell me about leave policy in two bullet points.") 