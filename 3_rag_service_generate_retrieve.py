# import json
# import os

# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS

# load_dotenv()

# JSON_PATH = "3_ticket_KB.json"
# VECTOR_DB_PATH = "vector_db2"

# os.makedirs(VECTOR_DB_PATH, exist_ok=True)


# def create_vector_database():

#     print("Loading JSON knowledge base...")

#     with open(JSON_PATH, "r", encoding="utf-8") as file:
#         data = json.load(file)

#     tickets = data["tickets"]

#     print(f"Loaded {len(tickets)} tickets")

#     documents = []

#     for ticket in tickets:

#         ticket_id = ticket["ticket_id"]
#         category = ticket["category"]
#         problem = ticket["problem_statement"]
#         solutions = ticket["solution"]

#         solution_text = "\n".join(
#             [f"{i + 1}. {step}" for i, step in enumerate(solutions)]
#         )

#         content = f"""
# Ticket ID: {ticket_id}
# Category: {category}

# Problem:
# {problem}

# Solution:
# {solution_text}
# """.strip()

#         documents.append(
#             Document(
#                 page_content=content,
#                 metadata={
#                     "ticket_id": ticket_id,
#                     "category": category
#                 }
#             )
#         )

#     print(f"Created {len(documents)} documents")

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=200,
#         chunk_overlap=10
#     )

#     chunks = splitter.split_documents(documents)

#     print(f"Created {len(chunks)} chunks")

#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-small"
#     )

#     print("Generating embeddings...")

#     vectorstore = FAISS.from_documents(
#         documents=chunks,
#         embedding=embeddings
#     )

#     print("Saving vector database...")

#     vectorstore.save_local(VECTOR_DB_PATH)

#     print("Done!")
#     print(f"Vector database saved in: {VECTOR_DB_PATH}")


# if __name__ == "__main__":
#     create_vector_database()




#--------------------------------------------------------


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_PATH = "vector_db2"


def load_vector_db():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def ask_question(question="hi"):

    vectorstore = load_vector_db()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    result = qa.invoke({
        "query": question
    })

    print("\n================ ANSWER ================\n")
    print(result["result"])

    print("\n================ SOURCES ================\n")

    for doc in result["source_documents"]:

        print("=" * 60)

        print("Ticket ID:", doc.metadata.get("ticket_id"))
        print("Category:", doc.metadata.get("category"))

        print("\nContent:")
        print(doc.page_content)


if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk Question (type exit to quit): "
        )

        if question.lower() == "exit":
            break

        ask_question(question)