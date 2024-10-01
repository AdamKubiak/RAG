from langchain_chroma import Chroma
from langchain.load import dumps, loads

def get_retriever(embedding_model, client,collection_name ,k=2):
    vectorstore = Chroma(
        client=client, 
        collection_name=collection_name,
        embedding_function=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={'k': k}, search_type="mmr")

def get_unique_docs(docs):
    """ Get unique documents """
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    return unique_docs
# def create_basic_chain(retriever, model):
# # Prompt
#     template = """Answer the question based only on the following context:
#     {context}

#     Question: {question}
#     """
#     prompt = ChatPromptTemplate.from_template(template)

#     rag_chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | model
#         | StrOutputParser()
#     )
    
#     return rag_chain