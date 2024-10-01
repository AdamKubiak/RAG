from langchain_chroma import Chroma


def get_retriever(embedding_model, client,collection_name ,k=2):
    vectorstore = Chroma(
        client=client, 
        collection_name=collection_name,
        embedding_function=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={'k': k}, search_type="mmr")

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