from flask import Flask, request, jsonify
from langchain_ollama.llms import OllamaLLM
from utils.pdf_preprocessing import PdfPreprocessor
from utils.retriver import get_retriever, get_unique_docs
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from time import perf_counter
import chromadb

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter

ALLOWED_EXTENSIONS = {'pdf'}
DB_DIRECTORY = os.path.curdir + "/chroma.sqlite3"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app():
    app = Flask(__name__)
    return app

app = create_app()

llm  = OllamaLLM(model="llama3.2:3b", base_url="http://localhost:11434", temperature=0)

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)

client = chromadb.HttpClient(host='localhost', port=8000)

@app.route('/ai/simple', methods=["POST"])
def simplePost():
    json_content = request.json
    retriever = get_retriever(embedding_model, client, "nutritient")
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    llm_response = rag_chain.invoke(json_content['query'])
    
    response = {"Question": json_content['query'], "Anwer": llm_response}
    return jsonify(response)

@app.route('/ai/multi_query', methods=["POST"])
def multiQueryPost():
    try:
        json_content = request.json
        retriever = get_retriever(embedding_model, client, "nutritient")

        query_generation_template = """You are an AI language model assistant. Your task is to generate five 
                    different versions of the given user question to retrieve relevant documents from a vector 
                    database. By generating multiple perspectives on the user question, your goal is to help
                    the user overcome some of the limitations of the distance-based similarity search. 
                    Provide these alternative questions separated by newlines. Original question: {question}"""
        query_generation_prompt = ChatPromptTemplate.from_template(query_generation_template)
        
        queries_chain = (
            {"question": itemgetter("query")}
            | query_generation_prompt
            | llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
            | (lambda x: [query for query in x if query.strip() != ''])
            | (lambda x: [line for line in x if line.strip().startswith(('1.', '2.', '3.', '4.', '5.'))])
        )
        
        retrieval_chain = (
            retriever.map()
            | (lambda docs: [doc for sublist in docs for doc in sublist])  # Flatten list of lists
            | get_unique_docs
        )
        
        base_prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context. Use all documents as context:
        {context}

        Question: {question}
        """
        )
        
        def format_docs(docs):
            return "\n".join([f"Document {i+1}:\n{doc.page_content}\n" for i, doc in enumerate(docs)])
        
        final_rag_chain = (
            {
                "context": lambda x: format_docs(x["context"]),
                "question": itemgetter("query")
            }
            | base_prompt
            | llm
            | StrOutputParser()
        )
        
        # Generate additional questions
        additional_questions = queries_chain.invoke(json_content)
        additional_questions.append(f'6. {json_content['query']}')
        # Retrieve documents
        retrieved_docs = retrieval_chain.invoke(additional_questions)
        # Generate final answer
        llm_response = final_rag_chain.invoke({"query": json_content['query'], "context": retrieved_docs})
        
        response = {
            "Original_Question": json_content['query'],
            "Generated_Questions": additional_questions,
            "Retrieved_Context": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in retrieved_docs
            ],
            "Answer": llm_response
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/pdf', methods = ["POST"])
def method_name():
    if ('file' not in request.files) and ('collection' not in request.form):
            return jsonify({"error": "No file or collection name part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        raise ValueError('No selected file')
    
    
    if file and allowed_file(file.filename):
        start = perf_counter()
        pdf_preproc = PdfPreprocessor(file, file.filename, 300)
        preprocessed_splits = pdf_preproc.preprocess_pdf()
        
        vectorstore = Chroma(
            client=client,
            collection_name=request.form.get('collection'),
            embedding_function=embedding_model,
        )
        
        ids = [str(_) for _ in range(len(preprocessed_splits))]
        vectorstore.add_documents(documents=preprocessed_splits, ids=ids)
        end = perf_counter()
        response = {"status": "Successfully Uploaded File",
                    "filename": file.filename,
                    "splits_num": len(preprocessed_splits), 
                    "duration": end-start}
        
    else:
        response = {"status": "Unsuccessfully Uploaded File, something went wrong!",
                    "filename": '',
                    "splits_num": 0,
                    'duration': 0.}    
    
    return jsonify(response)

@app.route('/getDb', methods=["GET"])
def dbGet():
    collections = client.list_collections()
    print(collections)

    # Convert collections to a JSON serializable format
    serializable_collections = []
    for collection in collections:
        serializable_collections.append({
            'name': collection.name, 
            'id': collection.id 
        })
    count_in_collection = client.get_collection("nutritient").count()

    return jsonify({"collections": serializable_collections,
                    "count": count_in_collection})

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    