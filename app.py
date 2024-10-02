from flask import Flask, request, jsonify
from langchain_ollama.llms import OllamaLLM
from utils.pdf_preprocessing import PdfPreprocessor
from utils.retriver import get_retriever, get_unique_docs, reciprocal_rank_fusion, format_qa_pair
from utils.prompts import base_prompt, query_generation_template, query_decomposition_template, query_COT_template, base_prompt_decomposition, router_template
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from time import perf_counter
import chromadb
import requests

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

@app.route('/ai/llm_route_question', methods=["POST"])
def llm_route_question():
    try:
        router_prompt = ChatPromptTemplate.from_template(router_template)

        router_chain = (
            router_prompt
            | llm
            | StrOutputParser()
        )
        json_content = request.json
        question = json_content['query']
        
        # Use the LLM to decide the routing
        method = router_chain.invoke({"question": question}).strip()
        
        base_url = request.url_root  # This gets the root URL of your Flask app
        
        # Route based on LLM decision
        if method == "1":
            endpoint = f"{base_url}ai/simple"
        elif method == "2":
            endpoint = f"{base_url}ai/multi_query_fusion"
        elif method == "3":
            endpoint = f"{base_url}ai/decomposition_fusion"
        else:
            return jsonify({"error": "Invalid routing decision"}), 400
        
        response = requests.post(endpoint, json=json_content)
        response_data = response.json()
        
        response_data['method'] = method
        return response_data, response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

@app.route('/ai/multi_query_fusion', methods=["POST"])
def queryFusion():
    try:
        json_content = request.json
        retriever = get_retriever(embedding_model, client, "nutritient")

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
            | reciprocal_rank_fusion
        )
        
        def format_docs(docs):
            return "\n".join([f"Document {i+1}:\n{doc[0].page_content}\n" for i, doc in enumerate(docs)])
        
        query_base = ChatPromptTemplate.from_template(base_prompt)
        
        final_rag_chain = (
            {
                "context": lambda x: format_docs(x["context"]),
                "question": itemgetter("query")
            }
            | query_base
            | llm
            | StrOutputParser()
        )
        
        # Generate additional questions
        additional_questions = queries_chain.invoke({"query": json_content["query"]})
        additional_questions.append(f'6. {json_content["query"]}')
        
        # Retrieve documents
        retrieved_docs = retrieval_chain.invoke(additional_questions)
        retrieved_docs = retrieved_docs[:3]
        llm_response = final_rag_chain.invoke({"query": json_content["query"], "context": retrieved_docs})
        
        response = {
            "Original_Question": json_content["query"],
            "Generated_Questions": additional_questions,
            "Retrieved_Context": [
                {
                    "content": doc[0].page_content,
                    "metadata": doc[0].metadata,
                    "score": doc[1]
                } for doc in retrieved_docs
            ],
            "Answer": llm_response
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ai/decomposition_fusion', methods=["POST"])
def queryDecompositionCOT():
    try:
        json_content = request.json
        retriever = get_retriever(embedding_model, client, "nutritient")

        query_generation_prompt = ChatPromptTemplate.from_template(query_decomposition_template)
        decompositon_prompt = ChatPromptTemplate.from_template(query_COT_template)
        queries_chain = (
            {"question": itemgetter("query")}
            | query_generation_prompt
            | llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
            | (lambda x: [query for query in x if query.strip() != ''])
            | (lambda x: [line for line in x if line.strip().startswith(('1.', '2.', '3.', '4.', '5.'))])
        )
        
        # Generate additional questions
        additional_questions = queries_chain.invoke({"query": json_content["query"]})
        q_a_pairs = ""
        for q in additional_questions:
            rag_chain = (
                {"context": itemgetter("question") | retriever,
                 "question": itemgetter("question"), "q_a_pairs": itemgetter("q_a_pairs")}
                | decompositon_prompt
                | llm
                | StrOutputParser()
            )
            print(q)
            answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
            q_a_pair = format_qa_pair(q,answer)
            q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
        
        final_prompt = ChatPromptTemplate.from_template(base_prompt_decomposition)
        
        rag_chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough() 
            }
            | final_prompt
            | llm
            | StrOutputParser()
        )
        
        final_anwer = rag_chain.invoke({"context": q_a_pair, "question": json_content['query']})
        
        response = {
            "Original_Question": json_content["query"],
            "Generated_Questions": additional_questions,
            "Answer": final_anwer,
            "context": q_a_pairs
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ai/multi_query', methods=["POST"])
def multiQueryPost():
    try:
        json_content = request.json
        retriever = get_retriever(embedding_model, client, "nutritient")

        
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
        
        def format_docs(docs):
            return "\n".join([f"Document {i+1}:\n{doc.page_content}\n" for i, doc in enumerate(docs)])
        
        query_base = ChatPromptTemplate.from_template(base_prompt)
        
        final_rag_chain = (
            {
                "context": lambda x: format_docs(x["context"]),
                "question": itemgetter("query")
            }
            | query_base
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
    