from flask import Flask, request, jsonify
from langchain_ollama.llms import OllamaLLM
from utils.pdf_preprocessing import PdfPreprocessor
from utils.retriver import get_retriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from time import perf_counter
import chromadb

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

ALLOWED_EXTENSIONS = {'pdf'}
DB_DIRECTORY = os.path.curdir + "/chroma.sqlite3"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app():
    app = Flask(__name__)
    return app

app = create_app()

llm  = OllamaLLM(model="llama3.2:3b", base_url="http://localhost:11434")

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)

client = chromadb.HttpClient(host='localhost', port=8000)

@app.route('/ai', methods=["POST"])
def modelPost():
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
    