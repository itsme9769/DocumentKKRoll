import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain.llms import OpenAI
import pypdf
import json

# Load the OpenAI API key from config.json

with open('config.json', 'r') as config_file:

    config = json.load(config_file)

    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "docs"

def clear_docs_folder():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def download_file():
    url = 'https://chatbot-client-onboarding-uat.idealake.com/temp/RollsMenu1.pdf'
    r = requests.get(url, allow_redirects=True)
    file_name = url.split("/")[-1]
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    with open(file_path, 'wb') as file: 
        file.write(r.content)
    return file_path

def construct_index(directory_path):
    num_outputs = 256
    _llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(llm_predictor=_llm_predictor)
    docs = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    index.storage_context.persist(persist_dir="storage")
    print("Indexing Completed....")
    return index

def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    query_engine = load_index_from_storage(storage_context).as_query_engine()
    response = query_engine.query(input_text)
    return response.response

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Clear previous files in the docs folder
    clear_docs_folder()

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"File uploaded and saved to {file_path}")

        # Reconstruct the index with the new file
        construct_index(UPLOAD_FOLDER)
        return jsonify({"message": "File uploaded and indexed successfully"}), 200

    return jsonify({"error": "File upload failed"}), 500

@app.route('/api/chat', methods=['GET'])
def chat_api():
    print("Request Received......")
    data = request.get_json()
    input_text = data.get('input_text', '')
    response_text = chatbot(input_text)
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)
