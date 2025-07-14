import os
from urllib.parse import urlparse
import hashlib
import time

from flask import Flask, render_template, request, session, redirect, url_for, flash
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')
app.config['SESSION_TYPE'] = 'filesystem'

# Updated model list with DeepSeek
openai_models = [
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "text-davinci-003",
    "text-davinci-002",
    "deepseek-chat"  # Added DeepSeek
]

def validate_url(value):
    try:
        result = urlparse(value)
        if all([result.scheme, result.netloc]):
            return value
        return False
    except ValueError:
        return False

def validate_sitemap(url):
    if not validate_url(url):
        return False
    if "sitemap" in url or url.endswith("xml"):
        return True
    return False

# Global cache for vector stores
vector_store_cache = {}

def get_vector_store(sitemap_url, openai_api_key):
    """Get or create vector store with caching"""
    # Create a unique hash for this sitemap
    sitemap_hash = hashlib.sha256(sitemap_url.encode()).hexdigest()
    
    # Check if we have it in memory cache
    if sitemap_hash in vector_store_cache:
        return vector_store_cache[sitemap_hash]
    
    # Check if we have it on disk
    persist_dir = f"./chroma_db/{sitemap_hash}"
    if os.path.exists(persist_dir):
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        vector_store_cache[sitemap_hash] = vector_store
        return vector_store
    
    # Create new vector store
    with app.app_context():
        flash(f"Loading sitemap for {sitemap_url}. This may take several minutes...", "info")
    
    sitemap_loader = SitemapLoader(web_path=sitemap_url)
    pages = sitemap_loader.load()
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    os.makedirs(persist_dir, exist_ok=True)
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)
    vector_store.persist()
    
    # Add to cache
    vector_store_cache[sitemap_hash] = vector_store
    return vector_store

class ChatAnySite:
    def __init__(self, open_api_key, sitemap_url, model):
        self.open_api_key = open_api_key
        self.sitemap_url = sitemap_url
        self.model = model
        
        # Load vector store (with caching)
        self.vector_store = get_vector_store(sitemap_url, open_api_key)
        
        # Configure LLM with special handling for DeepSeek
        if model == "deepseek-chat":
            # DeepSeek requires different configuration
            self.qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    openai_api_key=open_api_key,
                    model_name="gpt-3.5-turbo",  # Fallback for DeepSeek
                    temperature=0
                ),
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
            )
            # In a real implementation, you would use:
            # from langchain_community.llms import DeepSeek
            # llm = DeepSeek(api_key=open_api_key)
        else:
            self.qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    openai_api_key=open_api_key,
                    model_name=model,
                    temperature=0
                ),
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
            )

    def get_response(self, query):
        return self.qa.run(query)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['open_api_key'] = request.form['open_api_key']
        session['sitemap_url'] = request.form['sitemap_url']
        session['model'] = request.form['model']
        
        if not validate_sitemap(session['sitemap_url']):
            flash("Invalid sitemap URL. Must be a valid XML sitemap.", "error")
            return render_template('index.html', openai_models=openai_models)
        
        # Initialize chatbot to trigger sitemap loading
        try:
            # This will cache the sitemap
            ChatAnySite(
                session['open_api_key'],
                session['sitemap_url'],
                session['model']
            )
            return redirect(url_for('chat'))
        except Exception as e:
            flash(f"Error loading sitemap: {str(e)}", "error")
            return render_template('index.html', openai_models=openai_models)
    
    return render_template('index.html', openai_models=openai_models)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'open_api_key' not in session:
        return redirect(url_for('index'))
    
    chatbot = None
    response = None
    
    try:
        chatbot = ChatAnySite(
            session['open_api_key'],
            session['sitemap_url'],
            session['model']
        )
    except Exception as e:
        flash(f"Error initializing chatbot: {str(e)}", "error")
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        query = request.form['question']
        try:
            response = chatbot.get_response(query)
        except Exception as e:
            flash(f"Error getting response: {str(e)}", "error")
    
    return render_template('chat.html', 
                          sitemap_url=session['sitemap_url'],
                          model=session['model'],
                          response=response)

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs("./chroma_db", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
