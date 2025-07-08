import os
import hashlib
import pickle
import time
from pathlib import Path
import faiss

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.llms.openai_like import OpenAILike

class KnowledgeBase:
    """Knowledge base using LlamaIndex for document retrieval and querying"""

    def __init__(
        self,
        api_key,
        embed_base_url,
        llm_base_url,
        model_name="gpt-4o",
        knowledge_index_dir="./index_db",
        tracker_file="document_tracker.pkl",
        embedding_model="text-embedding-ada-002",
        embedding_dim=1536
    ):
        """Initialize the knowledge base

        Args:
            api_key (str): API key for OpenAI or compatible services
            embed_base_url (str): Base URL for embedding API
            llm_base_url (str): Base URL for LLM API
            model_name (str): Model name for the LLM
            knowledge_index_dir (str): Directory to store the vector index
            tracker_file (str): File to track document changes
            embedding_model (str): Model name for embeddings
            embedding_dim (int): Dimension of embeddings
        """
        self.api_key = api_key
        self.embed_base_url = embed_base_url
        self.llm_base_url = llm_base_url
        self.model_name = model_name
        self.knowledge_index_dir = knowledge_index_dir
        self.tracker_file = tracker_file
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

        # Create embedding model
        self.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=api_key,
            base_url=embed_base_url
        )

        # Create LLM
        self.llm = OpenAILike(
            model=model_name,
            api_base=llm_base_url,
            api_key=api_key,
            is_chat_model=True
        )

        # Initialize index
        self.index = None
        self.query_engine = None

    def _create_document_tracker(self):
        """Create or load a document tracker to monitor file changes"""
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_document_tracker(self, tracker):
        """
        The tracker file is saved as a pickled dictionary with the following structure:
        {
            "file_path1": {
                "hash": "md5_hash_of_content_and_mtime",
                "document": Document(...),  # LlamaIndex Document object
                "last_updated": timestamp  #timestamp of when the document was last processed or updated, not the document's original timestamp.
            },
            "file_path2": {
                "hash": "md5_hash_of_content_and_mtime",
                "document": Document(...),
                "last_updated": timestamp
            },
            "https://example.com": {
                "hash": "md5_hash_of_url",
                "document": Document(...),
                "last_updated": timestamp
            },
            ...
        }
        """
        with open(self.tracker_file, "wb") as f:
            pickle.dump(tracker, f)

    def _get_document_hash(self, file_path):
        """Generate hash based on file content and modification time"""
        with open(file_path, "rb") as f:
            content = f.read()
        mtime = os.path.getmtime(file_path)
        return hashlib.md5(f"{content}{mtime}".encode()).hexdigest()

    def _load_documents_with_change_tracking(self, code_dirs, urls):
        """Load documents and track which ones have changed"""
        tracker = self._create_document_tracker()

        changed_code_files = []
        unchanged_documents = []

        # Process code files from multiple directories
        for code_dir in code_dirs:
            if not os.path.exists(code_dir):
                print(f"[Warning]: Directory not found: {code_dir}")
                continue

            for root, _, files in os.walk(code_dir):
                for file in files:
                    if file.endswith(('.py', '.md')):
                        file_path = os.path.join(root, file)
                        current_hash = self._get_document_hash(file_path)

                        if file_path in tracker and tracker[file_path]["hash"] == current_hash:
                            # File hasn't changed, use existing document
                            unchanged_documents.append(tracker[file_path]["document"])
                            print(f"[Info]: Using cached document: {file_path}")
                        else:
                            # File is new or changed
                            changed_code_files.append(file_path)
                            print(f"[Info]: new/changes file: {file_path}")

        # Process new/changed code files
        new_documents = []
        if changed_code_files:
            reader = SimpleDirectoryReader(input_files=changed_code_files)
            new_documents = reader.load_data() # list of doc object

            # Update tracker with new documents
            for doc in new_documents:
                for file_path in changed_code_files:
                    if file_path in doc.metadata.get('file_path', ''):
                        current_hash = self._get_document_hash(file_path)
                        tracker[file_path] = {
                            "hash": current_hash,
                            "document": doc,
                            "last_updated": time.time()
                        }
                        break

        # Process URLs
        url_documents = []
        urls_to_fetch = []

        for url in urls:
            if url in tracker and (time.time() - tracker[url]["last_updated"]) < 86400:  # 24h cache
                # URL content hasn't expired
                url_documents.append(tracker[url]["document"])
                print(f"[Info]: Using cached URL: {url}")
            else:
                # URL is new or cache expired
                urls_to_fetch.append(url)
                print(f"[Info]: Will fetch URL: {url}")

        # Fetch new/expired URLs
        new_url_docs = []
        if urls_to_fetch:
            url_loader = BeautifulSoupWebReader()
            try:
                new_url_docs = url_loader.load_data(urls=urls_to_fetch)

                # Update tracker with new URL documents
                for i, doc in enumerate(new_url_docs):
                    if i < len(urls_to_fetch):
                        url = urls_to_fetch[i]
                        tracker[url] = {
                            "hash": hashlib.md5(url.encode()).hexdigest(),
                            "document": doc,
                            "last_updated": time.time()
                        }
            except Exception as e:
                print(f"[Error]: loading URLs: {e}")

        # Save tracker
        self._save_document_tracker(tracker)

        # Combine all documents
        all_documents = unchanged_documents + new_documents + url_documents
        has_changes = (len(new_documents) > 0 or len(new_url_docs) > 0)
        return all_documents, has_changes

    def build_index(self, code_dirs, urls):
        """Build or update the index from code directories and URLs

        Args:
            code_dirs (list): List of directories to index
            urls (list): List of URLs to index
        """
        # Load documents with change tracking
        all_documents, has_changes = self._load_documents_with_change_tracking(code_dirs, urls)

        if len(all_documents) == 0:
            print("[Info]: no documents to index")
            return

        # If no changes and index exists, just load it
        if not has_changes and os.path.exists(self.knowledge_index_dir):
            print("[Info]: no changes detected, loading existing index...")
            vector_store = FaissVectorStore.from_persist_dir(self.knowledge_index_dir)
            storage_context = StorageContext.from_defaults(persist_dir=self.knowledge_index_dir, vector_store=vector_store)
            self.index = load_index_from_storage(storage_context)
        else:
            print("[Info]: Building/updating index...")
            # Parse documents to nodes
            parser = SimpleNodeParser()
            nodes = parser.get_nodes_from_documents(all_documents)

            if not os.path.exists(self.knowledge_index_dir):
                print("[Info]: creating new index...")
                # Create new index
                faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self.index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    show_progress=True
                )
                # Save the index
                self.index.storage_context.persist(persist_dir=self.knowledge_index_dir)
            else:
                print("[Info]: loading existing index...")
                # Load existing index
                vector_store = FaissVectorStore.from_persist_dir(self.knowledge_index_dir)
                storage_context = StorageContext.from_defaults(persist_dir=self.knowledge_index_dir, vector_store=vector_store)
                self.index = load_index_from_storage(storage_context)

                # Update index with new nodes
                for node in nodes:
                    self.index.insert(node) # The insert() method only processes the new nodes, saving API calls and computation

                # Save updated index
                self.index.storage_context.persist(persist_dir=self.knowledge_index_dir)

            print(f"[Info]: Index saved to {self.knowledge_index_dir}")

        # Create query engine
        self.query_engine = self.index.as_query_engine(llm=self.llm)

    def query(self, query_str):
        """Query the knowledge base

        Args:
            query_str (str): Query string

        Returns:
            str: Response from the knowledge base
        """
        if not self.query_engine:
            raise ValueError("Index not built. Call build_index() first.")

        try:
            response = self.query_engine.query(query_str)
            return response.response.strip()
        except Exception as e:
            print(f"[Error]: querying knowledge base: {e}")
            return f"[Error]: retrieving information: {str(e)}"
