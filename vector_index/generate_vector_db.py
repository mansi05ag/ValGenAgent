import os
import hashlib
import pickle
import time
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.llms.openai_like import OpenAILike

from llama_index.readers.github import GithubRepositoryReader,GithubClient #pip install llama-index-readers-github
os.environ["GITHUB_TOKEN"] = "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"
github_token = os.environ.get("GITHUB_TOKEN")

owner = "pytorch"
repo = "pytorch"
branch = "v2.7.1"
target_directories = ["test/distributed"]
github_client = GithubClient(github_token=github_token, verbose=True)

# reader = GithubRepositoryReader(
#     github_client=github_client,
#     owner=owner,
#     repo=repo,
#     use_parser=True,
#     verbose=False,
#     # Use filter_directories to specify which directories to include
#     filter_directories=(
#         target_directories,
#         GithubRepositoryReader.FilterType.INCLUDE,
#     ),
#     # ignore_directories=["docs", "examples"],
#     # And filter by file extensions
#     # filter_file_extensions=(
#     #     [
#     #         ".png",
#     #     ],
#     #     GithubRepositoryReader.FilterType.EXCLUDE,
#     # ),
#     # timeout=httpx.Timeout(30.0)
# )

# documents = reader.load_data(branch=branch)

# print(f"Ingested {len(documents)} documents from the specified directory(ies).")

# # index = VectorStoreIndex.from_documents(documents)

# import pdb; pdb.set_trace()

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
        self.retriever = None # Gets the most relevant document chunks without LLM processing

    def build_index(self, code_dirs, urls):
        """Build the index from code directories and URLs - always rebuild to avoid corruption"""
        try:
            # Check if index already exists
            if os.path.exists(self.knowledge_index_dir):
                try:
                    print("[Info]: Loading existing index...")
                    storage_context = StorageContext.from_defaults(persist_dir=self.knowledge_index_dir)
                    self.index = load_index_from_storage(
                        storage_context,
                        embed_model=self.embed_model
                    )

                    # Create query engine
                    print("[Info]: Creating query engine from existing index...")
                    self.query_engine = self.index.as_query_engine(llm=self.llm)
                    print("[Info]: Successfully loaded existing index")
                    return

                except Exception as e:
                    print(f"[Warning]: Failed to load existing index: {e}")
                    print("[Info]: Will rebuild index from scratch...")
                    # Remove corrupted index
                    import shutil
                    shutil.rmtree(self.knowledge_index_dir)

            print("[Info]: Building index from scratch...")

            # Load all documents
            all_documents = []

            # Load code files
            for code_dir in code_dirs:
                if os.path.exists(code_dir):
                    print(f"[Info]: Loading documents from {code_dir}")
                    reader = SimpleDirectoryReader(code_dir, recursive=True)
                    docs = reader.load_data()
                    all_documents.extend(docs)
                    print(f"[Info]: Loaded {len(docs)} documents from {code_dir}")

            # Load URLs
            if urls:
                print(f"[Info]: Loading {len(urls)} URLs")
                url_loader = BeautifulSoupWebReader()
                try:
                    url_docs = url_loader.load_data(urls=urls)
                    all_documents.extend(url_docs)
                    print(f"[Info]: Loaded {len(url_docs)} URL documents")
                except Exception as e:
                    print(f"[Warning]: Failed to load URLs: {e}")

            if len(all_documents) == 0:
                print("[Warning]: No documents to index")
                return

            print(f"[Info]: Total documents loaded: {len(all_documents)}")

            # Parse documents to nodes
            print("[Info]: Parsing documents into nodes...")
            parser = SimpleNodeParser()
            nodes = parser.get_nodes_from_documents(all_documents)
            print(f"[Info]: Created {len(nodes)} nodes")

            if len(nodes) == 0:
                print("[Warning]: No nodes created from documents")
                return

            # Create index using simple storage (no FAISS)
            print("[Info]: Building vector index...")
            self.index = VectorStoreIndex(
                nodes,
                embed_model=self.embed_model,
                show_progress=True
            )

            # Save the index
            print(f"[Info]: Saving index to {self.knowledge_index_dir}")
            self.index.storage_context.persist(persist_dir=self.knowledge_index_dir)

            # Create query engine
            print("[Info]: Creating query engine...")
            self.query_engine = self.index.as_query_engine(llm=self.llm)

        except Exception as e:
            print(f"[Error]: Failed to build index: {e}")
            import traceback
            traceback.print_exc()
            raise

    def query(self, query_str):
        """Query the knowledge base"""
        if not self.query_engine:
            raise ValueError("Index not built. Call build_index() first.")

        try:
            response = self.query_engine.query(query_str)
            return response.response.strip()
        except Exception as e:
            print(f"[Error]: Query failed: {e}")
            import traceback
            traceback.print_exc()
            return f"[Error]: Query failed: {str(e)}"

    def force_rebuild_index(self, code_dirs, urls):
        """Force rebuild the index from scratch"""
        print("[Info]: Force rebuilding index...")
        if os.path.exists(self.knowledge_index_dir):
            import shutil
            shutil.rmtree(self.knowledge_index_dir)

        self.index = None
        self.query_engine = None
        self.build_index(code_dirs, urls)

    def retrive_document_chunks(self, query_str, top_k=2):
        """Retrieve documents based on a query"""
        if not self.query_engine:
            raise ValueError("Index not built. Call build_index() first.")

        try:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            response = retriever.retrieve(query_str)
            return response
        except Exception as e:
            print(f"[Error]: Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return f"[Error]: Retrieval failed: {str(e)}"