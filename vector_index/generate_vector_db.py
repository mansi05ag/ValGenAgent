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
from llama_index.core.node_parser import CodeSplitter, HierarchicalNodeParser
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser

class KnowledgeBase:
    """Knowledge base using LlamaIndex for document retrieval and querying"""

    def __init__(
        self,
        api_key,
        embed_base_url,
        llm_base_url,
        model_name="gpt-4o",
        knowledge_index_dir="./index_db",
        embedding_model="text-embedding-ada-002",
        embedding_dim=1536
    ):
        self.api_key = api_key
        self.embed_base_url = embed_base_url
        self.llm_base_url = llm_base_url
        self.model_name = model_name
        self.knowledge_index_dir = knowledge_index_dir
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

    def load_documents(self, code_dirs: str, urls:list[str])->list[Document]:
        """ load documents from code directories and URLs """
        try:
            all_documents = []

            if not os.path.exists(code_dirs):
                raise FileNotFoundError(f"inpur directory '{code_dirs}' does not exist.")

            print(f"[Info]: Loading documents from {code_dirs}")
            reader = SimpleDirectoryReader(code_dirs, recursive=True)
            docs = reader.load_data(num_workers=8)
            all_documents.extend(docs)
            print(f"[Info]: Loaded {len(docs)} documents from {code_dirs}")

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
            return all_documents

        except Exception as e:
            print(f"[Error]: Failed to load documents: {e}")
            import traceback
            traceback.print_exc()
            raise

    def parse_documents_by_type(self, docs: list[Document]):
        # Documents objects are converted into nodes (or a small meaningful chunk) based on their type.
        # Nodes are what's actually stored, embedded, and retrieved.
        # Supported languages: https://github.com/Goldziher/tree-sitter-language-pack?tab=readme-ov-file#available-languages

        doc_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 512])  # multiple granularities

        nodes = []
        for doc in docs:
            try:
                if "\t" in doc.text:
                    doc.set_content(doc.text.replace("\t", "    "))

                file_name = doc.metadata.get("file_name", "")

                if file_name.endswith(".py"):
                    parser = CodeHierarchyNodeParser(
                        language="python",
                        code_splitter=CodeSplitter(language="python", chunk_lines=1000, max_chars=2000)
                    )
                elif file_name.endswith(".cpp"):
                    parser = CodeHierarchyNodeParser(
                        language="cpp",
                        code_splitter=CodeSplitter(language="cpp", chunk_lines=30, max_chars=2000)
                    )
                elif file_name.endswith(".c"):
                    parser = CodeHierarchyNodeParser(
                        language="c",
                        code_splitter=CodeSplitter(language="c", chunk_lines=30, max_chars=2000)
                    )
                elif file_name.endswith(".asm"):
                    parser = CodeHierarchyNodeParser(
                        language="asm",
                        code_splitter=CodeSplitter(language="asm", chunk_lines=20, max_chars=2000)
                    )
                else:
                    parser = doc_parser  # fallback parser

                nodes += parser.get_nodes_from_documents([doc])

            except Exception as e:
                failed_docs.append({"file": file_name, "error": str(e)})
                # optionally log
                print(f"Failed to parse {file_name}: {e}")

        # Replace None relationships with empty lists -> The CodeHierarchyNodeParser if there is no next or previous relationships sets it to none but expected to be empty list.(seeing error without this)
        for node in nodes:
             if hasattr(node, "relationships"):
                 node.relationships.update({k: [] for k, v in node.relationships.items() if v is None})

        print(f"[Info]: Parsed {len(nodes)} nodes from {len(docs)} documents")
        return nodes

    def create_vector_index_from_nodes(self, nodes, persist_dir=None):
        try:
            # Check if index already exists
            if os.path.exists(persist_dir):
                try:
                    print("[Info]: Loading existing vector index...")
                    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                    self.index = load_index_from_storage(
                        storage_context,
                        embed_model=self.embed_model
                    )
                    return

                except Exception as e:
                    print(f"[Warning]: Failed to load existing vector index: {e}")
                    print("[Info]: Building vector index freshly ...")
                    import shutil
                    shutil.rmtree(persist_dir, ignore_errors=True)
                    self.index = VectorStoreIndex(
                        nodes,
                        embed_model=self.embed_model,
                        show_progress=True
                    )
                    print(f"[Info]: Saving freshly created vector index to {persist_dir}")
                    self.index.storage_context.persist(persist_dir=persist_dir)
                    return

            # Create index using simple storage
            print("[Info]: Building vector index...")
            self.index = VectorStoreIndex(
                nodes,
                embed_model=self.embed_model,
                show_progress=True
            )
            print(f"[Info]: Saving vector index to {persist_dir}")
            self.index.storage_context.persist(persist_dir=persist_dir)
        except Exception as e:
            print(f"[Error]: Failed to build/load vector index: {e}")
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
            import traceback
            traceback.print_exc()
            return f"[Error]: Query failed: {str(e)}"

    def retrive_document_chunks(self, query_str, top_k=5):
        """Retrieve documents based on a query"""
        if not self.query_engine:
            raise ValueError("Index not built. Build vector index first.")

        try:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            response = retriever.retrieve(query_str)
            return response
        except Exception as e:
            print(f"[Error]: Retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return f"[Error]: Retrieval failed: {str(e)}"

    def build_index(self, input_dirs=None, public_urls_file=None):
        """Build the knowledge base index from code directories and URLs"""
        try:
            # Check if index already exists
            if os.path.exists(self.knowledge_index_dir):
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
            print(f"[Error]: Failed to load existing vector index: {e}")
            print(f"[Info]: Deleting existing index and rebuilding...")
            import shutil
            shutil.rmtree(self.knowledge_index_dir)

        print("[Info]: Building new vector index...")

        if not input_dirs and not public_urls_file:
            raise ValueError("No input directories or URLs provided to build index.")

        # check if public_urls_file exists
        if public_urls_file and not os.path.exists(public_urls_file):
            raise FileNotFoundError(f"Public URLs file not found: {public_urls_file}")

        urls = open(public_urls_file, "r", encoding="utf-8").read()
        urls = [url.strip() for url in urls.split("\n") if url.strip()]
        # ignore if line start with comment
        for i, url in enumerate(urls):
            if url.startswith("#"):
                urls.pop(i)

        # Load documents
        docs = self.load_documents(input_dirs, urls)
        if not docs:
            print("[Warning]: No documents loaded. Cannot build index.")
            return

        # Parse documents into nodes
        nodes = self.parse_documents_by_type(docs=docs)

        # Create vector index from nodes
        self.create_vector_index_from_nodes(nodes, persist_dir=self.knowledge_index_dir)

        # Create query engine
        print("[Info]: Creating query engine...")
        self.query_engine = self.index.as_query_engine(llm=self.llm)