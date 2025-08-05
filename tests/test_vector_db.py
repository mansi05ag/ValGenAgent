import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from dotenv import load_dotenv
from vector_index.generate_vector_db import KnowledgeBase
from utils.openai_api_key_utils import get_openai_api_key

from utils.openai_endpoints import (
    OPENAI_BASE_URL,
    EMBEDDING_BASE_URL,
    INFERENCE_BASE_URL,
    AUTH_BASE_URL,
)

CODE_DIR="../input_dirs"
PUCLIC_URLS_FILE= "../input_dirs/public_urls.txt"

if __name__ == "__main__":
    load_dotenv()

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set the OPENAI_API_KEY environment variable. Or use the .env file.")

    api_key = get_openai_api_key()
    os.environ["OPENAI_API_BASE"] = EMBEDDING_BASE_URL
    kb = KnowledgeBase(
        api_key=api_key,
        embed_base_url = EMBEDDING_BASE_URL,
        llm_base_url = INFERENCE_BASE_URL,
        knowledge_index_dir="./test_index_db"
    )

    # check if the code directory exists
    if not os.path.exists(CODE_DIR):
        raise FileNotFoundError(f"The code directory '{CODE_DIR}' does not exist.")

    kb.build_index(
        input_dirs=CODE_DIR,
        public_urls_file=PUCLIC_URLS_FILE,
    )
    print("[Info]: vector undex built successfully.")

    query_str = "Write test for Collective API all_reduce based on the code context for hpu device"
    print(f"Querying for: {query_str}")
    response = kb.query(query_str)
    print(response)
    doc_chuks = kb.retrive_document_chunks("all reduce PyTorch Collective API test cases")
    print("Retrieved document chunk:")
    print(doc_chuks)
    print("index built and queried successfully.")
