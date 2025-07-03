# ValCodeGenAgent

Validation Code Generator Agent: An agentic application (based on AutoGen) for automating the validation code generation for PyTorch Collective APIs.

## Complete Pipeline Flow:
+-------------------+
|   Configuration   |
|  (API Keys, URLs) |
+-------------------+
          ↓
+-------------------+
| Document Loading  |
| Local Files + URLs|
+-------------------+
          ↓
+-------------------+
| Parsing & Embedding|
| Nodes + Embeddings |
+-------------------+
          ↓
+-------------------+
| Index Management  |
| Load/Create Index |
+-------------------+
          ↓
+-------------------+
|  Query Engine     |
| Vector DB + LLM   |
+-------------------+
          ↓
+-------------------+
|   Agent System    |
| TestWriter, etc.  |
+-------------------+
          ↓
+-------------------+
|    Execution      |
| Query + GroupChat |
+-------------------+

## Deatils overview
1. Configuration
    •	Load environment variables using dotenv.
    •	Define constants:
        o	API_KEY, BASE_URL, BASE_URL_EMB, MODEL, INDEX_DIR, etc.

2. Document Loading
    •	Local Files: Use SimpleDirectoryReader to load .py files recursively from the code directory.
    •	Web URLs: Use BeautifulSoupWebReader to fetch content from URLs (e.g., PyTorch documentation).

3. Parsing & Embedding
    •	Parse loaded documents into nodes using SimpleNodeParser.
    •	Generate embeddings for nodes using OpenAIEmbedding.

4. Index Management
    •	Existing Index: Load the FAISS vector store and index from INDEX_DIR if it exists.
    •	New Index: Create a new FAISS vector store and index, then persist it to INDEX_DIR.

5. Query Engine
    •	Create a query engine using VectorStoreIndex and OpenAILike LLM.

6. Agent System
    •	Define agents:
        o	TestWriter: Writes unit tests for PyTorch APIs.
        o	Reviewer: Reviews and improves the tests.
        o	UserProxy: Represents the user in the group chat.
    •	Set up a GroupChat with agents and manage it using GroupChatManager.

7. Execution
    •	Query the vector database for context using query_engine.
    •	Pass the context to agents via GroupChatManager for collaborative test generation and review.

