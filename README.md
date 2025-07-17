# ValCodeGenAgent

Validation Code Generator Agent: An agentic application (based on AutoGen) for automating the validation code generation for PyTorch Collective APIs.

![Alt text](./workflow-arch.jpg "Workflow Architecture")

## Deatils overview
1. Configuration
    - Load environment variables using dotenv.
    - Define constants:
        - API_KEY, BASE_URL, BASE_URL_EMB, MODEL, INDEX_DIR, etc.

2. Document Loading
    - Local Files: Use SimpleDirectoryReader to load .py files recursively from the code directory.
    - Web URLs: Use BeautifulSoupWebReader to fetch content from URLs (e.g., PyTorch documentation).

3. Parsing & Embedding
    - Parse loaded documents into nodes using SimpleNodeParser.
    - Generate embeddings for nodes using OpenAIEmbedding.

4. Index Management
    - Create a new index db that persists in INDEX_DIR.

5. Query Engine & documents retrival
    - Create interface API for query and documents retrival.

6. Agents
    - Code Agent generates initial test code.
    - Review Agent reviews the code and provides feedback.
    - Runner/Proxy agent executes the approved code and provides execution results.
    - Coordinator agent ensures smooth communication and adherence to workflow rules.

# Usage:
    - To generate and execute test cases from json file:
        - python test_automation_agent_v3.py --test-plan ../test_plan/collectives_test_plan_debug.json --output-dir test_output_final

    - Run complete workflow (generate plan + run automation)
        - python test_workflow_runner.py --feature collectives

    - Only generate test plan
        - python test_workflow_runner.py --feature collectives --generate-plan-only

    - Only run test automation (requires existing test plan)
        - python test_workflow_runner.py --feature collectives --test-automation-only --test-plan path/to/plan.docx

    - Run automation with auto-discovery of existing test plan
        - python test_workflow_runner.py --feature collectives --test-automation-only

