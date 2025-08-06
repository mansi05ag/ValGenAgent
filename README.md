
#  ValGenAgent

ValGenAgent is an **agentic framework** for automated test plan generation and execution on hardware using **RAG (Retrieval-Augmented Generation)**. It simplifies validation workflows by generating test plans from input files, refining them via RAG, executing on-device test cases, and repeating until successful.

---

##  Features

-  RAG-Based Test Plan Generation
-  Loop Until Success: Automatically retries until validation passes
-  On-Device Test Case Execution
-  Organized Result Storage in `test_results/`
-  Supports Code + Document Inputs
-  LLM + Context-Aware Retrieval Pipeline


![Alt text](./workflow-arch.jpg "Workflow Architecture")


---

##  Input Support

ValGenAgent accepts both code and document files as input for RAG context.

###  All Coding Languages are supported. These languages are using language specific parser

- `C`
- `C++`
- `Python`
- `Assembly`

Other languages are parsed using a **HierarchicalNodeParser**.

###  Supported File Formats

ValGenAgent uses [`SimpleDirectoryReader`](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/) from LlamaIndex, and supports:

- `.txt`, `.md`, `.py`, `.c`, `.cpp`, `.s`, `.json`, `.docx`, `.xlsx`, `.pptx`, etc.

---

##  Output Directory Structure

All outputs (plans, scripts, logs) are saved under `test_results/`:

```
test_results/
├── test_plan.docx
├── test_plan.json
├── generated tests/
│   ├── test_results.xml
│   └── test_operations.py
│   └── test_chat_log.txt
└── function_test_results.xlsx
```

---

##  Command-Line Usage

Get started with ValGenAgent using the terminal.

###  Initial Setup

1. **Prepare Target Hardware**  
   Ensure the device (e.g., 8-card Gaudi container) is up and running.

2. **Clone the Repository**
   ```bash
   git clone https://github.com/mansi05ag/ValGenAgent.git
   cd ValGenAgent
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

###  Prepare Input Directory

Create the `input_dirs` structure like so:

```
input_dirs/
├── code/              # Source code files (C, C++, Python, etc.)
├── docs/              # Supporting documents (e.g., .txt, .md, .pdf)
└── public_urls.txt    # (Optional) Public URLs for RAG context
```

---

###  Create Feature Input File

This is the instruction/spec file used to generate test cases.
Supported formats include: `.json`, `.docx`, `.pptx`, `.xlsx`, `.txt`

---

###  Run the Agent

```bash
python test_runner.py --feature-input <path_to_input_file> --output-dir test_results
```

---

###  Multi-Format Input Examples

```bash
python test_runner.py --feature-input input_file/collective_feature.json --output-dir test_results
python test_runner.py --feature-input requirements.docx --output-dir test_results
python test_runner.py --feature-input feature_spec.pptx --output-dir test_results
python test_runner.py --feature-input requirements.xlsx --output-dir test_results
python test_runner.py --feature-input requirements.txt --output-dir test_results
```

---

###  List All Supported Formats

```bash
python test_runner.py --list-formats
```

---

##  Use Cases

This application supports various types of execution plans. Depending on the requirement, you can either:
- only create the test cases,
- just generate the test plan,
- or run the complete End-to-End workflow.

### 1. End-to-End Workflow
Run the full pipeline — plan generation, test case creation, and execution:

```bash
python test_runner.py --feature-input input_file/collective_feature.json --output-dir test_results
```

### 2. Only Test Plan Generation
Generate just the test plan based on the input file:

```bash
python test_runner.py --feature-input input_file/collective_feature.json --generate-plan-only --output-dir test_results
```

### 3. Run from Existing Test Plan
Use an existing plan to automate the test execution:

```bash
python test_runner.py --test-automation-only --test-plan path/to/plan.json --output-dir test_results
```

### 4. Generate Test Cases Only
Generate test cases without executing them:

```bash
python test_runner.py --feature-input path/to/feature_input.json --output-dir path/to/output_dir --execute-tests=false
```
Generate tests from existing test plan without executing them:

```bash
python test_runner.py --test-plan path/to/plan.json --output-dir path/to/output_dir --execute-tests=false
```
---

## Webapplication

We provide a user interface to use this tool. This help us simple through interactive ui run the commands on the device that we are targetting. For example a 8 card gaudi container.

### Steps to run the application:
1. First clone the repo on your personal vm
   ```bash
   git clone https://github.com/mansi05ag/ValGenAgent.git
   cd ValGenAgent
   ```
2. Now go inside the directory ValGenAgent_webapplication
   ```bash
   cd ValGenAgent_webapplication
   ```
3. Now we install all the requirements
   ```bash
   pip install -r requirements-app.txt
   ```
4. After that run the application
   ```bash
   python app.py
   ```
5. Access the application at the defined port 8002 and through the ui.

### Guide to use the application:
1. First you need to go inside the container or device using the command use on your vm to access the device. press connect and the container will be set up.
NOTE: the conatiner name option can be used only if u have a hlctl container with namespace qa. the command used is hlctl container exec -w <container_name> bash -n qa
2. After the setup is done, upload all the files necessary (the code and docs is expected to be a zip file here.), select the desired functionality and click on run, this will now run the application on the device. You can see all the details on the screen about the implementation.
3. After the run is successful you can download the zip file for your output.