
#  ValGenAgent

ValGenAgent is an **agentic framework** for automated test plan generation and execution on hardware using **RAG (Retrieval-Augmented Generation)**. It simplifies validation workflows by generating test plans from input files, refining them via RAG, generating the test cases, executing test cases on target device, and repeating until successful(or max iterations).

---

##  Features

-  RAG-Based Test Plan Generation
-  Test case generation
-  Loop Until Success(or max iterations): Automatically retries until validation passes
-  Test Case Execution on target device
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
## Input Directory Structure
All Inputs should be kept in this folder inside the respective files.
```
input_dirs/
├── code/              # Source code files (C, C++, Python, etc.)
├── docs/              # Supporting documents (e.g., .txt, .md, .pdf)
└── public_urls.txt    # Public URLs for RAG context
```

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

Put the code, docs, and public urls inside the `input_dirs`.


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
   cd webapp
   ```
3. Now we install all the requirements
   ```bash
   pip install -r requirements.txt
   ```
4. After that run the application
   for running the application we must provide a common directory accessible by both the vm and the device you are connecting to.
   ```bash
   python app.py --common_dir <path to a common directory>
   ```
5. Access the application at the defined port 8002 and through the ui.

#  How to Use the Application

Follow these steps to run the application smoothly:

---

##  Step 1: Connect to the Target Container or Device

- Click the **Connect** button on your virtual machine (VM) interface to initiate the container setup.
- This will establish a connection to the required container or device.

- If you're working with an **`hlctl` container** in the **`qa` namespace**, you can also connect manually using the following command:

  ```bash
  hlctl container exec -w <container_name> bash -n qa
  ```

  >  Replace `<container_name>` with your actual container name.

> **Note:** The *Container Name* input field is relevant **only if you're using an `hlctl` container with the `qa` namespace**.

---

##  Step 2: Upload Files and Select Functionality

- Upload all required files.
  - Your **code and documentation** should be combined into a **single `.zip` file**.
- Choose the functionality you want the application to perform from the available options.
- Click the **Run** button to start execution.

>  The application will now run on the connected container or device, and you’ll see real-time output and logs on the screen.

---

##  Step 3: Download the Output

- Once execution completes successfully, a **ZIP file** containing the **generated output or results** will be available.
- Click on the **Download** button to save the output to your system.

---