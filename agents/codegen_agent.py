import os
import json
import re
import sys
import argparse
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from docx import Document
from dotenv import load_dotenv
import autogen
from autogen.coding import LocalCommandLineCodeExecutor

# Add the parent directory to sys.path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.expected_patterns import LANGUAGE_PATTERNS
from utils.openai_endpoints import (
    EMBEDDING_BASE_URL,
    INFERENCE_BASE_URL,
    MODEL_INFERENCE
)

# Conditional import for vector index to handle missing dependencies
try:
    from vector_index.generate_vector_db import KnowledgeBase
    VECTOR_DB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vector database not available due to missing dependencies: {e}")
    VECTOR_DB_AVAILABLE = False
    KnowledgeBase = None

# Load environment variables
load_dotenv()

# Configure autogen for Intel's internal API
config_list = [
    {
        "model": MODEL_INFERENCE,
        "base_url": INFERENCE_BASE_URL,
        "api_type": "openai",
        "max_tokens": 5000
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.1,
}


INPUT_DIR = 'input_dirs'
URLS_FILE = f"{INPUT_DIR}/public_urls.txt"
os.environ["OPENAI_API_BASE"] = EMBEDDING_BASE_URL



@dataclass
class TestCase:
    title: str
    description: str
    steps: List[str]
    expected_results: str
    implementation_file: Optional[str] = None
    data_types: List[str] = field(default_factory=list)

class TestPlanParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.is_json = file_path.endswith('.json')
        if not self.is_json:
            self.document = Document(file_path)

    def extract_test_cases(self) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        """Extract test cases from the test plan file.

        Returns:
            tuple: (success: bool, test_cases: List[Dict], implementation_files: List[str])
        """
        try:
            if self.is_json:
                return self._extract_from_json()
            else:
                return self._extract_from_docx()
        except Exception as e:
            print(f"Error extracting test cases: {e}")
            return False, [], []

    def _extract_from_json(self) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        try:
            with open(self.file_path, 'r') as f:
                test_plan = json.load(f)

            test_cases = []
            implementation_files = set()

            # Handle different JSON structures
            if 'tests' in test_plan:
                tests_section = test_plan['tests']

                # Check if tests is a list (array of test categories)
                if isinstance(tests_section, list):
                    for test_category in tests_section:
                        implementation_file = test_category.get('implementation_file')
                        if implementation_file:
                            implementation_files.add(implementation_file)

                        # Extract test cases and add implementation file info to each
                        for test_case in test_category.get('test_cases', []):
                            enhanced_test_case = test_case.copy()
                            enhanced_test_case['implementation_file'] = implementation_file
                            enhanced_test_case['test_category'] = test_category.get('test_category', '')

                            # Map test_id to id for compatibility
                            if 'test_id' in enhanced_test_case and 'id' not in enhanced_test_case:
                                enhanced_test_case['id'] = enhanced_test_case['test_id']

                            test_cases.append(enhanced_test_case)

                # Check if it's the new single test category format (tests as object)
                elif isinstance(tests_section, dict) and 'test_category' in tests_section and 'implementation_file' in tests_section:
                    # New format: single test category with implementation file
                    implementation_file = tests_section.get('implementation_file')
                    if implementation_file:
                        implementation_files.add(implementation_file)

                    # Extract test cases and add implementation file info to each
                    for test_case in tests_section.get('test_cases', []):
                        # Create a copy of the test case and add implementation file
                        enhanced_test_case = test_case.copy()
                        enhanced_test_case['implementation_file'] = implementation_file
                        enhanced_test_case['test_category'] = tests_section.get('test_category', '')

                        # Map test_id to id for compatibility
                        if 'test_id' in enhanced_test_case and 'id' not in enhanced_test_case:
                            enhanced_test_case['id'] = enhanced_test_case['test_id']

                        test_cases.append(enhanced_test_case)

            # Handle the old format with test_categories directly
            elif 'test_categories' in test_plan:
                for test_category in test_plan['test_categories']:
                    if test_category.get('implementation_file'):
                        implementation_files.add(test_category['implementation_file'])
                        for test_case in test_category.get('test_cases', []):
                            test_cases.append(test_case)

            # Handle simple format with direct test_cases (like the original pytorch_collective_operations format)
            elif 'test_cases' in test_plan:
                for test_case in test_plan['test_cases']:
                    # For this format, we'll create a default implementation file name
                    test_category = test_plan.get('test_category', 'default_tests')
                    impl_file = f"test_{test_category.lower().replace(' ', '_').replace('.', '_')}.py"

                    enhanced_test_case = test_case.copy()
                    enhanced_test_case['implementation_file'] = impl_file
                    enhanced_test_case['test_category'] = test_category

                    # Map test_id to id for compatibility
                    if 'test_id' in enhanced_test_case and 'id' not in enhanced_test_case:
                        enhanced_test_case['id'] = enhanced_test_case['test_id']

                    test_cases.append(enhanced_test_case)
                    implementation_files.add(impl_file)

            return True, test_cases, list(implementation_files)

        except Exception as e:
            print(f"Error reading JSON test plan: {e}")
            import traceback
            traceback.print_exc()
            return False, [], []

    def _extract_from_docx(self) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        try:
            test_cases = []
            implementation_files = set()
            current_test_case = {}
            for paragraph in self.document.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                if 'Implementation file:' in text:
                    file_name = text.split('Implementation file:')[1].strip()
                    if file_name.endswith('.py'):
                        implementation_files.add(file_name)
                        if current_test_case:
                            current_test_case['implementation_file'] = file_name
                if re.match(r'^(Test Case|TC)\s*\d+:', text, re.IGNORECASE):
                    if current_test_case:
                        test_cases.append(current_test_case)
                    current_test_case = self._init_test_case(text)
                elif current_test_case:
                    self._update_test_case(current_test_case, text)
            if current_test_case:
                test_cases.append(current_test_case)
            return True, test_cases, list(implementation_files)
        except Exception as e:
            print(f"Error extracting from DOCX: {e}")
            return False, [], []

    def _init_test_case(self, text: str) -> Dict[str, Any]:
        return {
            'title': text,
            'description': '',
            'steps': [],
            'expected_results': '',
            'implementation_file': None,
            'data_types': []
        }

    def _update_test_case(self, test_case: Dict[str, Any], text: str) -> None:
        if text.lower().startswith('steps:'):
            test_case['steps'] = []
        elif text.lower().startswith('description:'):
            test_case['description'] = text.split(':', 1)[1].strip()
        elif text.lower().startswith('expected result:') or text.lower().startswith('expected_results:'):
            test_case['expected_results'] = text.split(':', 1)[1].strip()
        elif text.lower().startswith(('data types:', 'data_types:')):
            test_case['data_types'] = [dt.strip() for dt in re.split(r'[,;\s]+', text.split(':', 1)[1].strip()) if dt.strip()]
        elif test_case.get('steps') is not None:
            test_case['steps'].append(text)
        else:
            test_case['description'] += text + '\n'

# --- Multi-Agent System ---

class MessageLogger:
    def __init__(self):
        self.messages = []
    def log(self, sender, message):
        print(f"[{sender}] {message}")
        self.messages.append((sender, message))
    def get_log(self):
        return self.messages

# Agent 3: Smart Test Execution Agent
class TestRunnerUserProxy(autogen.UserProxyAgent):
    def __init__(self, logger, output_dir="generated_tests"):
        # The executor will only run code received in conversation
        self.executor = LocalCommandLineCodeExecutor(
            timeout=120,
            work_dir=output_dir,
        )

        super().__init__(
            name="TestExecutionProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=50,
            code_execution_config={"executor": self.executor},
        )

        self.logger = logger
        self.output_dir = output_dir

# Function to save code to a file.
# The function is registered with the UserProxyAgent to handle code saving requests
def save_code_to_file(code: str, filename: str, directory: str) -> str:
    """Save code to a file in the specified directory."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)

    return f"Code saved successfully to {filepath}"


class ContextManagedGroupChat(autogen.GroupChat):
    """GroupChat with automatic context management"""

    def __init__(self, agents, messages, max_round, max_context_messages=25, logger=None):
        super().__init__(agents=agents, messages=messages, max_round=max_round)
        self.max_context_messages = max_context_messages
        self.logger = logger

    def append(self, message, speaker):
        """Override append to manage context automatically"""
        # Call parent's append method with correct signature
        super().append(message, speaker)

        # Trigger context management when approaching limit
        if len(self.messages) > self.max_context_messages * 0.9:
            if self.logger:
                self.logger.log("GroupChat", f"Auto-managing context at {len(self.messages)} messages")
            self._auto_manage_context()

    def _auto_manage_context(self):
        """Automatically manage context using the same strategy as the orchestrator"""
        if len(self.messages) <= self.max_context_messages:
            return

        # Keep initial message
        initial_message = self.messages[0] if self.messages else None
        if not initial_message:
            return

        # Find important messages
        important_keywords = [
            'APPROVED FOR EXECUTION', 'CODE IS READY FOR EXECUTION', 'APPROVE THIS CODE FOR TESTING',
            'FAILED:', 'SUCCESS:', 'ERROR:', 'filename:', 'def test_', 'import pytest',
            'subprocess.run', 'if __name__ == "__main__":', 'Exit code:', 'Test execution'
        ]

        important_messages = []
        for msg in self.messages[1:]:
            content = msg.get('content', '')
            if any(keyword in content for keyword in important_keywords) or '```python' in content or '```bash' in content:
                important_messages.append(msg)

        # Keep recent messages
        recent_count = min(8, self.max_context_messages // 4)
        recent_messages = self.messages[-recent_count:]

        # Combine and deduplicate
        managed_messages = [initial_message]
        seen_contents = {initial_message.get('content', '')[:100]}

        for msg in important_messages + recent_messages:
            content_key = msg.get('content', '')[:100]
            if content_key not in seen_contents:
                managed_messages.append(msg)
                seen_contents.add(content_key)

        # Final limit check
        if len(managed_messages) > self.max_context_messages:
            managed_messages = [managed_messages[0]] + managed_messages[-(self.max_context_messages-1):]

        self.messages = managed_messages
        if self.logger:
            self.logger.log("GroupChat", f"Context auto-managed: reduced to {len(self.messages)} messages")

class MultiAgentTestOrchestrator:
    def __init__(self, output_dir: str,
                 max_retries: int = 2,
                 max_context_messages: int = 25,
                 execute_tests: bool = True,
                 code_agent_prompt: str = "",
                 review_agent_prompt: str = "",
                 test_coordinator_prompt: str = ""):
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.max_context_messages = max_context_messages
        self.logger = MessageLogger()
        self.kb = None # Knowledge base instance if available
        self.execute_tests = execute_tests
        self.code_agent_prompt = code_agent_prompt
        self.review_agent_prompt = review_agent_prompt
        self.test_coordinator_prompt = test_coordinator_prompt

        # check if source code dir exists
        if not os.path.exists(INPUT_DIR):
            raise FileNotFoundError(f"The source code directory '{INPUT_DIR}' does not exist.")

        # Initialize the agents
        # Create the code generation agent
        self.codegen_agent = autogen.AssistantAgent(
            name="TestGenerationAgent",
            llm_config=llm_config,
            system_message=self.code_agent_prompt)

        # Create the review agent
        self.review_agent = autogen.AssistantAgent(
            name="TestCodeReviewAgent",
            llm_config=llm_config,
            system_message=self.review_agent_prompt)

        # Create a runner agent to execute tests
        if self.execute_tests:
            # Full execution capability when tests should be executed
            self.runner_agent = autogen.UserProxyAgent(
                name="TestExecutionProxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=50,
                code_execution_config={"executor": LocalCommandLineCodeExecutor(
                                                    timeout=120,
                                                    work_dir=output_dir)
                },
            )
        else:
            self.runner_agent = autogen.ConversableAgent(
                name="TestFileSaveAgent",
                system_message="""You are a file manager. Your job is to save code to files when requested.
                When you receive reviewed code, save it to the specified directory with an appropriate filename.""",
                llm_config={"config_list": config_list},
                human_input_mode="NEVER",
                max_consecutive_auto_reply=2
            )

            self.runner_agent.register_for_execution(name="save_code")(save_code_to_file)
            self.runner_agent.register_for_llm(name="save_code", description=f"Save code to a file to the path mentioned in {output_dir}. Do not change/modify the path where the output files should be saved. the Directory value when making the tool call should always be the value of {output_dir}")(save_code_to_file)

        # Create a coordinator agent to manage the conversation
        self.coordinator = autogen.UserProxyAgent(
            name="TestCoordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,  # Allow coordinator to participate in conversation
            code_execution_config=False,
            system_message="Coordinate the test generation and execution process between agents."
        )

        # Set up GroupChat - always include runner_agent but with different roles
        self.group_chat = ContextManagedGroupChat(
            agents=[self.coordinator, self.codegen_agent, self.review_agent, self.runner_agent],
            messages=[],
            max_round=50,  # Allow enough rounds for iterations
            max_context_messages=self.max_context_messages,
            logger=self.logger
        )
        # GroupChat manager with custom speaker selection logic
        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=llm_config,
            system_message=self.test_coordinator_prompt,
        )

        os.makedirs(output_dir, exist_ok=True)

        # Initialize knowledge base only if available
        if VECTOR_DB_AVAILABLE and KnowledgeBase:
            self.kb = KnowledgeBase(
                api_key=os.getenv("OPENAI_API_KEY"),
                embed_base_url=EMBEDDING_BASE_URL,
                llm_base_url=INFERENCE_BASE_URL,
                model_name=MODEL_INFERENCE,
            )

        kb_success = self.build_knowledge_base(
            input_dirs=INPUT_DIR,
            urls=URLS_FILE
        )
        if not kb_success:
            self.logger.log("Orchestrator", "Warning: Knowledge base initialization failed, proceeding without it")

    def orchestrate_test_generation(self, test_plan_path: str):
        """Main orchestration method using GroupChat for natural agent communication"""
        self.logger.log("Orchestrator", f"Starting multi-agent test generation from {test_plan_path}")

        # Parse test plan
        parser = TestPlanParser(test_plan_path)
        success, test_cases, implementation_files = parser.extract_test_cases()

        if not success:
            self.logger.log("Orchestrator", "ERROR: Failed to parse test plan")
            return False

        if not test_cases:
            self.logger.log("Orchestrator", "ERROR: No test cases found in the test plan")
            return False

        self.logger.log("Orchestrator", f"Found {len(test_cases)} test cases across {len(implementation_files)} files")
        self.logger.log("Orchestrator", f"Expected implementation files: {implementation_files}")

        # Process each implementation file using GroupChat
        successful_files = []
        failed_files = []

        for impl_file in implementation_files:
            if self._process_implementation_file_with_groupchat(impl_file, test_cases):
                successful_files.append(impl_file)
            else:
                failed_files.append(impl_file)

        # Check if all expected files were generated successfully
        all_files_generated = self._validate_all_files_generated(implementation_files, successful_files, failed_files)

        # If execute_tests is False, skip the execution step but still validate file generation
        if not self.execute_tests:
            self.logger.log("Orchestrator", "Skipping test execution as per configuration.")
            return all_files_generated

        # Summary
        self.logger.log("Orchestrator", f"Successfully generated tests for {len(successful_files)}/{len(implementation_files)} files")
        return all_files_generated

    def _process_implementation_file_with_groupchat(self, impl_file: str, all_test_cases: List[Dict]) -> bool:
        """Process a single implementation file using GroupChat for dynamic agent interaction"""
        # Filter relevant test cases
        relevant_tests = [tc for tc in all_test_cases if tc.get('implementation_file') == impl_file]
        if not relevant_tests:
            self.logger.log("Orchestrator", f"WARNING: No relevant tests found for {impl_file}")
            return False

        self.logger.log("Orchestrator", f"Processing {impl_file} with {len(relevant_tests)} test cases using GroupChat")

        # Format test cases for the group chat
        test_cases_text = self._format_test_cases_for_chat(relevant_tests)

        # Create initial message to start the group chat
        execution_note = ""
        if not self.execute_tests:
            execution_note = "\n\nNOTE: Test execution is disabled. Only generate, review, and save the test code to files. Do not execute the tests."
        else:
            execution_note = "\n\nGenerate, review, save, and execute the test code."

        initial_message = f"""
            We need to generate and review tests for: {impl_file}
            {execution_note}

            Test Cases to implement:
            {test_cases_text}
            """
        try:
            # Start the group chat
            self.logger.log("Orchestrator", f"Starting GroupChat for {impl_file}")

            # Initialize a fresh group chat for this file
            self.group_chat.messages = []  # Clear previous messages

            # Get context from knowledge base if available
            test_categories_string = ' '.join({tc['test_category'] for tc in relevant_tests})
            context_input=f"Test is {test_categories_string}"
            context = ""
            if self.kb:
                try:
                    context = self.kb.retrive_document_chunks(context_input)
                    if "[Error]" in context or not context:
                        self.logger.log("Orchestrator", f"WARNING: Failed to retrieve doc chunks for {impl_file}, proceeding without context")
                        context = ""
                except Exception as e:
                    self.logger.log("Orchestrator", f"WARNING: Error retrieving doc chunks: {e}, proceeding without context")
                    context = ""
            else:
                self.logger.log("Orchestrator", f"WARNING: Knowledge base not available, proceeding without context")

            if context:
                prompt_with_context = f"Based on the following code context:\n\n{context}\n\n {initial_message}"
            else:
                prompt_with_context = initial_message

            # Start the conversation
            self.coordinator.initiate_chat(
                self.manager,
                message=prompt_with_context,
                max_turns=20  # Limit turns to prevent infinite loops
            )

            # Manage context after conversation to keep it within limits
            self._manage_context_length()

            _, ext = os.path.splitext(impl_file)
            language=ext.lstrip(".")

            # Check if we have successful test generation
            success = self._extract_success_from_chat(language)

            # Additionally verify that the specific file was actually created
            file_actually_created = self._verify_file_created(impl_file)

            # Debug logging
            self.logger.log("Orchestrator", f"SUCCESS DETECTION: Found {len(self.group_chat.messages)} messages in chat")
            self.logger.log("Orchestrator", f"Chat success detected: {success}")
            self.logger.log("Orchestrator", f"File actually created: {file_actually_created}")

            for i, msg in enumerate(self.group_chat.messages[-3:]):  # Log last 3 messages for debugging
                content_preview = str(msg.get('content', ''))[:200]  # First 200 chars
                self.logger.log("Orchestrator", f"Message {i}: {content_preview}...")

            # Final success is both chat success AND file creation
            final_success = success and file_actually_created

            if final_success:
                self.logger.log("Orchestrator", f"SUCCESS: GroupChat completed and file created for {impl_file}")
                # Save artifacts from the group chat
                artifact_saved = self._save_artifacts_from_chat(impl_file)
                if not artifact_saved:
                    self.logger.log("Orchestrator", f"Warning: Failed to save artifacts for {impl_file}")

                return True
            else:
                if success and not file_actually_created:
                    self.logger.log("Orchestrator", f"PARTIAL FAILURE: GroupChat succeeded but file {impl_file} was not created")
                elif not success and file_actually_created:
                    self.logger.log("Orchestrator", f"PARTIAL FAILURE: File {impl_file} was created but chat did not complete successfully")
                else:
                    self.logger.log("Orchestrator", f"COMPLETE FAILURE: Neither chat success nor file creation for {impl_file}")
                return False

        except Exception as e:
            self.logger.log("Orchestrator", f"ERROR: GroupChat failed for {impl_file}: {str(e)}")
            return False

    def _format_test_cases_for_chat(self, test_cases: List[Dict]) -> str:
        """Format test cases for group chat message"""
        formatted = []
        for i, tc in enumerate(test_cases, 1):
            formatted.append(f"Test Case {i}")
            formatted.append(f"Description: {tc.get('description', 'N/A')}")
            if tc.get('steps'):
                formatted.append(f"Steps: {'; '.join(tc['steps'])}")
            if tc.get('data_types'):
                formatted.append(f"Data Types: {', '.join(tc['data_types'])}")
            formatted.append("")  # Empty line for separation
        return '\n'.join(formatted)

    def _extract_success_from_chat(self, language='py') -> bool:
        """Extract success status from the group chat messages and logger for a specific language"""
        if language not in LANGUAGE_PATTERNS:
            self.logger.log("Orchestrator", f"Unsupported language: {language}")
            return False

        patterns = LANGUAGE_PATTERNS[language]
        file_extension = language

        # Check for test generation patterns
        for message in self.group_chat.messages:
            content = str(message.get('content', ''))
            for pattern in patterns['generation']:
                if re.search(pattern, content, re.IGNORECASE):
                    self.logger.log("Orchestrator", f"TEST GENERATION PATTERN MATCHED: '{pattern}' in GroupChat message")
                    return True

        # Check for file saving patterns if execute_tests is False
        if not self.execute_tests:
            for message in self.group_chat.messages:
                content = str(message.get('content', ''))
                for pattern in patterns['file_saving']:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.logger.log("Orchestrator", f"FILE SAVING PATTERN MATCHED: '{pattern}' in GroupChat message")
                        return True

        # Check for execution success patterns if execute_tests is True
        if self.execute_tests:
            for message in self.group_chat.messages:
                content = str(message.get('content', ''))
                for pattern in patterns['execution']:
                    if re.search(pattern, content, re.IGNORECASE):
                        if 'failed' not in content.lower() or 'passed' in content.lower():
                            self.logger.log("Orchestrator", f"EXECUTION SUCCESS PATTERN MATCHED: '{pattern}' in GroupChat message")
                            return True

        # Check if any files exist in output directory
        try:
            if os.path.exists(self.output_dir):
                files = []
                files.extend([f for f in os.listdir(self.output_dir) if f.endswith(f'.{file_extension}')])
                if files:
                    self.logger.log("Orchestrator", f"SUCCESS: Found {len(files)} {language} files in output directory: {files}")
                    return True
        except Exception as e:
            self.logger.log("Orchestrator", f"Error checking for {language} files: {e}")

        # Fallback: check if any meaningful conversation happened
        meaningful_messages = [msg for msg in self.group_chat.messages
                               if len(str(msg.get('content', ''))) > 100]
        if len(meaningful_messages) >= 3:
            self.logger.log("Orchestrator", f"SUCCESS: Meaningful conversation detected with {len(meaningful_messages)} substantial messages")
            return True

        self.logger.log("Orchestrator", "No success patterns found in chat messages")
        return False

    def _save_artifacts_from_chat(self, impl_file: str) -> bool:
        """Save artifacts from the group chat conversation

        Returns:
            bool: True if artifacts were saved successfully, False otherwise
        """
        try:
            base_name = os.path.splitext(impl_file)[0]

            # Save the entire conversation log
            chat_log_file = os.path.join(self.output_dir, f"{base_name}_chat_log.txt")
            with open(chat_log_file, 'w') as f:
                f.write("=== GROUP CHAT CONVERSATION LOG ===\n\n")
                for i, message in enumerate(self.group_chat.messages):
                    f.write(f"Message {i+1} - {message.get('name', 'Unknown')}:\n")
                    f.write(f"{message.get('content', '')}\n")
                    f.write("-" * 50 + "\n")

            self.logger.log("Orchestrator", f"Saved chat log for {impl_file}")
            return True

        except Exception as e:
            self.logger.log("Orchestrator", f"Error saving artifacts for {impl_file}: {e}")
            return False

    def _manage_context_length(self):
        """Manage GroupChat context to prevent token overflow while preserving important information"""
        if len(self.group_chat.messages) <= self.max_context_messages:
            return  # No management needed

        self.logger.log("Orchestrator", f"Managing context: {len(self.group_chat.messages)} messages -> target: {self.max_context_messages}")

        # Strategy: Keep initial prompt + important messages + recent messages
        if not self.group_chat.messages:
            return

        initial_message = self.group_chat.messages[0]

        # Find important messages (approvals, code blocks, errors, successes)
        important_messages = []
        important_keywords = [
            'APPROVED FOR EXECUTION', 'CODE IS READY FOR EXECUTION', 'APPROVE THIS CODE FOR TESTING',
            'FAILED:', 'SUCCESS:', 'ERROR:', 'filename:', 'def test_', 'import pytest',
            'subprocess.run', 'if __name__ == "__main__":', 'Exit code:', 'Test execution',
            '#include ', 'TEST(', 'TEST_F(', 'CU_ASSERT(', 'CU_ADD_TEST(', 'global _?main',
            'section .text', 'mov ', 'assert.h', 'CUnit/CUnit.h', 'main'
        ]

        for msg in self.group_chat.messages[1:]:
            content = msg.get('content', '')
            # Check if message contains important keywords
            if any(keyword in content for keyword in important_keywords):
                important_messages.append(msg)
            # Also keep messages with code blocks
            elif any(code_block in content for code_block in ['```python', '```cpp', '```c', '```asm', '```bash']):
                important_messages.append(msg)

        # Keep most recent messages (last 8-10 messages are usually most relevant)
        recent_count = min(10, self.max_context_messages // 3)
        recent_messages = self.group_chat.messages[-recent_count:]

        # Combine and deduplicate while preserving order
        managed_messages = [initial_message]
        seen_contents = {initial_message.get('content', '')[:100]}  # Use first 100 chars as key

        # Add important messages first
        for msg in important_messages:
            content_key = msg.get('content', '')[:100]
            if content_key not in seen_contents:
                managed_messages.append(msg)
                seen_contents.add(content_key)

        # Add recent messages
        for msg in recent_messages:
            content_key = msg.get('content', '')[:100]
            if content_key not in seen_contents:
                managed_messages.append(msg)
                seen_contents.add(content_key)

        # If still too many, keep only most recent within limit
        if len(managed_messages) > self.max_context_messages:
            managed_messages = [managed_messages[0]] + managed_messages[-(self.max_context_messages-1):]

        # Update the group chat messages
        self.group_chat.messages = managed_messages
        self.logger.log("Orchestrator", f"Context managed: reduced to {len(self.group_chat.messages)} messages")

    def _validate_all_files_generated(self, expected_files: List[str], successful_files: List[str], failed_files: List[str]) -> bool:
        """Validate that all expected files were generated successfully and provide detailed reporting"""

        # Check which files actually exist in the output directory
        actually_generated_files = []
        missing_files = []

        try:
            if os.path.exists(self.output_dir):
                existing_files = set(os.listdir(self.output_dir))

                for expected_file in expected_files:
                    if expected_file in existing_files:
                        actually_generated_files.append(expected_file)
                    else:
                        missing_files.append(expected_file)
            else:
                missing_files = expected_files.copy()
        except Exception as e:
            self.logger.log("Orchestrator", f"Error checking output directory: {e}")
            missing_files = expected_files.copy()

        # Print detailed file generation report
        self.logger.log("Orchestrator", "===== FILE GENERATION REPORT ======")
        self.logger.log("Orchestrator", f"Expected files: {len(expected_files)}")
        self.logger.log("Orchestrator", f"Successfully processed: {len(successful_files)}")
        self.logger.log("Orchestrator", f"Actually generated: {len(actually_generated_files)}")

        if actually_generated_files:
            self.logger.log("Orchestrator", f"Generated files: {actually_generated_files}")

        if missing_files:
            self.logger.log("Orchestrator", f"Missing files: {missing_files}")

        if failed_files:
            self.logger.log("Orchestrator", f"Failed to process: {failed_files}")

        # Determine overall success
        all_files_generated = len(missing_files) == 0 and len(actually_generated_files) == len(expected_files)

        if all_files_generated:
            self.logger.log("Orchestrator", "SUCCESS: All expected files were generated successfully!")
        else:
            self.logger.log("Orchestrator", f"PARTIAL SUCCESS: Only {len(actually_generated_files)}/{len(expected_files)} files were generated")
            self.logger.log("Orchestrator", "This is considered a failure as all files must be generated")

        self.logger.log("Orchestrator", "===== END REPORT =====")

        return all_files_generated

    def _verify_file_created(self, expected_file: str) -> bool:
        """Verify that a specific file was actually created in the output directory with meaningful content"""
        try:
            if not os.path.exists(self.output_dir):
                return False

            extensions_and_indicators = {
                '.py': ['def test_', 'import pytest', 'class Test'],
                '.cpp': ['TEST(', 'ASSERT_', 'EXPECT_','TEST','TEST_F','INSTANTIATE_TEST_SUITE_P','cout','iostream'],
                '.c': ['printf'],
                '.asm': ['; Test', '; Assert', '; Expect']  # Assembly comments might indicate test content
            }

            file_path = os.path.join(self.output_dir, expected_file)
            exists = os.path.exists(file_path)

            if exists:
                # Check if file has meaningful content (not empty)
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                   # Also check if it contains test-like content
                    _, ext = os.path.splitext(expected_file)

                    # Get the test indicators for the file extension
                    indicators = extensions_and_indicators.get(ext, [])

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        # Look for test indicators
                        has_test_content = any(indicator in content for indicator in indicators)

                        if has_test_content:
                            self.logger.log("Orchestrator", f"File verified with test content: {expected_file} ({file_size} bytes)")
                            return True
                        else:
                            self.logger.log("Orchestrator", f"File exists but lacks test content: {expected_file}")
                            return False

                    except Exception as read_error:
                        self.logger.log("Orchestrator", f"File exists but couldn't read content: {expected_file} - {read_error}")
                        # Still consider it valid if file exists and has size
                        return True
                else:
                    self.logger.log("Orchestrator", f"File exists but is empty: {expected_file}")
                    return False
            else:
                self.logger.log("Orchestrator", f"File not found: {expected_file}")
                return False

        except Exception as e:
            self.logger.log("Orchestrator", f"Error verifying file {expected_file}: {e}")
            return False

    def _check_generated_test_files(self) -> bool:
        """Check if test files were actually generated and executed successfully"""
        try:
            # Check if any Python test files exist in the output directory
            if not os.path.exists(self.output_dir):
                return False

            test_files = []
            print(f"{self.output_dir} {os.listdir(self.output_dir)}")
            for file in os.listdir(self.output_dir):
                print(f"filename: {file}")
                if file.endswith('.py') and ('test_' in file or '_test' in file):
                    test_files.append(os.path.join(self.output_dir, file))

            if not test_files:
                self.logger.log("Orchestrator", "No test files found in output directory")
                return False

            # Check if any test files have recent modification time (within last few minutes)
            import time
            current_time = time.time()
            recent_files = []

            for test_file in test_files:
                if os.path.exists(test_file):
                    file_mtime = os.path.getmtime(test_file)
                    # Consider files modified within the last 10 minutes as "recent"
                    if current_time - file_mtime < 600:  # 600 seconds = 10 minutes
                        recent_files.append(test_file)

            if recent_files:
                self.logger.log("Orchestrator", f"Found {len(recent_files)} recently generated test file(s): {recent_files}")
                return True

            self.logger.log("Orchestrator", f"Found {len(test_files)} test files but none are recent")
            return False

        except Exception as e:
            self.logger.log("Orchestrator", f"Error checking generated test files: {str(e)}")
            return False

    def build_knowledge_base(self, input_dirs, urls):
        """Build the knowledge base

        Args:
            code_dirs (list): List of directories to index
            urls (list): List of URLs to index

        Returns:
            bool: True if knowledge base was built successfully, False otherwise
        """
        if self.kb:
            try:
                self.kb.build_index(input_dirs, urls)
                return True
            except Exception as e:
                print(f"Warning: Failed to build knowledge base: {e}")
                return False
        else:
            print("Warning: Cannot build knowledge base - dependencies not available")
            return False

def run_test_automation(test_plan_path: str,
                        output_dir: str = "generated_tests",
                       max_retries: int = 20,
                       max_context: int = 25,
                       verbose: bool = False,
                       execute_tests: bool = True,
                       code_agent_prompt: str = "",
                       review_agent_prompt: str = "",
                       test_coordinator_prompt: str = "") -> bool:
    """
    Run the multi-agent test automation system.

    Args:
        test_plan_path: Path to test plan document (JSON or DOCX format)
        output_dir: Output directory for generated tests
        max_retries: Maximum retries for code correction
        max_context: Maximum context messages in GroupChat
        verbose: Enable verbose output
        execute_tests: Whether to execute the generated tests

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate input file
        if not os.path.exists(test_plan_path):
            print(f"ERROR: Test plan file '{test_plan_path}' not found")
            return False

        if not test_plan_path.endswith(('.json', '.docx')):
            print("ERROR: Test plan must be a JSON or DOCX file")
            return False

        if verbose:
            print("Multi-Agent Test Automation System")
            print("=" * 40)
            print(f"Test Plan: {test_plan_path}")
            print(f"Output Directory: {output_dir}")
            print(f"Max Retries: {max_retries}")
            print(f"Max Context Messages: {max_context}")
            print(f"Execute Tests: {'Yes' if execute_tests else 'No'}")
            print("=" * 40)

        # Initialize and run the multi-agent orchestrator
        orchestrator = MultiAgentTestOrchestrator(
            output_dir=output_dir,
            max_retries=max_retries,
            max_context_messages=max_context,
            execute_tests=execute_tests,
            code_agent_prompt=code_agent_prompt,
            review_agent_prompt=review_agent_prompt,
            test_coordinator_prompt=test_coordinator_prompt
        )

        success = orchestrator.orchestrate_test_generation(test_plan_path)

        if success:
            if verbose:
                print("\n" + "="*50)
                print("COMPLETE SUCCESS: All expected test files generated!")
                print(f"Generated tests are available in: {output_dir}")
                print("="*50)
            else:
                print("\nSUCCESS: All expected test files generated successfully!")
        else:
            if verbose:
                print("\n" + "="*50)
                print("FAILURE: Not all expected test files were generated!")
                print("Check the logs above for details on which files failed.")
                print("="*50)
            else:
                print("\nFAILURE: Test generation incomplete - not all files generated!")

        return success

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

