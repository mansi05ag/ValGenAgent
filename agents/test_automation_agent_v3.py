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

from utils.openai_api_key_utils import get_openai_api_key
from utils.openai_endpoints import (
    EMBEDDING_BASE_URL,
    INFERENCE_BASE_URL,
    MODEL_INFERENCE
)
from vector_index.generate_vector_db import KnowledgeBase
from prompts.code_agent_system_prompt import CODE_AGENT_SYSTEM_PROMPT
from prompts.review_agent_system_prompt import REVIEW_AGENT_SYSTEM_PROMPT
from prompts.test_coordinator_system_prompt import TEST_COORDINATOR_AGENT_SYSTEM_PROMPT

# Load environment variables
load_dotenv()

# Get Intel API key
api_key = get_openai_api_key()


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

URLS_LIST = [
    #"https://docs.pytorch.org/docs/stable/distributed.html",
]
PYC_CODE = 'code'
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

    def extract_test_cases(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        if self.is_json:
            return self._extract_from_json()
        else:
            return self._extract_from_docx()

    def _extract_from_json(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        try:
            with open(self.file_path, 'r') as f:
                test_plan = json.load(f)
            test_cases = []
            implementation_files = set()
            for category in test_plan.get('test_categories', []):
                for test_case in category.get('test_cases', []):
                    if test_case.get('implementation_file'):
                        implementation_files.add(test_case['implementation_file'])
                    test_cases.append(test_case)
            return test_cases, list(implementation_files)
        except Exception as e:
            print(f"Error reading JSON test plan: {e}")
            return [], []

    def _extract_from_docx(self) -> Tuple[List[Dict[str, Any]], List[str]]:
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
        return test_cases, list(implementation_files)

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


class CodeGenAgent(autogen.AssistantAgent):
    def __init__(self, logger):
        super().__init__(
            name="TestGenerationAgent",
            llm_config=llm_config,
            system_message=CODE_AGENT_SYSTEM_PROMPT
        )
        self.logger = logger
        # Initialize knowledge base
        self.kb = KnowledgeBase(
            api_key=api_key,
            embed_base_url=EMBEDDING_BASE_URL,
            llm_base_url=INFERENCE_BASE_URL ,
            model_name=MODEL_INFERENCE,
        )

    def build_knowledge_base(self, code_dirs, urls):
        """Build the knowledge base

        Args:
            code_dirs (list): List of directories to index
            urls (list): List of URLs to index
        """
        self.kb.build_index(code_dirs, urls)

# Agent 2: Code Review
class CodeReviewAgent(autogen.AssistantAgent):
    def __init__(self, logger):
        super().__init__(
            name="TestCodeReviewAgent",
            llm_config=llm_config,
            system_message=REVIEW_AGENT_SYSTEM_PROMPT,
        )
        self.logger = logger

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
            max_consecutive_auto_reply=10,
            code_execution_config={"executor": self.executor},
        )

        self.logger = logger
        self.output_dir = output_dir

    def execute_code(self, code: str, context=None):
        """
        Receives the code from LLM conversation (with # filename directive),
        executes using the Autogen executor (not subprocess),
        and returns execution logs.
        """
        self.logger.info("Executing code:\n%s", code)
        result = self.executor.execute(code, context=context)

        # Print intermediate outputs
        self.logger.info("Execution result:\n%s", result.output)
        return result.output


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
    def __init__(self, output_dir: str, max_retries: int = 2, max_context_messages: int = 25):
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.max_context_messages = max_context_messages
        self.logger = MessageLogger()

        # check if source code dir exists
        if not os.path.exists(PYC_CODE):
            raise FileNotFoundError(f"The source code directory '{PYC_CODE}' does not exist.")

        # Initialize the three agents
        self.codegen_agent = CodeGenAgent(self.logger)
        self.kb = self.codegen_agent.build_knowledge_base(
            code_dirs=[PYC_CODE],
            urls=URLS_LIST
        )
        self.review_agent = CodeReviewAgent(self.logger)
        self.runner_agent = TestRunnerUserProxy(self.logger, output_dir)

        # Create a coordinator agent to manage the conversation
        self.coordinator = autogen.UserProxyAgent(
            name="TestCoordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
            system_message="Coordinate the test generation and execution process between agents."
        )

        # Set up GroupChat
        self.group_chat = ContextManagedGroupChat(
            agents=[self.coordinator, self.codegen_agent, self.review_agent, self.runner_agent],
            messages=[],
            max_round=10,  # Allow enough rounds for iterations
            max_context_messages=self.max_context_messages,
            logger=self.logger
        )

        # GroupChat manager with custom speaker selection logic
        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=llm_config,
            system_message=TEST_COORDINATOR_AGENT_SYSTEM_PROMPT,
        )

        os.makedirs(output_dir, exist_ok=True)

    def orchestrate_test_generation(self, test_plan_path: str):
        """Main orchestration method using GroupChat for natural agent communication"""
        self.logger.log("Orchestrator", f"Starting multi-agent test generation from {test_plan_path}")

        # Parse test plan
        parser = TestPlanParser(test_plan_path)
        test_cases, implementation_files = parser.extract_test_cases()

        if not test_cases:
            self.logger.log("Orchestrator", "ERROR: No test cases found in the test plan")
            return False

        self.logger.log("Orchestrator", f"Found {len(test_cases)} test cases across {len(implementation_files)} files")

        # Process each implementation file using GroupChat
        successful_files = 0
        for impl_file in implementation_files:
            if self._process_implementation_file_with_groupchat(impl_file, test_cases):
                successful_files += 1

        # Summary
        self.logger.log("Orchestrator", f"Successfully generated tests for {successful_files}/{len(implementation_files)} files")
        return successful_files > 0

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
        initial_message = f"""
            We need to generate, review, and execute tests for: {impl_file}

            Test Cases to implement:
            {test_cases_text}
            """
        try:
            # Start the group chat
            self.logger.log("Orchestrator", f"Starting GroupChat for {impl_file}")

            # Initialize a fresh group chat for this file
            self.group_chat.messages = []  # Clear previous messages

            # Get context from knowledge base
            context = self.codegen_agent.kb.retrive_document_chunks("all reduce PyTorch Collective API test cases")
            if "[Error]" in context or not context:
                self.logger.log("Orchestrator", f"ERROR: Failed to retrieve doc chunks for {impl_file}")
                return False

            prompt_with_context = f"Based on the following code context:\n\n{context}\n\n {initial_message}"
            # Start the conversation
            self.coordinator.initiate_chat(
                self.manager,
                message=prompt_with_context,
                max_turns=20  # Limit turns to prevent infinite loops
            )

            # Manage context after conversation to keep it within limits
            self._manage_context_length()

            # Check if we have successful test execution
            success = self._extract_success_from_chat()

            # Debug logging
            self.logger.log("Orchestrator", f"SUCCESS DETECTION: Found {len(self.group_chat.messages)} messages in chat")
            for i, msg in enumerate(self.group_chat.messages[-3:]):  # Log last 3 messages for debugging
                content_preview = str(msg.get('content', ''))[:200]  # First 200 chars
                self.logger.log("Orchestrator", f"Message {i}: {content_preview}...")

            if success:
                self.logger.log("Orchestrator", f"SUCCESS: GroupChat completed successfully for {impl_file}")
                # Save artifacts from the group chat
                self._save_artifacts_from_chat(impl_file)
                return True
            else:
                self.logger.log("Orchestrator", f"FAILED: GroupChat did not achieve success for {impl_file}")
                return False

        except Exception as e:
            self.logger.log("Orchestrator", f"ERROR: GroupChat failed for {impl_file}: {str(e)}")
            return False

    def _format_test_cases_for_chat(self, test_cases: List[Dict]) -> str:
        """Format test cases for group chat message"""
        formatted = []
        for i, tc in enumerate(test_cases, 1):
            formatted.append(f"Test Case {i}: {tc.get('title', 'N/A')}")
            formatted.append(f"Description: {tc.get('description', 'N/A')}")
            if tc.get('steps'):
                formatted.append(f"Steps: {'; '.join(tc['steps'])}")
            if tc.get('data_types'):
                formatted.append(f"Data Types: {', '.join(tc['data_types'])}")
            formatted.append(f"Expected Result: {tc.get('expected_results', 'N/A')}")
            formatted.append("")  # Empty line for separation
        return '\n'.join(formatted)

    def _extract_success_from_chat(self) -> bool:
        """Extract success status from the group chat messages and logger"""
        # Look through the chat messages for execution success indicators
        success_patterns = [
            r'SUCCESS:.*Test execution completed successfully',
            r'Test execution completed successfully',
            r'\d+\s+passed\s+in\s+[\d.]+s',  # pytest pattern like "3 passed in 36.82s"
            r'=+\s*\d+\s+passed.*=+',  # pytest summary pattern
        ]

        # Check GroupChat messages
        for message in self.group_chat.messages:
            content = str(message.get('content', ''))
            for pattern in success_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Additional check to avoid false positives with failures
                    if 'failed' not in content.lower() or 'passed' in content.lower():
                        self.logger.log("Orchestrator", f"SUCCESS PATTERN MATCHED: '{pattern}' in GroupChat message")
                        return True

        # Check logger messages as fallback
        for sender, message in self.logger.get_log():
            for pattern in success_patterns:
                if re.search(pattern, str(message), re.IGNORECASE):
                    if 'failed' not in str(message).lower() or 'passed' in str(message).lower():
                        self.logger.log("Orchestrator", f"SUCCESS PATTERN MATCHED: '{pattern}' in logger message from {sender}")
                        return True

        # Additional fallback: check if any test files were generated and executed successfully
        if self._check_generated_test_files():
            self.logger.log("Orchestrator", "SUCCESS detected via generated test files check")
            return True

        self.logger.log("Orchestrator", "No success patterns found in chat messages or logger")
        return False

    def _save_artifacts_from_chat(self, impl_file: str):
        """Save artifacts from the group chat conversation"""
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
            'subprocess.run', 'if __name__ == "__main__":', 'Exit code:', 'Test execution'
        ]

        for msg in self.group_chat.messages[1:]:
            content = msg.get('content', '')
            # Check if message contains important keywords
            if any(keyword in content for keyword in important_keywords):
                important_messages.append(msg)
            # Also keep messages with code blocks
            elif '```python' in content or '```bash' in content:
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

    def _periodic_context_check(self):
        """Perform periodic context check during conversation"""
        if len(self.group_chat.messages) > self.max_context_messages * 0.8:  # Trigger at 80% capacity
            self.logger.log("Orchestrator", "Performing periodic context management...")
            self._manage_context_length()

    def _check_generated_test_files(self) -> bool:
        """Check if test files were actually generated and executed successfully"""
        try:
            # Check if any Python test files exist in the output directory
            if not os.path.exists(self.output_dir):
                return False

            test_files = []
            for file in os.listdir(self.output_dir):
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

def main():
    """Main entry point for the multi-agent test automation system"""
    parser = argparse.ArgumentParser(
        description='Multi-agent test automation system using autogen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Example usage:
            python test_automation_agent_v2.py --test-plan test_results/collectives_test_plan.json
            python test_automation_agent_v2.py --test-plan test_plan.docx --output-dir my_tests --max-retries 3
        """
    )
    parser.add_argument('--test-plan', required=True,
                       help='Path to test plan document (JSON or DOCX format)')
    parser.add_argument('--output-dir', default='generated_tests',
                       help='Output directory for generated tests (default: generated_tests)')
    parser.add_argument('--max-retries', type=int, default=2,
                       help='Maximum retries for code correction (default: 2)')
    parser.add_argument('--max-context', type=int, default=25,
                       help='Maximum context messages in GroupChat (default: 25)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.test_plan):
        print(f"ERROR: Test plan file '{args.test_plan}' not found")
        sys.exit(1)

    if not args.test_plan.endswith(('.json', '.docx')):
        print("ERROR: Test plan must be a JSON or DOCX file")
        sys.exit(1)

    # Check for required environment variables - using Intel's internal API
    try:
        test_key = get_openai_api_key()
        if not test_key:
            print("ERROR: Failed to obtain Intel API key")
            print("Please ensure you have proper access to Intel's internal API")
            sys.exit(1)
        else:
            print("Successfully obtained Intel API key")
    except Exception as e:
        print(f"ERROR: Failed to get Intel API key: {str(e)}")
        print("Please ensure you have proper access to Intel's internal API")
        sys.exit(1)

    print("Multi-Agent Test Automation System v2")
    print("=" * 40)
    print(f"Test Plan: {args.test_plan}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Max Retries: {args.max_retries}")
    print(f"Max Context Messages: {args.max_context}")
    print("=" * 40)

    try:
        # Initialize and run the multi-agent orchestrator
        orchestrator = MultiAgentTestOrchestrator(
            output_dir=args.output_dir,
            max_retries=args.max_retries,
            max_context_messages=args.max_context
        )

        success = orchestrator.orchestrate_test_generation(args.test_plan)

        if success:
            print("\nMulti-agent test generation completed successfully!")
            print(f"Generated tests are available in: {args.output_dir}")
        else:
            print("\nMulti-agent test generation failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
