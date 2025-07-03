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
from utils.openai_api_key_utils import get_openai_api_key
from autogen.coding import LocalCommandLineCodeExecutor

from vector_index.vector_db import KnowledgeBase

# Load environment variables
load_dotenv()

# Get Intel API key
api_key = get_openai_api_key()

# Configure autogen for Intel's internal API
config_list = [
    {
        "model": "gpt-4o",
        "base_url": "https://apis-internal.intel.com/generativeaiinference/v4",
        "api_type": "openai",
        "max_tokens": 5000
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.1,
}

BASE_URL = "https://apis-internal.intel.com/generativeaiinference/v4"
BASE_URL_EMB = "https://apis-internal.intel.com/generativeaiembedding/v2/"
URLS_LIST = [
    "https://docs.pytorch.org/docs/stable/distributed.html",
]
PYC_CODE = 'code'
os.environ["OPENAI_API_BASE"] = BASE_URL_EMB


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
            system_message="""You are a test generation agent in a collaborative team. Your role is to generate high-quality Python test code based on test case specifications.

            When you receive a request to generate test code:
            1. Analyze the test case requirements carefully
            2. Generate pytest-compatible test code with proper imports and setup
            3. Include parameterized tests when appropriate
            4. Add proper error handling and timeouts
            5. Follow Python testing best practices
            6. Present your code in ```python code blocks with a filename directive as the first line
            7. Use '# filename: test_<name>.py' at the start of code blocks to specify the filename
            8. If you receive feedback from the review agent, incorporate their suggestions and generate improved code

            **MANDATORY**: ALWAYS include a main execution block that runs the tests. Your code MUST include either:
            - A subprocess.run() call that executes pytest on the test file
            - OR an if __name__ == "__main__": block that runs the tests

            The execution block should include proper error handling and output reporting.

            Always be collaborative and responsive to feedback from other agents in the conversation.

            Example format (ALWAYS include the execution part):
            ```python
            # filename: test_example.py
            import pytest
            import subprocess
            import sys
            import os

            def run_example(rank, world_size):
                # Example test function

            def test_example():
                mp.spawn(run_example, args=(world_size), nprocs=world_size, join=True)

            # MANDATORY: Execute tests with comprehensive error handling
            if __name__ == "__main__":
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', __file__,
                    '-v', '--tb=short', '--disable-warnings', '--junitxml=test_results.xml'
                ], capture_output=True, text=True, timeout=120)

                print(f"Exit code: {result.returncode}")
                print(f"STDOUT:\\n{result.stdout}")
                if result.stderr:
                    print(f"STDERR:\\n{result.stderr}")

                if result.returncode == 0:
                    print("SUCCESS: Test execution completed successfully!")
                else:
                    print("FAILED: Test execution failed!")
            ```"""
        )
        self.logger = logger
        # Initialize knowledge base
        self.kb = KnowledgeBase(
            api_key=api_key,
            embed_base_url="https://apis-internal.intel.com/generativeaiembedding/v2/",
            llm_base_url="https://apis-internal.intel.com/generativeaiinference/v4",
            model_name="gpt-4o"
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
            system_message="""You are a code review agent specializing in Python test code review within a collaborative team.

            **IMPORTANT**: You are ONLY responsible for reviewing code, NOT generating or writing code. Your role is strictly limited to analysis and feedback.

            Your role in the GroupChat workflow:
            1. ALWAYS review any test code generated by TestGenerationAgent
            2. Provide specific, constructive feedback for improvements
            3. If code has issues, provide detailed feedback and request that TestGenerationAgent regenerate the code
            4. If code is good, give EXPLICIT APPROVAL using these exact phrases:
            - "APPROVED FOR EXECUTION"
            - "CODE IS READY FOR EXECUTION"
            - "APPROVE THIS CODE FOR TESTING"

            CRITICAL REVIEW CRITERIA - ALL MUST BE PRESENT:
            - Code quality, correctness, and best practices
            - Proper error handling and edge cases
            - Pytest compatibility and proper test structure
            - Code clarity and maintainability
            - Proper resource cleanup and timeout handling
            - Testing best practices and conventions

            **MANDATORY REQUIREMENT**: The test file MUST include a main execution block that actually runs the tests. Look for code like:
            ```python
            if __name__ == "__main__":
                # Code that executes the tests
            ```
            OR a direct subprocess.run() call that executes pytest, like:
            ```python
            result = subprocess.run([sys.executable, '-m', 'pytest', 'filename.py', ...], ...)
            ```

            **DO NOT APPROVE** any code that lacks a way to actually execute the tests. If the test file only contains test functions but no execution mechanism, provide feedback requesting the addition of a main block or pytest execution code.

            **CRITICAL**: You must NEVER generate, write, or provide code. Only provide review comments, feedback, and approval/rejection decisions. Always direct TestGenerationAgent to make the actual code changes.

            IMPORTANT: You must explicitly approve code before execution can proceed.
            Be thorough but decisive - either request specific improvements OR give clear approval."""
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
            max_round=50,  # Allow enough rounds for iterations
            max_context_messages=self.max_context_messages,
            logger=self.logger
        )

        # GroupChat manager with custom speaker selection logic
        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=llm_config,
            system_message="""You are managing a test generation workflow with the following agents:
            1. TestGenerationAgent: Generates Python test code (ONLY agent that writes code)
            2. TestCodeReviewAgent: Reviews generated code and provides feedback (NEVER generates code)
            3. TestExecutionAgent: Executes test code only after approval
            4. TestCoordinator: Coordinates the workflow

            CRITICAL WORKFLOW RULES:
            1. TestGenerationAgent ALWAYS speaks first to generate initial code
            2. TestCodeReviewAgent ALWAYS reviews any code from TestGenerationAgent
            3. TestCodeReviewAgent MUST NEVER generate or write code - only provide reviews and feedback
            4. If TestCodeReviewAgent finds issues/improvements needed:
               - TestCodeReviewAgent provides specific feedback requesting changes
               - TestGenerationAgent must regenerate code addressing the feedback
               - TestCodeReviewAgent must re-review the updated code from TestGenerationAgent
               - Continue this cycle until TestCodeReviewAgent EXPLICITLY APPROVES
            5. TestExecutionAgent ONLY executes code AFTER TestCodeReviewAgent gives explicit approval
            6. If TestExecutionAgent fails, provide feedback to TestGenerationAgent to fix issues

            ROLE ENFORCEMENT:
            - TestGenerationAgent: The ONLY agent allowed to generate, write, or modify code
            - TestCodeReviewAgent: ONLY provides reviews, feedback, and approval/rejection decisions

            APPROVAL KEYWORDS: TestCodeReviewAgent must use phrases like:
            - "APPROVED FOR EXECUTION"
            - "CODE IS READY FOR EXECUTION"
            - "APPROVE THIS CODE FOR TESTING"

            Only allow TestExecutionAgent to speak after seeing these approval keywords.
            Ensure proper iterative feedback loops between generation and review.

            NOTE: Context is automatically managed to prevent token overflow.
            """
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

            Please work together to:
            1. Generate Python test code that implements these test cases
            2. Review the generated code for quality and correctness
            3. Execute the tests autonomously with full automation (handle dependencies, environment setup, etc.)
            4. If there are issues, iterate and improve until successful

            Let's start with code generation.
            """

        try:
            # Start the group chat
            self.logger.log("Orchestrator", f"Starting GroupChat for {impl_file}")

            # Initialize a fresh group chat for this file
            self.group_chat.messages = []  # Clear previous messages

            # Get context from knowledge base
            context = self.codegen_agent.kb.query("all reduce PyTorch Collective API test cases")
            print(f"[Info]: knowledge context retrieved ({len(context)} chars)")

            full_prompt = f"Based on the following code context:\n\n{context}\n\n {initial_message}"
            # Start the conversation
            self.coordinator.initiate_chat(
                self.manager,
                message=initial_message,
                max_turns=20  # Limit turns to prevent infinite loops
            )

            # Manage context after conversation to keep it within limits
            self._manage_context_length()

            # Check if we have successful test execution
            success = self._extract_success_from_chat()

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
        """Extract success status from the group chat messages"""
        # Look through the chat messages for execution success indicators
        for message in self.group_chat.messages:
            content = str(message.get('content', ''))
            if 'SUCCESS: Test execution completed successfully!' in content:
                return True
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
