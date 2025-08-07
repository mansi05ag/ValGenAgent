#!/usr/bin/env python3
"""
Enhanced Test Workflow Runner

This script coordinates the entire test workflow:
1. Accept input directory containing docs/ and code/ subdirectories
2. Generate test plan document using documentation from docs/ directory
3. Run test_automation_agent.py to create test code using code/ directory for reference
4. Execute the generated tests
5. Record results in an Excel file

Input Directory Structure:
input_directory/
├── docs/     (documentation files: .docx, .pptx, .xlsx, .pdf, .txt, .md, .html, etc.)
└── code/     (source code files for reference during test generation)
"""

import os
import sys
import time
import json
import argparse
import subprocess
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Import the new OpenAI API key utility
from utils.openai_api_key_utils import get_openai_api_key

# Import the new document processor
from utils.document_processor import DocumentProcessor

# Import the modules directly
from test_plangenerator import generate_test_plan_files
from agents.test_codegen_agent import run_test_automation

# Load environment variables
load_dotenv()

@dataclass
class TestResult:
    """Data class for test execution results"""
    test_name: str
    status: str
    execution_time: float
    error_message: Optional[str] = None

class TestWorkflowRunner:
    """Manages the end-to-end test workflow"""

    def __init__(self, output_dir: str, verbose: bool = False,
                 test_plan_file: Optional[str] = None,
                 input_dir: Optional[str] = None, api_key = None,
                 generate_plan: bool = True, run_automation: bool = True):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.test_plan_file = test_plan_file
        self.api_key = api_key or get_openai_api_key()
        self.input_dir = input_dir
        self.generate_plan = generate_plan
        self.run_automation = run_automation

        # Initialize feature_name (will be set when loading feature info)
        self.feature_name = "sample"

        # Initialize document processor
        self.doc_processor = DocumentProcessor(
            api_key=self.api_key,
            verbose=self.verbose
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_feature_info(self) -> Dict:
        """Load feature information from input directory, creating a structured feature input JSON."""
        if not self.input_dir or not os.path.exists(self.input_dir):
            return {}

        try:
            print(f"Processing input directory: {self.input_dir}")
            stage_start = time.time()

            # Use document processor to create feature info
            success, json_file_path, feature_info = self.doc_processor.process_input_directory(
                input_dir=self.input_dir,
                output_dir=str(self.output_dir)
            )

            stage_time = time.time() - stage_start

            if not success:
                print("Failed to process input directory")
                return {}

            print(f"Feature input JSON saved to: {json_file_path}")
            print(f"Document processing completed in {stage_time:.2f} seconds")
            return feature_info

        except Exception as e:
            print(f"Error loading feature info from input directory: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return {}

    def generate_test_plan(self) -> Tuple[bool, str]:
        """Generate test plan document."""
        if self.test_plan_file and os.path.exists(self.test_plan_file):
            print(f"Using existing test plan: {self.test_plan_file}")
            return True, self.test_plan_file

        try:
            print("Loading feature information...")
            feature_info = self.load_feature_info()

            # Extract name from feature_info and create filename
            if feature_info and 'name' in feature_info:
                feature_name = feature_info['name']
                # Clean the name for use as filename (remove spaces, special chars)
                feature_name = "".join(c for c in feature_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                feature_name = feature_name.replace(' ', '_').lower()
                self.feature_name = feature_name  # Store for use in other methods

            test_plan_file = self.output_dir / f"{self.feature_name}_test_plan.docx"
            test_plan_json = self.output_dir / f"{self.feature_name}_test_plan.json"

            feature_info_path = None
            if feature_info:
                feature_info_path = self.output_dir / "temp_feature_info.json"
                with open(feature_info_path, 'w') as f:
                    json.dump(feature_info, f)

            print("Generating test plan with AI...")
            plan_start = time.time()

            success = generate_test_plan_files(
                output_file=str(test_plan_file),
                json_file=str(test_plan_json),
                feature_info_file=str(feature_info_path) if feature_info_path else None,
                verbose=self.verbose
            )

            plan_time = time.time() - plan_start

            if not success:
                print("Error generating test plan")
                return False, ""

            print(f"Test plan generation completed in {plan_time:.2f} seconds")

            # Clean up temporary file
            if feature_info_path and feature_info_path.exists():
                feature_info_path.unlink()

            # Return the JSON file path for the test automation agent to use
            return True, str(test_plan_json)

        except Exception as e:
            print(f"Error in test plan generation: {e}")
            import traceback
            traceback.print_exc()
            return False, ""

    def run_test_automation(self, test_plan_file: str, execute_tests: bool = True) -> bool:
        """Run test automation agent to generate and execute tests."""
        print("Initializing test automation...")
        try:
            output_dir = str(self.output_dir / "generated_tests")

            print("Generating test code...")

            # Call the function directly instead of subprocess
            success = run_test_automation(
                test_plan_path=test_plan_file,
                output_dir=output_dir,
                max_retries=2,  # default value
                max_context=25,  # default value
                verbose=self.verbose,
                execute_tests=execute_tests
            )

            if not success:
                print("Error in test automation: Test generation failed")
                return False

            if execute_tests:
                print("Test generation and execution completed")
            else:
                print("Test code generation completed")

            return True

        except Exception as e:
            print(f"Error running test automation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def collect_test_results(self) -> List[TestResult]:
        """Collect results from test execution."""
        print("Parsing test results...")
        results_start = time.time()

        results = []
        xml_files = glob.glob(str(self.output_dir / "generated_tests" / "*.xml"))

        for xml_file in xml_files:
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for testcase in root.findall(".//testcase"):
                    result = TestResult(
                        test_name=testcase.get('name', 'Unknown'),
                        status='Failed' if testcase.findall('./failure') else 'Passed',
                        execution_time=float(testcase.get('time', 0)),
                        error_message=testcase.findall('./failure')[0].text if testcase.findall('./failure') else None
                    )
                    results.append(result)

            except Exception as e:
                print(f"Error parsing test results from {xml_file}: {e}")

        results_time = time.time() - results_start
        print(f"Results collection completed in {results_time:.2f} seconds")

        return results

    def save_results_to_excel(self, results: List[TestResult]) -> None:
        """Save test results to Excel file."""
        if not results:
            print("No test results to save")
            return

        try:
            print("Creating Excel report...")
            excel_start = time.time()

            df = pd.DataFrame([
                {
                    'Test Name': r.test_name,
                    'Status': r.status,
                    'Execution Time (s)': r.execution_time,
                    'Error Message': r.error_message or ''
                }
                for r in results
            ])

            excel_file = self.output_dir / f"{self.feature_name}_test_results.xlsx"
            df.to_excel(excel_file, index=False)

            excel_time = time.time() - excel_start
            print(f"Excel report saved to: {excel_file}")
            print(f"Excel generation completed in {excel_time:.2f} seconds")

        except Exception as e:
            print(f"Error saving results to Excel: {e}")

    def print_workflow_summary(self, execute_tests: bool = True):
        """Print a summary of the workflow configuration."""
        print("=" * 60)
        print(f"Test Workflow Configuration")
        print("=" * 60)
        print(f"Output Directory: {self.output_dir}")
        print(f"Generate Test Plan: {'Yes' if self.generate_plan else 'No'}")
        print(f"Run Test Automation: {'Yes' if self.run_automation else 'No'}")
        print(f"Execute Tests: {'Yes' if execute_tests else 'No'}")
        if self.test_plan_file:
            print(f"Test Plan File: {self.test_plan_file}")
        if self.input_dir:
            print(f"Input Directory: {self.input_dir}")
            print(f"  - Docs Directory: {Path(self.input_dir) / 'docs'}")
            print(f"  - Code Directory: {Path(self.input_dir) / 'code'}")
        print("=" * 60)

    def run(self, execute_tests: bool = True) -> bool:
        """Run the complete test workflow."""
        # Print workflow configuration
        self.print_workflow_summary(execute_tests)

        workflow_start = time.time()
        test_plan_file = None

        # Step 1: Generate test plan (if enabled)
        if self.generate_plan:
            print("\n" + "="*60)
            print("STAGE 1: TEST PLAN GENERATION")
            print("="*60)
            stage1_start = time.time()

            success, test_plan_file = self.generate_test_plan()

            stage1_time = time.time() - stage1_start

            if not success:
                print("FAILED: Failed to generate test plan.")
                return False

            print(f"SUCCESS: Stage 1 completed successfully in {stage1_time:.2f} seconds")
            print(f"   Test plan saved: {test_plan_file}")
        else:
            # Use provided test plan file or look for existing one
            if self.test_plan_file and os.path.exists(self.test_plan_file):
                test_plan_file = self.test_plan_file
                print(f"Using existing test plan: {test_plan_file}")
            else:
                # Look for existing test plan in output directory (prefer JSON for automation)
                possible_plans = [
                    self.output_dir / f"{self.feature_name}_test_plan.json",
                    self.output_dir / f"{self.feature_name}_test_plan.docx"
                ]
                for plan_file in possible_plans:
                    if plan_file.exists():
                        test_plan_file = str(plan_file)
                        print(f"Found existing test plan: {test_plan_file}")
                        break

                if not test_plan_file:
                    print("Error: No test plan file found and test plan generation is disabled.")
                    print("Either provide --test-plan or enable --generate-plan")
                    return False

        # Step 2: Run test automation (if enabled)
        if self.run_automation:
            print("\n" + "="*60)
            if execute_tests:
                print("STAGE 2: TEST CODE GENERATION & EXECUTION")
            else:
                print("STAGE 2: TEST CODE GENERATION")
            print("="*60)
            stage2_start = time.time()

            if not self.run_test_automation(test_plan_file, execute_tests):
                print("FAILED: Failed to run test automation.")
                return False

            stage2_time = time.time() - stage2_start
            print(f"SUCCESS: Stage 2 completed successfully in {stage2_time:.2f} seconds")

            # Step 3: Collect and save results (only if automation ran and tests are executed)
            if execute_tests:
                print("\n" + "="*60)
                print("STAGE 3: RESULTS COLLECTION & REPORTING")
                print("="*60)
                stage3_start = time.time()

                results = self.collect_test_results()
                self.save_results_to_excel(results)

                stage3_time = time.time() - stage3_start
                print(f"SUCCESS: Stage 3 completed successfully in {stage3_time:.2f} seconds")

                # Print test execution summary
                passed = sum(1 for r in results if r.status == 'Passed')
                failed = sum(1 for r in results if r.status == 'Failed')

                print("\n" + "="*60)
                print("TEST EXECUTION SUMMARY")
                print("="*60)
                print(f"Total Tests: {len(results)}")
                print(f"Passed: {passed}")
                print(f"Failed: {failed}")
                print(f"Success Rate: {(passed/len(results)*100):.1f}%" if results else "0%")

                workflow_time = time.time() - workflow_start
                print(f"\nTotal Workflow Time: {workflow_time:.2f} seconds")

                return failed == 0
            else:
                print("Test execution step skipped.")

        workflow_time = time.time() - workflow_start
        print(f"\nTotal Workflow Time: {workflow_time:.2f} seconds")
        return True

def main() -> None:

    parser = argparse.ArgumentParser(
        description='Run the complete test workflow using input directory with docs/ and code/ subdirectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run complete workflow (generate plan + run automation) with input directory
            python test_runner.py --input-dir path/to/input_directory --output-dir path/to/output_dir

            # Input directory should contain:
            #   input_directory/
            #   ├── docs/     (documentation files: .docx, .pptx, .xlsx, .pdf, .txt, .md, .html)
            #   └── code/     (source code files for reference)

            # Only generate test plan from input directory
            python test_runner.py --generate-plan-only --input-dir path/to/input_directory --output-dir test_results

            # Only run test automation (requires existing test plan)
            python test_runner.py --test-automation-only --test-plan path/to/plan.json --output-dir test_results

            # Generate tests from input directory without executing them
            python test_runner.py --input-dir path/to/input_directory --output-dir path/to/output_dir --execute-tests=false

            # Generate tests from existing test plan without executing them
            python test_runner.py --test-plan path/to/plan.json --output-dir path/to/output_dir --execute-tests=false
        """
    )
    parser.add_argument('--output-dir', default='test_results', help='Output directory for all artifacts')
    parser.add_argument('--test-plan', help='Path to existing test plan (optional)')
    parser.add_argument('--input-dir', help='Path to input directory containing docs/ and code/ subdirectories')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--code-dir', default='./code', help='Path to the code directory for RAG.')

    # Step control arguments
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument('--generate-plan-only', action='store_true',
                           help='Only generate test plan, skip test automation')
    step_group.add_argument('--test-automation-only', action='store_true',
                           help='Only run test automation, skip test plan generation')

    # Test execution control
    parser.add_argument('--execute-tests', type=lambda x: x.lower() in ('true', '1', 'yes'),
                       default=True, help='Execute generated tests (default: True). Set to False to only generate tests without execution.')

    args = parser.parse_args()

    # Determine which steps to run
    generate_plan = True
    run_automation = True

    if args.generate_plan_only:
        generate_plan = True
        run_automation = False
        print("Mode: Generate test plan only")
    elif args.test_automation_only:
        generate_plan = False
        run_automation = True
        print("Mode: Test automation only")
    else:
        # Default mode: generate plan (if needed) + test automation
        # The execute_tests flag will control whether tests are actually executed
        if args.execute_tests:
            print("Mode: Complete workflow (generate plan + test automation + execution)")
        else:
            print("Mode: Generate tests only (skip execution)")

    runner = TestWorkflowRunner(
        output_dir=args.output_dir,
        verbose=args.verbose,
        test_plan_file=args.test_plan,
        generate_plan=generate_plan,
        run_automation=run_automation,
        input_dir=args.input_dir
    )

    success = runner.run(execute_tests=args.execute_tests)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()