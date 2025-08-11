#!/usr/bin/env python3
"""
Enhanced Test Workflow Runner

This script coordinates the entire test workflow:
1. Accept feature input file (JSON) containing name and description fields
2. Load additional documentation from static input_dirs directory (docs/ and public_urls_testplan.txt)
3. Generate test plan document using combined feature info and additional documentation
4. Run test_automation_agent.py to create test code using code/ directory for reference
5. Execute the generated tests
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
import shutil

# Import the new OpenAI API key utility
from utils.openai_api_key_utils import get_openai_api_key

# Import the new document processor
from utils.document_processor import DocumentProcessor

# Import the modules directly
from test_plangenerator import generate_test_plan_files
from agents.codegen_agent import run_test_automation

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
                 feature_input_file: Optional[str] = None, api_key = None,
                 generate_plan: bool = True, run_automation: bool = True):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.test_plan_file = test_plan_file
        self.api_key = api_key or get_openai_api_key()
        self.feature_input_file = feature_input_file
        self.generate_plan = generate_plan
        self.run_automation = run_automation

        # Static input directory for documents and URLs
        self.input_dirs_path = Path("input_dirs")

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
        """Load feature information from feature input file."""
        if not self.feature_input_file or not os.path.exists(self.feature_input_file):
            print("Warning: No feature input file provided or file does not exist")
            return {}

        try:
            print(f"Loading feature information from: {self.feature_input_file}")

            with open(self.feature_input_file, 'r', encoding='utf-8') as f:
                feature_info = json.load(f)

            # Validate required fields
            if 'name' not in feature_info or 'description' not in feature_info:
                print("Error: Feature input file must contain 'name' and 'description' fields")
                return {}

            print(f"Loaded feature: {feature_info['name']}")
            return feature_info

        except Exception as e:
            print(f"Error loading feature info from file: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return {}

    def load_additional_docs_content(self) -> str:
        """Load additional documentation content from input_dirs directory."""
        if not self.input_dirs_path.exists():
            print(f"Warning: Static input directory {self.input_dirs_path} does not exist")
            return ""

        try:
            print(f"Loading additional documentation from: {self.input_dirs_path}")
            stage_start = time.time()

            # Load documents from docs directory and URLs
            doc_infos = self.doc_processor.load_documents_from_directory(self.input_dirs_path / "docs")

            if not doc_infos:
                print("No additional documentation found")
                return ""

            # Prepare content for inclusion in test plan generation
            prepared_content = self.doc_processor.prepare_content(doc_infos)

            stage_time = time.time() - stage_start
            print(f"Additional documentation loading completed in {stage_time:.2f} seconds")
            return prepared_content

        except Exception as e:
            print(f"Error loading additional documentation: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return ""

    def generate_test_plan(self) -> Tuple[bool, str]:
        """Generate test plan document."""
        if self.test_plan_file and os.path.exists(self.test_plan_file):
            print(f"Using existing test plan: {self.test_plan_file}")
            return True, self.test_plan_file

        try:
            print("Loading feature information...")
            feature_info = self.load_feature_info()

            if not feature_info:
                print("Error: No valid feature information found")
                return False, ""

            # Extract name from feature_info and create filename
            feature_name = feature_info['name']
            # Clean the name for use as filename (remove spaces, special chars)
            feature_name = "".join(c for c in feature_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            feature_name = feature_name.replace(' ', '_').lower()
            self.feature_name = feature_name  # Store for use in other methods

            test_plan_file = self.output_dir / f"{self.feature_name}_test_plan.docx"
            test_plan_json = self.output_dir / f"{self.feature_name}_test_plan.json"

            # Load additional documentation content
            additional_docs_content = self.load_additional_docs_content()

            # Combine feature info with additional documentation
            enhanced_feature_info = feature_info.copy()
            if additional_docs_content:
                enhanced_feature_info['additional_documentation'] = additional_docs_content
                print("Added additional documentation content to feature info")

            # Save enhanced feature info to temporary file
            feature_info_path = self.output_dir / "temp_feature_info.json"
            with open(feature_info_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_feature_info, f, indent=2, ensure_ascii=False)

            print(f"Temporary feature info saved to: {feature_info_path}")

            print("Generating test plan...")
            plan_start = time.time()

            success = generate_test_plan_files(
                output_file=str(test_plan_file),
                json_file=str(test_plan_json),
                feature_info_file=str(feature_info_path),
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
        if self.feature_input_file:
            print(f"Feature Input File: {self.feature_input_file}")
        print(f"Static Input Directory: {self.input_dirs_path}")
        print(f"  - Docs Directory: {self.input_dirs_path / 'docs'}")
        print(f"  - URLs File: {self.input_dirs_path / 'public_urls_testplan.txt'}")
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
        description='Run the complete test workflow using a feature input file and static input_dirs directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run complete workflow (generate plan + run automation) with feature input file
            python test_runner.py --feature-input-file path/to/feature.json --output-dir path/to/output_dir

            # Only generate test plan from feature input file
            python test_runner.py --generate-plan-only --feature-input-file path/to/feature.json --output-dir test_results

            # Only run test automation (requires existing test plan)
            python test_runner.py --test-automation-only --test-plan path/to/plan.json --output-dir test_results

            # Generate tests from feature input file without executing them
            python test_runner.py --feature-input-file path/to/feature.json --output-dir path/to/output_dir --execute-tests=false

            # Generate tests from existing test plan without executing them
            python test_runner.py --test-plan path/to/plan.json --output-dir path/to/output_dir --execute-tests=false
        """
    )
    parser.add_argument('--output-dir', default='test_results', help='Output directory for all artifacts')
    parser.add_argument('--test-plan', help='Path to existing test plan (optional)')
    parser.add_argument('--feature-input-file', help='Path to feature input JSON file containing name and description fields')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--code-dir', default='./code', help='Path to the code directory for RAG.')
    parser.add_argument('--remove_index_db', action='store_true', help='deletes the already created index db for RAG')
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
    
    index_db_dir='index_db/'
    if args.remove_index_db:
        if os.path.exists(index_db_dir):
            shutil.rmtree(index_db_dir)
            print(f"Deleted the directory: {index_db_dir}")
        else:
            print(f"The directory {index_db_dir} does not exist.")

    runner = TestWorkflowRunner(
        output_dir=args.output_dir,
        verbose=args.verbose,
        test_plan_file=args.test_plan,
        generate_plan=generate_plan,
        run_automation=run_automation,
        feature_input_file=args.feature_input_file
    )

    success = runner.run(execute_tests=args.execute_tests)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()