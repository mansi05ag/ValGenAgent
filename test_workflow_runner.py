#!/usr/bin/env python3
"""
Enhanced Test Workflow Runner

This script coordinates the entire test workflow:
1. Collect feature information from command line or JSON file
2. Generate test plan document using test_plangenerator.py
3. Run test_automation_agent.py to create test code
4. Execute the generated tests
5. Record results in an Excel file
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

    def __init__(self, feature_name: str, output_dir: str, verbose: bool = False,
                 test_plan_file: Optional[str] = None, feature_info: Optional[Dict] = None,
                 feature_info_file: Optional[str] = None, api_key = None,
                 generate_plan: bool = True, run_automation: bool = True):
        self.feature_name = feature_name
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.test_plan_file = test_plan_file
        self.api_key = api_key or get_openai_api_key()
        self.feature_info = feature_info
        self.feature_info_file = feature_info_file
        self.generate_plan = generate_plan
        self.run_automation = run_automation

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_feature_info(self) -> Dict:
        """Load feature information from file if provided."""
        if self.feature_info:
            return self.feature_info

        if self.feature_info_file and os.path.exists(self.feature_info_file):
            try:
                with open(self.feature_info_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading feature info file: {e}")
                return {}
        return {}

    def generate_test_plan(self) -> Tuple[bool, str]:
        """Generate test plan document."""
        if self.test_plan_file and os.path.exists(self.test_plan_file):
            print(f"Using existing test plan: {self.test_plan_file}")
            return True, self.test_plan_file

        print("Generating test plan...")
        test_plan_file = self.output_dir / f"{self.feature_name}_test_plan.docx"
        test_plan_json = self.output_dir / f"{self.feature_name}_test_plan.json"

        try:
            feature_info = self.load_feature_info()
            cmd = [
            sys.executable,
            "test_plangenerator.py",
            "--feature", self.feature_name,
                "--output", str(test_plan_file),
                "--json", str(test_plan_json)
            ]

            if feature_info:
                feature_info_path = self.output_dir / "temp_feature_info.json"
                with open(feature_info_path, 'w') as f:
                    json.dump(feature_info, f)
                cmd.extend(["--feature-info-file", str(feature_info_path)])

            if self.verbose:
                cmd.append("--verbose")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error generating test plan: {result.stderr}")
                return False, ""

            return True, str(test_plan_file)

        except Exception as e:
            print(f"Error in test plan generation: {e}")
            return False, ""

    def run_test_automation(self, test_plan_file: str) -> bool:
        """Run test automation agent to generate and execute tests."""
        print("Running test automation...")
        try:
            cmd = [
            sys.executable,
            "agents/test_automation_agent_v3.py",
                "--test-plan", test_plan_file,
                "--output-dir", str(self.output_dir / "generated_tests")
            ]

            if self.verbose:
                cmd.append("--verbose")

            # Use Popen for real-time output streaming
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, universal_newlines=True) as process:

                # Stream output in real-time
                for line in process.stdout:
                    print(line.rstrip())

                # Wait for process to complete
                process.wait()

                if process.returncode != 0:
                    print(f"Error in test automation: Process exited with code {process.returncode}")
                    return False

            return True

        except Exception as e:
            print(f"Error running test automation: {e}")
            return False

    def collect_test_results(self) -> List[TestResult]:
        """Collect results from test execution."""
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

        return results

    def save_results_to_excel(self, results: List[TestResult]) -> None:
        """Save test results to Excel file."""
        if not results:
            print("No test results to save")
            return

        try:
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
            print(f"Test results saved to: {excel_file}")

        except Exception as e:
            print(f"Error saving results to Excel: {e}")

    def print_workflow_summary(self):
        """Print a summary of the workflow configuration."""
        print("=" * 60)
        print(f"Test Workflow Configuration")
        print("=" * 60)
        print(f"Feature: {self.feature_name}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Generate Test Plan: {'Yes' if self.generate_plan else 'No'}")
        print(f"Run Test Automation: {'Yes' if self.run_automation else 'No'}")
        if self.test_plan_file:
            print(f"Test Plan File: {self.test_plan_file}")
        if self.feature_info_file:
            print(f"Feature Info File: {self.feature_info_file}")
        print("=" * 60)

    def run(self) -> bool:
        """Run the complete test workflow."""
        # Print workflow configuration
        self.print_workflow_summary()

        start_time = time.time()
        test_plan_file = None

        # Step 1: Generate test plan (if enabled)
        if self.generate_plan:
            print("Step 1: Generating test plan...")
            success, test_plan_file = self.generate_test_plan()
            if not success:
                print("Failed to generate test plan.")
                return False
            print(f"Test plan generated successfully: {test_plan_file}")
        else:
            # Use provided test plan file or look for existing one
            if self.test_plan_file and os.path.exists(self.test_plan_file):
                test_plan_file = self.test_plan_file
                print(f"Using existing test plan: {test_plan_file}")
            else:
                # Look for existing test plan in output directory
                possible_plans = [
                    self.output_dir / f"{self.feature_name}_test_plan.docx",
                    self.output_dir / f"{self.feature_name}_test_plan.json"
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
            print("Step 2: Running test automation...")
            if not self.run_test_automation(test_plan_file):
                print("Failed to run test automation.")
                return False
            print("Test automation completed successfully.")

            # Step 3: Collect and save results (only if automation ran)
            print("Step 3: Collecting and saving test results...")
            results = self.collect_test_results()
            self.save_results_to_excel(results)

            # Print summary
            execution_time = time.time() - start_time
            passed = sum(1 for r in results if r.status == 'Passed')
            failed = sum(1 for r in results if r.status == 'Failed')

            print("\nTest Execution Summary:")
            print(f"Total Tests: {len(results)}")
            print(f"Passed: {passed}")
            print(f"Failed: {failed}")
            print(f"Total Execution Time: {execution_time:.2f} seconds")

            return failed == 0
        else:
            print("Test automation step skipped.")
            execution_time = time.time() - start_time
            print(f"Total Execution Time: {execution_time:.2f} seconds")
            return True

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run the complete test workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow (generate plan + run automation)
  python test_workflow_runner.py --feature collectives

  # Only generate test plan
  python test_workflow_runner.py --feature collectives --generate-plan-only

  # Only run test automation (requires existing test plan)
  python test_workflow_runner.py --feature collectives --test-automation-only --test-plan path/to/plan.docx

  # Run automation with auto-discovery of existing test plan
  python test_workflow_runner.py --feature collectives --test-automation-only
        """
    )
    parser.add_argument('--feature', required=True, help='Name of the feature to test')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for all artifacts')
    parser.add_argument('--test-plan', help='Path to existing test plan (optional)')
    parser.add_argument('--feature-info', help='Path to feature info JSON file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    # Step control arguments
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument('--generate-plan-only', action='store_true',
                           help='Only generate test plan, skip test automation')
    step_group.add_argument('--test-automation-only', action='store_true',
                           help='Only run test automation, skip test plan generation')

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
        print("Mode: Complete workflow (generate plan + test automation)")

    runner = TestWorkflowRunner(
        feature_name=args.feature,
        output_dir=args.output_dir,
        verbose=args.verbose,
        test_plan_file=args.test_plan,
        feature_info_file=args.feature_info,
        generate_plan=generate_plan,
        run_automation=run_automation
    )

    success = runner.run()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()