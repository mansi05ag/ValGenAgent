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
from google import genai

# Load environment variables
load_dotenv()

# Initialize Gemini client
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai_client = genai.Client(api_key=GEMINI_API_KEY)

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
                 feature_info_file: Optional[str] = None):
        self.feature_name = feature_name
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.test_plan_file = test_plan_file
        self.feature_info = feature_info
        self.feature_info_file = feature_info_file
        
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
            "test_automation_agent.py",
                "--test-plan", test_plan_file,
                "--output-dir", str(self.output_dir / "generated_tests"),
                "--timeout", "30"
            ]
            
            if self.verbose:
                cmd.append("--verbose")

            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error in test automation: {result.stderr}")
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

    def run(self) -> bool:
        """Run the complete test workflow."""
        start_time = time.time()
        
        # Step 1: Generate test plan
        success, test_plan_file = self.generate_test_plan()
        if not success:
            return False

        # Step 2: Run test automation
        if not self.run_test_automation(test_plan_file):
            return False

        # Step 3: Collect and save results
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

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run the complete test workflow')
    parser.add_argument('--feature', required=True, help='Name of the feature to test')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for all artifacts')
    parser.add_argument('--test-plan', help='Path to existing test plan (optional)')
    parser.add_argument('--feature-info', help='Path to feature info JSON file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    runner = TestWorkflowRunner(
        feature_name=args.feature,
        output_dir=args.output_dir,
        verbose=args.verbose,
        test_plan_file=args.test_plan,
        feature_info_file=args.feature_info
    )

    success = runner.run()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()