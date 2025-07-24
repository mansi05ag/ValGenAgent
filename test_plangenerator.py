import os
import json
import argparse
import re
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Load environment variables
load_dotenv()

def generate_test_plan(api_key: Optional[str] = None, feature_info: Optional[Dict] = None, verbose: bool = False) -> tuple[Dict[str, Any], str]:
    """
    Generate a test plan for a specified feature in JSON format.

    Args:
        feature_name (str): The name of the feature to create a test plan for
        api_key (str): Not used for Gemini
        feature_info (dict): Optional feature information to enhance the prompt
        verbose (bool): Whether to print verbose output

    Returns:
        dict: Generated test plan in JSON format
        str: Raw response text for debugging
    """
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OpenAI API key not found. Provide it as an argument or set the OPENAI_API_KEY environment variable.")

    # Use Intel's internal API if needed, otherwise use standard OpenAI endpoint
    base_url = "https://apis-internal.intel.com/generativeaiinference/v4"
    client = OpenAI(api_key=openai.api_key, base_url=base_url)

    # Start with base prompt
    base_prompt = f"""Generate a detailed test plan for validating the {feature_info['name']} feature in PyTorch.
    The test plan should include:
    1. Test category name
    2. Test cases with the following information:
        - Test case ID : to link test plan and actual test,
        - Description : Information about what the test is doing. Any corener cases, that should be taken care during implementation. The expected result. Implementation details. Parameterize the test on data types, tensor sizes and world size. How to check the Performance of the test.
    Note:
    1. Ensure that the test plan does not have any duplicate test cases. The test cases should be unique and not repeated across categories.
    2. Do not create separate tests for each datatypes, or tensor sizes. Instead, parameterize the test cases to cover all relevant data types and tensor sizes in a single test case.
    3. Each test case should have a unique ID that can be linked to the actual test implementation.


    The test plan should be structured as a JSON object with the following format:
    {{
        "test_plan" : "{feature_info['name']}"
        tests:
        {{
            "test_category": "API name for which the below test cases are written",
            "implementation_file": "name/of/implementation/file.py",
            "test_cases": {
                {
                    "test_id": "to link test plan and actual test",
                    "description": "test_case_description",
                    "test_id": "to link test plan and actual test",
                    "description": "test_case_description",
                    "test_id": "to link test plan and actual test",
                    "description": "test_case_description"
                }
            },
            "test_category": "API name for which the below test cases are written",
            "implementation_file": "name/of/implementation/file.py",
            "test_cases": {
                {
                    "test_id": "to link test plan and actual test",
                    "description": "test_case_description",
                    "test_id": "to link test plan and actual test",
                    "description": "test_case_description",
                    "test_id": "to link test plan and actual test"
                }
            },
        }}
    }}
    """

    # Add feature info to prompt if available
    if feature_info:
        feature_info_str = json.dumps(feature_info, indent=2)
        base_prompt += f"\n\nConsider the feature information while generating the test plan:\n {feature_info_str}"

    print(f"Generating test plan for feature: ")
    if feature_info:
        print(f"Using feature info: {feature_info}")

    try:
        print(f"Base prompt for test plan generation:\n{base_prompt}\n")
        # Generate test plan using Gemini
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a testing expert that creates reliable JSON-formatted test plans. \n Do not stop output until the entire JSON object is fully completed. \n Ensure every opening { or [ has a matching closing } or ]. \n Do not output explanatory text or partial data. \n If the JSON is large, always generate the full object in one response. \n If truncation is likely, indicate it clearly with a comment at the end like // TRUNCATED. \n All tests for a feature or collective must go into one file."},
                {"role": "user", "content": base_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,        # Lower temperature for more consistent formatting
            max_completion_tokens=5000
        )

        # Get the raw text response
        response_text = response.choices[0].message.content

        # Extract and parse JSON from response
        # response_text = response.text
        # Find JSON content between triple backticks if present
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)

        try:
            test_plan = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to fix common JSON formatting issues
            fixed_text = repair_json(response_text)
            test_plan = json.loads(fixed_text)

        if verbose:
            print("Successfully generated test plan")

        return test_plan, response_text

    except Exception as e:
        print(f"Error generating test plan: {str(e)}")
        raise

def repair_json(text: str) -> str:
    """Repair common JSON formatting issues."""
    # Fix unterminated strings
    lines = text.split('\n')
    in_quote = False
    fixed_lines = []

    for line in lines:
        for i, char in enumerate(line):
            if char == '"' and (i == 0 or line[i-1] != '\\'):
                in_quote = not in_quote
        if in_quote:
            line = line + '"'
            in_quote = False
        fixed_lines.append(line)

    text = '\n'.join(fixed_lines)

    # Common JSON fixes
    text = re.sub(r"(?<![\\])\'", '"', text)  # Replace single quotes with double quotes
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # Remove trailing commas
    text = re.sub(r":\s*,", ": null,", text)  # Replace empty values with null
    text = re.sub(r":\s*}", ": null}", text)  # Replace empty values with null

    return text

def create_test_plan_document(test_plan: Dict[str, Any], output_file: str, feature_info: str) -> None:
    """Create a Word document from the test plan."""
    doc = Document()

    # Add title - handle both feature_info as dict or string
    if isinstance(feature_info, dict) and 'name' in feature_info:
        title_text = f'Test Plan: {feature_info["name"]}'
    elif isinstance(feature_info, str):
        title_text = f'Test Plan: {feature_info}'
    elif 'test_category' in test_plan:
        title_text = f'Test Plan: {test_plan["test_category"]}'
    else:
        title_text = 'Test Plan'

    title = doc.add_heading(title_text, 0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    # Handle different JSON structure
    if 'test_category' in test_plan and 'test_cases' in test_plan:
        # New simplified structure with single category
        # Add category heading
        cat_heading = doc.add_heading(f'Category: {test_plan["test_category"]}', level=1)

        # Add test cases
        for test_case in test_plan['test_cases']:
            # Test case title using test_id
            tc_heading = doc.add_heading(f'Test Case {test_case["test_id"]}', level=2)

            # Description (which contains all the test details)
            doc.add_paragraph(f'Description: {test_case["description"]}')

            # Add spacing between test cases
            doc.add_paragraph()

    # Save the document
    doc.save(output_file)

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate test plan for PyTorch feature')
    parser.add_argument('--output', required=True, help='Output file path for test plan document')
    parser.add_argument('--json', required=True, help='Output file path for test plan JSON')
    parser.add_argument('--feature-info-file', help='Path to feature info JSON file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Load feature info if provided
    feature_info = None
    if args.feature_info_file and os.path.exists(args.feature_info_file):
        with open(args.feature_info_file, 'r') as f:
            feature_info = json.load(f)

    # Generate test plan
    test_plan, raw_response = generate_test_plan(
        feature_info=feature_info,
        verbose=args.verbose
    )

    # Save test plan as JSON
    with open(args.json, 'w') as f:
        json.dump(test_plan, f, indent=2)

    # import pdb; pdb.set_trace()  # Debugging breakpoint
    # Create Word document
    create_test_plan_document(test_plan, args.output, feature_info)

    if args.verbose:
        print(f"Test plan saved to: {args.output}")
        print(f"JSON test plan saved to: {args.json}")

if __name__ == '__main__':
    main()