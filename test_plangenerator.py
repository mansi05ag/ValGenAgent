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
    Generate a test plan for a specified feature using Google's Gemini API in JSON format.

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
    base_prompt = f"""Generate a detailed test plan for validating the following feature in PyTorch.
    The test plan should include:
    1. Test categories
    2. Test cases with clear steps
    3. Expected results
    4. Implementation details
    5. Data types to test with
    6. Edge cases to consider
    7. Performance considerations

    Format the response as a JSON with the following structure:
    {{
        "feature_name": "{{feature_name}}",
        "test_categories": [
            {{
                "name": "category_name",
                "description": "category_description",
                "test_cases": [
                    {{
                        "id": "TC1",
                        "title": "test_case_title",
                        "description": "test_case_description",
                        "steps": ["step1", "step2", ...],
                        "expected_results": "expected_results",
                        "data_types": ["type1", "type2", ...],
                        "implementation_file": "test_filename.py"
                    }}
                ]
            }}
        ]
    }}
    """

    # Add feature info to prompt if available
    if feature_info:
        feature_info_str = json.dumps(feature_info, indent=2)
        base_prompt += f"\n\nConsider this additional feature information while generating the test plan:\n{feature_info_str}"

    if verbose:
        print(f"Generating test plan for feature: ")
        if feature_info:
            print(f"Using feature info: {feature_info}")

    try:
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

def create_test_plan_document(test_plan: Dict[str, Any], output_file: str) -> None:
    """Create a Word document from the test plan."""
    doc = Document()

    # Add title
    title = doc.add_heading(f'Test Plan: {test_plan["feature_name"]}', 0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    # Add categories and test cases
    for category in test_plan['test_categories']:
        # Add category heading
        cat_heading = doc.add_heading(f'Category: {category["name"]}', level=1)
        doc.add_paragraph(category['description'])

        # Add test cases
        for test_case in category['test_cases']:
            # Test case title
            tc_heading = doc.add_heading(f'Test Case {test_case["id"]}: {test_case["title"]}', level=2)

            # Description
            doc.add_paragraph(f'Description: {test_case["description"]}')

            # Implementation file
            if 'implementation_file' in test_case:
                doc.add_paragraph(f'Implementation file: {test_case["implementation_file"]}')

            # Steps
            steps_para = doc.add_paragraph('Steps:')
            for step in test_case['steps']:
                doc.add_paragraph(step, style='List Bullet')

            # Expected Results
            doc.add_paragraph(f'Expected Result: {test_case["expected_results"]}')

            # Data Types
            if 'data_types' in test_case and test_case['data_types']:
                doc.add_paragraph(f'Data Types: {", ".join(test_case["data_types"])}')

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

    # Create Word document
    create_test_plan_document(test_plan, args.output)

    if args.verbose:
        print(f"Test plan saved to: {args.output}")
        print(f"JSON test plan saved to: {args.json}")

if __name__ == '__main__':
    main()