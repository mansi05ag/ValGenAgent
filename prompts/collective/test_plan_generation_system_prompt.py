TEST_PLAN_SYSTEM_PROMPT=\
"""Generate a detailed test plan for validating the {feature_info['name']} feature in PyTorch.
    The test plan should include:
    1. Test category name
    2. Test cases with the following information:
        - Test case ID : to link test plan and actual test,
        - Description : test case description,
    Note:
    1. Ensure that the test plan does not have any duplicate test cases. The test cases should be unique and not repeated across categories.
    2. Do not create separate tests for each datatypes, or tensor sizes. Instead, parameterize the test cases to cover all relevant data types and tensor sizes in a single test case.
    3. Each test case should have a unique ID that can be linked to the actual test implementation.
    4. For now, create a test plan with only 1 test with 4 parameterization, as is mentioned in the input document.


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

