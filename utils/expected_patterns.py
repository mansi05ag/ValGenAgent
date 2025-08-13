LANGUAGE_PATTERNS = {
    'py': {
        'generation': [
            r'def test_.*\(',
            r'import pytest', 
            r'class Test.*:', 
            r'```python', 
            r'test.*\.py',
            r'LGTM', #Looks good to me (acronym)
            r'looks good', 
            r'approved', 
            r'filename.*\.py'
        ],
        'file_saving': [
            r'Code saved to.*\.py', 
            r'File saved.*\.py', 
            r'Saved.*\.py', 
            r'Successfully saved.*\.py',
            r'Written to.*\.py',
            r'Created.*\.py', 
            r'with open.*\.py.*w.*as f', 
            r'f\.write\(', 
            r'exitcode.*0'
        ],
        'execution': [
            r'SUCCESS:.*Test execution completed successfully', 
            r'Test execution completed successfully',
            r'\d+\s+passed\s+in\s+[\d.]+s', 
            r'=+\s*\d+\s+passed.*=+'
        ]
    },
    'cpp': {
        'generation': [
            r'#include',
            r'```cpp',
            r'LGTM', #Looks good to me (acronym)
            r'looks good',
            r'approved',
            r'filename.*\.(cpp|hpp|h)'
        ],
        'file_saving': [
            r'Code saved to.*\.(cpp|hpp|h)',
            r'File saved.*\.(cpp|hpp|h)',
            r'Saved.*\.(cpp|hpp|h)',
            r'Successfully saved.*\.(cpp|hpp|h)',
            r'Written to.*\.(cpp|hpp|h)',
            r'Created.*\.(cpp|hpp|h)',
            r'std::ofstream.*\.(cpp|hpp|h)',
            r'<<',
            r'exitcode.*0',
        ],
        'execution': [
            r'\[==========\] .*tests from .*test cases ran.',
            r'\[  PASSED  \] \d+ tests.',
            r'\[  FAILED  \] 0 tests.',
            r'\[  ALL TESTS PASSED  \]',
            r'\[==========\] .*tests from .*test cases ran in [\d.]+s',
        ]
    },
    'c': {
        'generation': [
            r'#include\s',
            r'#include\s',
            r'```c',
            r'LGTM', #Looks good to me (acronym)
            r'looks good',
            r'approved',
            r'filename.*\.(c|h)'
        ],
        'file_saving': [
            r'Code saved to.*\.(c|h)',
            r'File saved.*\.(c|h)',
            r'Saved.*\.(c|h)',
            r'Successfully saved.*\.(c|h)',
            r'Written to.*\.(c|h)',
            r'Created.*\.(c|h)',
            r'fopen\(.*\.(c|h)',
            r'fwrite\(',
            r'exitcode.*0',
        ],
        'execution': [
            r'\[==========\] .*tests from .*test cases ran.',  
            r'\[  PASSED  \] \d+ tests.',
            r'All tests passed',
            r'Test execution completed successfully',
            r'\d+\s+passed\s+in\s+[\d.]+s',
        ]
    },
    'asm': {
        'generation': [
            r';.*test',  
            r'global\s+_?main',
            r'section\s+\.text',
            r'mov\s+',
            r'```asm',
            r'test.*\.(asm|s)',
            r'LGTM', #Looks good to me (acronym)
            r'looks good',
            r'approved',
            r'filename.*\.(asm|s)'
        ],
        'file_saving': [
            r'Code saved to.*\.(asm|s)',
            r'File saved.*\.(asm|s)',
            r'Saved.*\.(asm|s)',
            r'Successfully saved.*\.(asm|s)',
            r'Written to.*\.(asm|s)',
            r'Created.*\.(asm|s)',
            r'fopen\(.*\.(asm|s)',
            r'fwrite\(',
            r'exitcode.*0',
        ],
        'execution': [
            r'Program exited with code 0',
            r'exitcode.*0',
            r'Successfully assembled',
            r'Successfully linked',
            r'Test execution completed successfully',
        ]
    }
}