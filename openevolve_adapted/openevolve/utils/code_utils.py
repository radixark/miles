"""
Utilities for code parsing, diffing, and manipulation
"""

import re
import hashlib
from typing import Dict, List, Optional, Tuple, Union
from openevolve.utils.strip import strip_text


def apply_diff(original_code: str, diff_text: str) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)


def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    print(f"DEBUG: parse_full_rewrite called with language={language}")
    
    # Check if llm_response is None
    if llm_response is None:
        print("DEBUG: llm_response is None")
        return None
        
    print(f"DEBUG: Response length: {len(llm_response)}")
    print(f"DEBUG: First 200 chars: {repr(llm_response[:200])}")
    
    # Try language-specific code block first with more flexible pattern
    # Handle potential spaces and different newline styles
    code_block_pattern = r"```\s*" + re.escape(language) + r"\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL | re.IGNORECASE)
    print(f"DEBUG: Language-specific pattern matches: {len(matches)}")

    if matches:
        print(f"DEBUG: Found language-specific code block, length: {len(matches[0])}")
        return matches[0].strip()

    # Try generic code block with language specification (more flexible)
    code_block_pattern = r"```\s*(?:go|python|java|cpp|javascript|rust|sql)\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL | re.IGNORECASE)
    print(f"DEBUG: Generic language pattern matches: {len(matches)}")

    if matches:
        # Filter matches for the correct language
        for match in matches:
            code = match.strip()
            if language == "go" and (code.startswith("package ") or "func " in code[:100]):
                print(f"DEBUG: Found Go code block via generic pattern, length: {len(code)}")
                return code
            elif language == "python" and ("import " in code[:200] or "def " in code[:200] or code.startswith("from ")):
                print(f"DEBUG: Found Python code block via generic pattern, length: {len(code)}")
                return code
        
        # If no specific language match, return the first one
        if matches:
            print(f"DEBUG: Found generic language code block, length: {len(matches[0])}")
            return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```[^`]*?\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)
    print(f"DEBUG: Any code block pattern matches: {len(matches)}")

    if matches:
        # Filter out obvious non-code blocks
        for match in matches:
            code = match.strip()
            # Skip if it looks like natural language or markdown
            if any(word in code.lower() for word in ['here is', 'here\'s', 'this code', 'the following', 'i\'ll', 'i will']):
                continue
            # For Go, ensure it starts with valid Go syntax
            if language == "go":
                if code.startswith("package ") or "func " in code[:100]:
                    print(f"DEBUG: Found Go code via fallback pattern, length: {len(code)}")
                    return code
            elif language == "python":
                if ("import " in code[:200] or "def " in code[:200] or 
                    code.startswith("from ") or code.startswith("class ")):
                    print(f"DEBUG: Found Python code via fallback pattern, length: {len(code)}")
                    return code
            else:
                print(f"DEBUG: Found code via fallback pattern, length: {len(code)}")
                return code

    # NEW: Try to extract code without markdown blocks if response looks like it contains code
    if language == "go" and "package " in llm_response and "func " in llm_response:
        print("DEBUG: Attempting to extract Go code without markdown blocks")
        # Find the start of the package declaration
        package_match = re.search(r'package\s+\w+', llm_response)
        if package_match:
            start_pos = package_match.start()
            # Try to find the end by looking for the last } or end of string
            remaining_text = llm_response[start_pos:]
            # Remove any trailing markdown or explanation text
            lines = remaining_text.split('\n')
            code_lines = []
            for line in lines:
                # Stop if we hit markdown closing or explanatory text
                if line.strip().startswith('```') or line.strip().startswith('This code'):
                    break
                code_lines.append(line)
            
            extracted_code = '\n'.join(code_lines).strip()
            if extracted_code and extracted_code.startswith('package '):
                print(f"DEBUG: Extracted Go code without markdown, length: {len(extracted_code)}")
                return extracted_code

    # Don't return the entire response as fallback - this causes compilation errors
    print(f"DEBUG: No valid code blocks found, returning None instead of entire response")
    return None


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Create a short summary
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = (
                f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            )
            replace_summary = (
                f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            )
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    if code1 == code2:
        return 0

    # Simple implementation of Levenshtein distance
    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    print(f"DEBUG: extract_code_language called with code length: {len(code)}")
    print(f"DEBUG: First 500 chars: {repr(code[:500])}")
    
    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        print("DEBUG: Detected Python")
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        print("DEBUG: Detected Java")
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        print("DEBUG: Detected C++")
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        print("DEBUG: Detected JavaScript")
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        print("DEBUG: Detected Rust")
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        print("DEBUG: Detected SQL")
        return "sql"
    elif re.search(r"^(package|import go|package main)", code, re.MULTILINE):
        print("DEBUG: Detected Go")
        return "go"

    print("DEBUG: No language detected, returning 'unknown'")
    return "unknown"


def remove_comments_and_normalize(code: str, language: str = "python") -> str:
    """
    Remove comments and normalize code format

    Args:
        code: Original code
        language: Programming language (currently supports "python")

    Returns:
        Normalized code string
    """
    if language.lower() == "python":
        try:
            return strip_text(code)
        except Exception:
            raise RuntimeError(f"Failed to normalize Python code: {e}")
    else:
        raise ValueError(f"Unsupported language for normalization: {language}")


def calculate_normalized_hash(code: str, language: str = "python") -> str:
    """
    Calculate SHA256 hash of code with comments removed

    Args:
        code: Original code
        language: Programming language

    Returns:
        SHA256 hash of the code
    """
    normalized_code = remove_comments_and_normalize(code, language)
    return hashlib.sha256(normalized_code.encode('utf-8')).hexdigest()


def check_code_identical(code1: str, code2: str, language: str = "python") -> bool:
    """
    Check if two code snippets are functionally identical

    Args:
        code1: First code snippet
        code2: Second code snippet
        language: Programming language

    Returns:
        True if codes are functionally identical, False otherwise
    """
    # 1. Direct string comparison (fastest check)
    if code1 == code2:
        return True

    # 2. Normalized comparison (handles whitespace, comments differences)
    try:
        normalized_code1 = remove_comments_and_normalize(code1, language)
        normalized_code2 = remove_comments_and_normalize(code2, language)
        return normalized_code1 == normalized_code2
    except Exception:
        # If normalization fails, fallback to direct comparison
        return False
