#!/usr/bin/env python3
"""
Script to count lines of code in Python files.
"""

import ast
import os
from typing import Set, Tuple


def get_excluded_lines(source_code: str) -> Set[int]:
    """Get line numbers for docstrings and multi-line strings."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return set()
    
    excluded_lines = set()
    
    def visit_node(node):
        # Check for docstrings
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.AsyncFunctionDef)):
            if (node.body and 
                isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                docstring_node = node.body[0].value
                start_line = docstring_node.lineno
                end_line = docstring_node.end_lineno if hasattr(docstring_node, 'end_lineno') else start_line
                excluded_lines.update(range(start_line, end_line + 1))
        
        # Check for multi-line strings
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if '\n' in node.value or (hasattr(node, 'end_lineno') and node.end_lineno > node.lineno):
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                excluded_lines.update(range(start_line, end_line + 1))
        
        for child in ast.iter_child_nodes(node):
            visit_node(child)
    
    visit_node(tree)
    return excluded_lines


def count_lines(file_path: str) -> Tuple[int, int]:
    """Count lines in a Python file. Returns: (total_lines, code_lines)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except (UnicodeDecodeError, IOError):
        return 0, 0
    
    source_code = ''.join(lines)
    excluded_lines = get_excluded_lines(source_code)
    
    total_lines = len(lines)
    code_lines = 0
    
    for i, line in enumerate(lines, 1):
        stripped_line = line.strip()
        
        # Skip empty lines, comments, and excluded lines
        if not stripped_line or stripped_line.startswith('#') or i in excluded_lines:
            continue
            
        code_lines += 1
    
    return total_lines, code_lines


def find_python_files(root_dir: str) -> list[str]:
    """Find all Python files, excluding common directories."""
    python_files = []
    exclude_dirs = {'__pycache__', '.git', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.ropeproject', 'examples', 'tests'}
    script_name = os.path.basename(__file__)
    
    for root, dirs, files in os.walk(root_dir):
        # Exclude dirs in the exclude set AND any dir starting with .venv
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.venv')]
        
        for file in files:
            if file.endswith('.py') and file != script_name:
                python_files.append(os.path.join(root, file))
    
    return sorted(python_files)


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    python_files = find_python_files(project_root)
    
    print("Ignoring directories: .venv*, __pycache__, .git, .pytest_cache, .mypy_cache, .ruff_cache, .ropeproject, examples, tests")
    print("Ignoring files: this script\n")
    
    total_files = 0
    total_lines = 0
    total_code = 0
    
    print(f"{'File':<60} {'Total':<8} {'Code':<8}")
    print("-" * 76)
    
    for file_path in python_files:
        if os.path.exists(file_path):
            file_total, file_code = count_lines(file_path)
            total_files += 1
            total_lines += file_total
            total_code += file_code
            
            rel_path = os.path.relpath(file_path, project_root)
            print(f"{rel_path:<60} {file_total:<8} {file_code:<8}")
    
    print("-" * 76)
    print(f"{'TOTAL':<60} {total_lines:<8} {total_code:<8}")
    print(f"\nSummary:")
    print(f"Total Python files: {total_files}")
    print(f"Total lines: {total_lines}")
    print(f"Empty/comment/multi-line string lines: {total_lines - total_code}")
    print(f"Core Lines of Code: {total_code}")


if __name__ == "__main__":
    main()
