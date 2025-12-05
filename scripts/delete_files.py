#!/usr/bin/env python3
"""
delete_files.py - Terminal utility for deleting multiple files safely

This script provides a safe and interactive way to delete multiple files
from the command line with various modes:
- Interactive mode: Select files to delete with confirmation
- Pattern mode: Delete files matching a pattern (e.g., *.log, *.tmp)
- List mode: Delete specific files from a provided list
- Dry-run mode: Preview what would be deleted without actually deleting

Safety features:
- Confirmation prompts before deletion
- Dry-run mode to preview changes
- Excludes critical files and directories
- Detailed logging of deletions

Usage Examples:
    # Interactive mode - select files to delete
    python scripts/delete_files.py --interactive
    
    # Delete specific files
    python scripts/delete_files.py --files file1.txt file2.log file3.md
    
    # Delete files matching a pattern
    python scripts/delete_files.py --pattern "*.log"
    
    # Delete with pattern, excluding certain files
    python scripts/delete_files.py --pattern "*.md" --exclude README.md
    
    # Dry run - see what would be deleted without deleting
    python scripts/delete_files.py --pattern "*.log" --dry-run
    
    # Delete from a specific directory
    python scripts/delete_files.py --pattern "*.tmp" --directory /path/to/dir
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Set
import fnmatch


# Critical files and directories that should never be deleted
PROTECTED_PATTERNS = {
    '.git',
    '.gitignore',
    'requirements.txt',
    'README.md',
    'LICENSE',
    '*.py',  # Protect all Python scripts by default
    '.venv*',
    'venv',
    'node_modules',
}

# Critical directories that should not be traversed
PROTECTED_DIRS = {
    '.git',
    '.venv',
    'venv',
    'node_modules',
    '__pycache__',
}


def is_protected(filepath: Path, custom_excludes: Set[str] = None) -> bool:
    """
    Check if a file is protected from deletion.
    
    Args:
        filepath: Path to check
        custom_excludes: Additional patterns to exclude
        
    Returns:
        True if the file should be protected, False otherwise
    """
    filename = filepath.name
    
    # Check against protected patterns
    for pattern in PROTECTED_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True
    
    # Check against custom excludes
    if custom_excludes:
        for pattern in custom_excludes:
            if fnmatch.fnmatch(filename, pattern):
                return True
            if fnmatch.fnmatch(str(filepath), pattern):
                return True
    
    # Check if in protected directory
    for part in filepath.parts:
        if part in PROTECTED_DIRS:
            return True
    
    return False


def find_files_by_pattern(directory: Path, pattern: str, 
                          recursive: bool = False, 
                          excludes: Set[str] = None) -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match (e.g., "*.log", "temp_*")
        recursive: Whether to search recursively
        excludes: Set of patterns to exclude
        
    Returns:
        List of Path objects matching the pattern
    """
    matches = []
    
    if recursive:
        pattern_path = directory / "**" / pattern
        files = glob.glob(str(pattern_path), recursive=True)
    else:
        pattern_path = directory / pattern
        files = glob.glob(str(pattern_path))
    
    for file_str in files:
        filepath = Path(file_str)
        
        # Only include files, not directories
        if not filepath.is_file():
            continue
        
        # Skip protected files
        if is_protected(filepath, excludes):
            continue
        
        matches.append(filepath)
    
    return matches


def get_files_interactively(directory: Path) -> List[Path]:
    """
    Present an interactive menu for file selection.
    
    Args:
        directory: Directory to list files from
        
    Returns:
        List of selected files to delete
    """
    # Get all non-protected files in directory
    all_files = [f for f in directory.iterdir() if f.is_file() and not is_protected(f)]
    
    if not all_files:
        print(f"No files found in {directory}")
        return []
    
    # Sort files by name
    all_files.sort(key=lambda x: x.name)
    
    print(f"\n{'='*60}")
    print(f"Files in {directory}")
    print(f"{'='*60}\n")
    
    # Display files with numbers
    for idx, filepath in enumerate(all_files, 1):
        size = filepath.stat().st_size
        size_str = format_size(size)
        print(f"{idx:3d}. {filepath.name:40s} ({size_str})")
    
    print(f"\n{'='*60}")
    print("Enter file numbers to delete (space-separated), 'all', or 'quit'")
    print("Example: 1 3 5 7")
    print(f"{'='*60}\n")
    
    user_input = input("Selection: ").strip().lower()
    
    if user_input == 'quit':
        print("Cancelled.")
        return []
    
    if user_input == 'all':
        return all_files
    
    # Parse selection numbers
    try:
        selected_indices = [int(x) for x in user_input.split()]
        selected_files = []
        
        for idx in selected_indices:
            if 1 <= idx <= len(all_files):
                selected_files.append(all_files[idx - 1])
            else:
                print(f"Warning: Invalid selection {idx} (out of range)")
        
        return selected_files
    
    except ValueError:
        print("Error: Invalid input. Please enter numbers separated by spaces.")
        return []


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def confirm_deletion(files: List[Path], dry_run: bool = False) -> bool:
    """
    Show files to be deleted and ask for confirmation.
    
    Args:
        files: List of files to delete
        dry_run: If True, skip actual deletion
        
    Returns:
        True if user confirms, False otherwise
    """
    if not files:
        print("No files to delete.")
        return False
    
    total_size = sum(f.stat().st_size for f in files if f.exists())
    
    print(f"\n{'='*60}")
    print(f"{'[DRY RUN] ' if dry_run else ''}Files to be deleted ({len(files)} files, {format_size(total_size)} total):")
    print(f"{'='*60}\n")
    
    for filepath in files:
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  - {filepath} ({format_size(size)})")
        else:
            print(f"  - {filepath} (not found)")
    
    print(f"\n{'='*60}")
    
    if dry_run:
        print("[DRY RUN] No files will be deleted.")
        return False
    
    response = input(f"\nDelete {len(files)} file(s)? (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def delete_files(files: List[Path], dry_run: bool = False) -> dict:
    """
    Delete files and return statistics.
    
    Args:
        files: List of files to delete
        dry_run: If True, don't actually delete
        
    Returns:
        Dictionary with deletion statistics
    """
    stats = {
        'deleted': 0,
        'failed': 0,
        'skipped': 0,
        'total_size': 0,
        'errors': []
    }
    
    for filepath in files:
        try:
            if not filepath.exists():
                stats['skipped'] += 1
                print(f"  Skip (not found): {filepath}")
                continue
            
            size = filepath.stat().st_size
            
            if dry_run:
                print(f"  [DRY RUN] Would delete: {filepath} ({format_size(size)})")
                stats['deleted'] += 1
                stats['total_size'] += size
            else:
                filepath.unlink()
                print(f"  Deleted: {filepath} ({format_size(size)})")
                stats['deleted'] += 1
                stats['total_size'] += size
        
        except Exception as e:
            stats['failed'] += 1
            error_msg = f"Failed to delete {filepath}: {str(e)}"
            stats['errors'].append(error_msg)
            print(f"  Error: {error_msg}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Safe utility for deleting multiple files from terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode: select files to delete from a list'
    )
    
    parser.add_argument(
        '--files', '-f',
        nargs='+',
        help='Specific files to delete (space-separated)'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        help='Delete files matching pattern (e.g., "*.log", "temp_*")'
    )
    
    parser.add_argument(
        '--directory', '-d',
        default='.',
        help='Directory to operate in (default: current directory)'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search recursively in subdirectories'
    )
    
    parser.add_argument(
        '--exclude', '-e',
        nargs='+',
        help='Patterns to exclude from deletion'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt (use with caution!)'
    )
    
    parser.add_argument(
        '--allow-python',
        action='store_true',
        help='Allow deletion of Python files (disabled by default for safety)'
    )
    
    args = parser.parse_args()
    
    # Convert directory to Path
    directory = Path(args.directory).resolve()
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    
    # Setup excludes
    excludes = set(args.exclude) if args.exclude else set()
    
    # Remove Python protection if explicitly allowed
    protected_patterns = PROTECTED_PATTERNS.copy()
    if args.allow_python:
        protected_patterns.discard('*.py')
    
    # Determine which files to delete
    files_to_delete = []
    
    if args.interactive:
        files_to_delete = get_files_interactively(directory)
    
    elif args.files:
        for file_str in args.files:
            filepath = Path(file_str)
            if not filepath.is_absolute():
                filepath = directory / filepath
            
            if is_protected(filepath, excludes):
                print(f"Warning: Skipping protected file {filepath}")
                continue
            
            if filepath.exists() and filepath.is_file():
                files_to_delete.append(filepath)
            else:
                print(f"Warning: File not found: {filepath}")
    
    elif args.pattern:
        files_to_delete = find_files_by_pattern(
            directory, 
            args.pattern, 
            args.recursive, 
            excludes
        )
    
    else:
        parser.print_help()
        sys.exit(0)
    
    if not files_to_delete:
        print("No files selected for deletion.")
        sys.exit(0)
    
    # Confirm and delete
    if args.yes or confirm_deletion(files_to_delete, args.dry_run):
        print(f"\n{'='*60}")
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Deleting files...")
        print(f"{'='*60}\n")
        
        stats = delete_files(files_to_delete, args.dry_run)
        
        print(f"\n{'='*60}")
        print("Deletion Summary:")
        print(f"{'='*60}")
        print(f"  {'Would delete' if args.dry_run else 'Deleted'}: {stats['deleted']} files ({format_size(stats['total_size'])})")
        print(f"  Failed: {stats['failed']} files")
        print(f"  Skipped: {stats['skipped']} files")
        
        if stats['errors']:
            print(f"\nErrors:")
            for error in stats['errors']:
                print(f"  - {error}")
        
        print(f"{'='*60}\n")
        
        if args.dry_run:
            print("[DRY RUN] No files were actually deleted.")
            print("Remove --dry-run flag to perform actual deletion.\n")
    else:
        print("\nDeletion cancelled.")


if __name__ == '__main__':
    main()
