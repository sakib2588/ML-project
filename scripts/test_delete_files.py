#!/usr/bin/env python3
"""
Test suite for delete_files.py utility

This test suite validates the functionality of the file deletion utility
including pattern matching, file protection, and deletion operations.

Usage:
    python scripts/test_delete_files.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add scripts directory to path to import delete_files module
sys.path.insert(0, str(Path(__file__).parent))

try:
    import delete_files
except ImportError:
    print("Error: Could not import delete_files module")
    sys.exit(1)


class TestDeleteFiles:
    """Test cases for delete_files.py utility"""
    
    def __init__(self):
        self.test_dir = None
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def setup(self):
        """Create temporary test directory with test files"""
        self.test_dir = Path(tempfile.mkdtemp(prefix='test_delete_'))
        
        # Create test files
        test_files = [
            'test1.log',
            'test2.log',
            'test3.tmp',
            'test_data.txt',
            'README.md',
            'important.py',
            'requirements.txt',
        ]
        
        for filename in test_files:
            filepath = self.test_dir / filename
            filepath.write_text(f"Test content for {filename}")
        
        # Create subdirectory with more files
        subdir = self.test_dir / 'subdir'
        subdir.mkdir()
        (subdir / 'nested.log').write_text("Nested log file")
        (subdir / 'nested.tmp').write_text("Nested temp file")
        
        print(f"Test directory created: {self.test_dir}")
        return self.test_dir
    
    def teardown(self):
        """Clean up test directory"""
        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            print(f"Test directory cleaned up: {self.test_dir}")
    
    def assert_true(self, condition, message):
        """Assert that condition is true"""
        if condition:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            print(f"  ✗ {message}")
    
    def assert_equal(self, actual, expected, message):
        """Assert that actual equals expected"""
        if actual == expected:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            print(f"  ✗ {message} (expected {expected}, got {actual})")
    
    def test_is_protected(self):
        """Test file protection logic"""
        print("\n[Test 1: File Protection]")
        
        # Protected files
        self.assert_true(
            delete_files.is_protected(Path("README.md")),
            "README.md should be protected"
        )
        self.assert_true(
            delete_files.is_protected(Path("requirements.txt")),
            "requirements.txt should be protected"
        )
        self.assert_true(
            delete_files.is_protected(Path("test.py")),
            "Python files should be protected by default"
        )
        
        # Non-protected files
        self.assert_true(
            not delete_files.is_protected(Path("test.log")),
            "Log files should not be protected"
        )
        self.assert_true(
            not delete_files.is_protected(Path("temp.tmp")),
            "Temp files should not be protected"
        )
    
    def test_find_files_by_pattern(self):
        """Test pattern matching"""
        print("\n[Test 2: Pattern Matching]")
        
        # Find log files
        log_files = delete_files.find_files_by_pattern(
            self.test_dir, "*.log", recursive=False
        )
        self.assert_equal(
            len(log_files), 2,
            "Should find 2 log files in test directory"
        )
        
        # Find tmp files
        tmp_files = delete_files.find_files_by_pattern(
            self.test_dir, "*.tmp", recursive=False
        )
        self.assert_equal(
            len(tmp_files), 1,
            "Should find 1 tmp file in test directory"
        )
        
        # Recursive search
        log_files_recursive = delete_files.find_files_by_pattern(
            self.test_dir, "*.log", recursive=True
        )
        self.assert_equal(
            len(log_files_recursive), 3,
            "Should find 3 log files recursively"
        )
    
    def test_delete_files(self):
        """Test file deletion"""
        print("\n[Test 3: File Deletion]")
        
        # Create files to delete
        test_file1 = self.test_dir / 'delete_me1.txt'
        test_file2 = self.test_dir / 'delete_me2.txt'
        test_file1.write_text("Delete me")
        test_file2.write_text("Delete me too")
        
        self.assert_true(
            test_file1.exists(),
            "Test file 1 should exist before deletion"
        )
        self.assert_true(
            test_file2.exists(),
            "Test file 2 should exist before deletion"
        )
        
        # Delete files
        stats = delete_files.delete_files([test_file1, test_file2], dry_run=False)
        
        self.assert_equal(
            stats['deleted'], 2,
            "Should delete 2 files"
        )
        self.assert_equal(
            stats['failed'], 0,
            "Should have 0 failures"
        )
        self.assert_true(
            not test_file1.exists(),
            "Test file 1 should not exist after deletion"
        )
        self.assert_true(
            not test_file2.exists(),
            "Test file 2 should not exist after deletion"
        )
    
    def test_dry_run(self):
        """Test dry-run mode"""
        print("\n[Test 4: Dry-Run Mode]")
        
        # Create a test file
        test_file = self.test_dir / 'dry_run_test.txt'
        test_file.write_text("Should not be deleted")
        
        # Dry-run delete
        stats = delete_files.delete_files([test_file], dry_run=True)
        
        self.assert_equal(
            stats['deleted'], 1,
            "Dry-run should report 1 file would be deleted"
        )
        self.assert_true(
            test_file.exists(),
            "File should still exist after dry-run"
        )
    
    def test_protected_files_exclusion(self):
        """Test that protected files are not included in pattern matches"""
        print("\n[Test 5: Protected Files Exclusion]")
        
        # Find all Python files
        py_files = delete_files.find_files_by_pattern(
            self.test_dir, "*.py", recursive=False
        )
        
        self.assert_equal(
            len(py_files), 0,
            "Should find 0 Python files (protected by default)"
        )
        
        # README should not be in markdown matches
        md_files = delete_files.find_files_by_pattern(
            self.test_dir, "*.md", recursive=False
        )
        
        self.assert_equal(
            len(md_files), 0,
            "Should find 0 markdown files (README.md is protected)"
        )
    
    def test_format_size(self):
        """Test file size formatting"""
        print("\n[Test 6: Size Formatting]")
        
        self.assert_equal(
            delete_files.format_size(100),
            "100.0 B",
            "Should format bytes correctly"
        )
        self.assert_equal(
            delete_files.format_size(1024),
            "1.0 KB",
            "Should format kilobytes correctly"
        )
        self.assert_equal(
            delete_files.format_size(1024 * 1024),
            "1.0 MB",
            "Should format megabytes correctly"
        )
    
    def run_all_tests(self):
        """Run all test cases"""
        print("=" * 60)
        print("Running Delete Files Utility Test Suite")
        print("=" * 60)
        
        try:
            self.setup()
            
            self.test_is_protected()
            self.test_find_files_by_pattern()
            self.test_delete_files()
            self.test_dry_run()
            self.test_protected_files_exclusion()
            self.test_format_size()
            
        finally:
            self.teardown()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"  Passed: {self.passed}")
        print(f"  Failed: {self.failed}")
        print(f"  Total:  {self.passed + self.failed}")
        print("=" * 60)
        
        if self.failed == 0:
            print("\n✓ All tests passed!")
            return 0
        else:
            print(f"\n✗ {self.failed} test(s) failed")
            return 1


def main():
    """Main entry point"""
    test_suite = TestDeleteFiles()
    exit_code = test_suite.run_all_tests()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
