# Quick File Deletion Guide

**Quick answer**: Yes! You can delete multiple files from the terminal using our utility script.

## 🚀 Quick Start

### Option A: Use the Shell Script Wrapper (Easiest!)

```bash
# Delete log files (with preview)
./scripts/clean.sh logs --dry-run
./scripts/clean.sh logs

# Interactive mode
./scripts/clean.sh interactive

# Delete by pattern
./scripts/clean.sh pattern "*.tmp" --dry-run
```

### Option B: Use Python Script Directly

#### Method 1: Interactive Mode

```bash
python scripts/delete_files.py --interactive
```

This shows you a numbered list of files, and you can select which ones to delete by entering their numbers.

#### Method 2: Delete by Pattern

```bash
# See what would be deleted first (safe preview)
python scripts/delete_files.py --pattern "*.log" --dry-run

# If it looks good, delete them
python scripts/delete_files.py --pattern "*.log"
```

#### Method 3: Delete Specific Files

```bash
python scripts/delete_files.py --files file1.txt file2.log file3.md
```

## 📋 Common Use Cases

### Delete All Log Files

```bash
# Using shell wrapper (easiest)
./scripts/clean.sh logs --dry-run  # Preview
./scripts/clean.sh logs            # Delete

# Or using Python directly
python scripts/delete_files.py --pattern "*.log" --dry-run
python scripts/delete_files.py --pattern "*.log"
```

### Delete Temporary Files

```bash
# Using shell wrapper
./scripts/clean.sh tmp

# Or using Python directly
python scripts/delete_files.py --pattern "*.tmp"
```

### Delete Old Documentation Files

```bash
# Interactive selection is best for documentation
./scripts/clean.sh interactive
# Or: python scripts/delete_files.py --interactive
```

### Delete Files Except Important Ones

```bash
python scripts/delete_files.py --pattern "*.md" --exclude README.md LICENSE.md
```

### Delete Files in a Specific Directory

```bash
python scripts/delete_files.py --pattern "*.log" --directory logs/
```

## ✅ Safety Features

- ✅ **Protected files**: Python scripts, git files, README, requirements.txt are automatically protected
- ✅ **Confirmation prompts**: Always asks before deleting (unless you use `--yes`)
- ✅ **Dry-run mode**: See what would be deleted without actually deleting (`--dry-run`)
- ✅ **Size display**: Shows how much space you'll free up

## 📚 Full Documentation

For complete documentation, options, and examples, see:
- [File Deletion Utility Documentation](docs/FILE_DELETION_UTILITY.md)

## 💡 Pro Tips

1. **Always preview first**: Use `--dry-run` to see what would be deleted
2. **Use patterns wisely**: Be specific with patterns like `"experiment_*.log"` instead of `"*"`
3. **Interactive is safe**: When in doubt, use `--interactive` mode
4. **Check the directory**: Make sure you're in the right directory before deleting

## 🆘 Help

To see all available options:

```bash
python scripts/delete_files.py --help
```

## ⚡ Examples

```bash
# Example 1: Clean up log files in root directory
cd /home/runner/work/ML-project/ML-project
python scripts/delete_files.py --pattern "*.log" --dry-run

# Example 2: Remove temporary files recursively
python scripts/delete_files.py --pattern "*.tmp" --recursive

# Example 3: Delete specific old files
python scripts/delete_files.py --files old_doc1.md old_doc2.md backup.txt

# Example 4: Interactive cleanup of root directory
python scripts/delete_files.py --interactive
```

## 🔗 Related Files

- Main utility: `scripts/delete_files.py`
- Full docs: `docs/FILE_DELETION_UTILITY.md`
- Cleanup history: `CLEANUP_SUMMARY.md`
