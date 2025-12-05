# File Deletion Utility

A safe and powerful command-line utility for deleting multiple files from the terminal with various modes and safety features.

## 📍 Location

```bash
scripts/delete_files.py
```

## 🎯 Purpose

This utility provides a safe way to delete multiple files from the command line, especially useful for cleaning up:
- Log files (*.log)
- Temporary files (*.tmp, *.temp)
- Backup files (*.bak, *~)
- Old documentation files
- Build artifacts
- Any files matching specific patterns

## 🔒 Safety Features

1. **Protected Files**: Critical files are automatically protected:
   - `.git/` directory and all git files
   - `requirements.txt`
   - `README.md`
   - `LICENSE`
   - All Python scripts (`.py`) by default
   - Virtual environments
   - `node_modules/`

2. **Confirmation Prompts**: Always asks for confirmation before deleting (unless `--yes` flag is used)

3. **Dry Run Mode**: Preview what would be deleted without actually deleting anything

4. **Exclude Patterns**: Specify files or patterns to exclude from deletion

5. **Size Display**: Shows file sizes before deletion

## 📖 Usage

### Basic Syntax

```bash
python scripts/delete_files.py [OPTIONS]
```

### Modes

#### 1. Interactive Mode

Select files to delete from an interactive menu:

```bash
python scripts/delete_files.py --interactive
```

Or shorter:
```bash
python scripts/delete_files.py -i
```

**Example Session:**
```
============================================================
Files in /home/runner/work/ML-project/ML-project
============================================================

  1. ARCH_SETUP.md                          (6.3 KB)
  2. CHECKLIST.md                           (6.6 KB)
  3. CLEANUP_SUMMARY.md                     (6.0 KB)
  4. file1.log                              (1.2 KB)
  5. file2.log                              (834 B)
  6. temp_data.tmp                          (45.2 KB)

============================================================
Enter file numbers to delete (space-separated), 'all', or 'quit'
Example: 1 3 5 7
============================================================

Selection: 4 5 6
```

#### 2. Pattern Mode

Delete all files matching a specific pattern:

```bash
# Delete all .log files in current directory
python scripts/delete_files.py --pattern "*.log"

# Delete all .tmp files recursively
python scripts/delete_files.py --pattern "*.tmp" --recursive

# Delete all files starting with "temp_"
python scripts/delete_files.py --pattern "temp_*"
```

#### 3. File List Mode

Delete specific files by name:

```bash
python scripts/delete_files.py --files file1.txt file2.log old_doc.md
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--interactive` | `-i` | Interactive file selection mode |
| `--files FILE [FILE ...]` | `-f` | Delete specific files |
| `--pattern PATTERN` | `-p` | Delete files matching glob pattern |
| `--directory DIR` | `-d` | Directory to operate in (default: current) |
| `--recursive` | `-r` | Search recursively in subdirectories |
| `--exclude PATTERN [PATTERN ...]` | `-e` | Exclude files matching these patterns |
| `--dry-run` | | Preview without deleting |
| `--yes` | `-y` | Skip confirmation (use with caution!) |
| `--allow-python` | | Allow deletion of Python files |

## 📝 Examples

### Example 1: Delete All Log Files (Preview First)

```bash
# First, see what would be deleted
python scripts/delete_files.py --pattern "*.log" --dry-run

# If satisfied, delete them
python scripts/delete_files.py --pattern "*.log"
```

### Example 2: Clean Up Temporary Files

```bash
# Delete all temporary files recursively
python scripts/delete_files.py --pattern "*.tmp" --recursive
```

### Example 3: Delete Specific Files

```bash
python scripts/delete_files.py --files old_file1.txt old_file2.md backup.bak
```

### Example 4: Delete Files with Exclusions

```bash
# Delete all .md files except README.md and IMPORTANT.md
python scripts/delete_files.py --pattern "*.md" --exclude README.md IMPORTANT.md
```

### Example 5: Delete from Specific Directory

```bash
# Delete all .log files from logs/ directory
python scripts/delete_files.py --pattern "*.log" --directory logs/
```

### Example 6: Delete Multiple File Types

```bash
# Delete all .log and .tmp files
python scripts/delete_files.py --pattern "*.log"
python scripts/delete_files.py --pattern "*.tmp"
```

### Example 7: Non-Interactive Deletion (Automated)

```bash
# Delete without confirmation (use carefully!)
python scripts/delete_files.py --pattern "*.log" --yes
```

## 🎬 Real-World Scenarios

### Scenario 1: Clean Up After Experiments

```bash
# Remove all experiment log files older than needed
python scripts/delete_files.py --pattern "fold*.log" --directory .
```

### Scenario 2: Remove Old Documentation

```bash
# Interactive selection of documentation files to remove
cd /home/runner/work/ML-project/ML-project
python scripts/delete_files.py --interactive
```

### Scenario 3: Clean Build Artifacts

```bash
# Remove all temporary build files
python scripts/delete_files.py --pattern "*.pyc" --recursive
python scripts/delete_files.py --pattern "__pycache__" --recursive
```

### Scenario 4: Bulk Delete by Pattern with Safety Check

```bash
# First dry-run to see what would be deleted
python scripts/delete_files.py --pattern "PHASE*.md" --dry-run

# Review the list, then execute
python scripts/delete_files.py --pattern "PHASE*.md"
```

## ⚠️ Safety Guidelines

1. **Always Use Dry Run First**: When deleting files by pattern, always run with `--dry-run` first:
   ```bash
   python scripts/delete_files.py --pattern "*.log" --dry-run
   ```

2. **Be Specific**: Use specific patterns instead of wildcards when possible:
   ```bash
   # Good: Specific pattern
   python scripts/delete_files.py --pattern "experiment_*.log"
   
   # Risky: Too broad
   python scripts/delete_files.py --pattern "*"
   ```

3. **Use Excludes**: Protect important files explicitly:
   ```bash
   python scripts/delete_files.py --pattern "*.md" --exclude README.md LICENSE.md
   ```

4. **Check the Directory**: Always verify you're in the right directory:
   ```bash
   pwd  # Check current directory
   python scripts/delete_files.py --pattern "*.tmp"
   ```

5. **Avoid --yes in Scripts**: The `--yes` flag skips confirmation. Use it carefully:
   ```bash
   # Dangerous: No confirmation
   python scripts/delete_files.py --pattern "*.log" --yes
   ```

## 🛡️ Protected Files

The following are automatically protected from deletion:

- `.git/` and all git-related files
- `.gitignore`
- `requirements.txt`
- `README.md`
- `LICENSE`
- `*.py` files (unless `--allow-python` is used)
- `.venv/`, `venv/` directories
- `node_modules/` directory
- `__pycache__/` directories

## 🔧 Advanced Usage

### Custom Exclude Patterns

```bash
# Delete all markdown files except those starting with "README" or "LICENSE"
python scripts/delete_files.py --pattern "*.md" --exclude "README*" "LICENSE*"
```

### Combining with Shell Commands

```bash
# Count files before deletion
ls *.log | wc -l

# Delete them
python scripts/delete_files.py --pattern "*.log"

# Verify deletion
ls *.log 2>/dev/null || echo "All log files deleted"
```

### Working with Multiple Directories

```bash
# Delete log files from multiple directories
for dir in logs/ experiments/ artifacts/; do
    python scripts/delete_files.py --pattern "*.log" --directory "$dir"
done
```

## 📊 Output Examples

### Dry Run Output

```
============================================================
[DRY RUN] Files to be deleted (3 files, 52.1 KB total):
============================================================

  - /path/to/file1.log (1.2 KB)
  - /path/to/file2.log (834 B)
  - /path/to/temp.tmp (50.0 KB)

============================================================

[DRY RUN] No files will be deleted.
```

### Deletion Summary

```
============================================================
Deleting files...
============================================================

  Deleted: file1.log (1.2 KB)
  Deleted: file2.log (834 B)
  Deleted: temp.tmp (50.0 KB)

============================================================
Deletion Summary:
============================================================
  Deleted: 3 files (52.1 KB)
  Failed: 0 files
  Skipped: 0 files
============================================================
```

## 🐛 Troubleshooting

### Issue: "Permission Denied"

**Cause**: File is read-only or you don't have delete permissions.

**Solution**: 
```bash
# Check file permissions
ls -la filename

# Add write permission if needed
chmod u+w filename

# Then try deleting again
python scripts/delete_files.py --files filename
```

### Issue: "File Not Found"

**Cause**: File path is incorrect or file doesn't exist.

**Solution**:
```bash
# Verify file exists
ls -la filename

# Use absolute path or correct relative path
python scripts/delete_files.py --files /absolute/path/to/file
```

### Issue: Python Files Are Protected

**Cause**: By default, all `.py` files are protected from deletion.

**Solution**:
```bash
# Use --allow-python flag to enable Python file deletion
python scripts/delete_files.py --pattern "old_*.py" --allow-python
```

## 🔗 Related Documentation

- [Cleanup Summary](../CLEANUP_SUMMARY.md) - Previous cleanup operations
- [Project README](../README.md) - Main project documentation
- [Scripts Directory](../scripts/) - Other utility scripts

## 💡 Tips

1. **Always preview first**: Use `--dry-run` to see what would be deleted
2. **Use tab completion**: Bash/shell tab completion works with file names
3. **Combine with find**: For complex patterns, use `find` first to locate files
4. **Keep backups**: Before bulk deletions, ensure you have backups of important data
5. **Use version control**: Git tracks deleted files, so you can recover them if needed

## 🤝 Contributing

If you find issues or want to add features:
1. Test thoroughly with `--dry-run`
2. Add safety checks for new features
3. Update this documentation
4. Consider edge cases (permissions, symlinks, etc.)

## 📄 License

This utility is part of the ML-project repository and follows the same license.
