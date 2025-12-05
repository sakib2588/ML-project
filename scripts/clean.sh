#!/bin/bash
# clean.sh - Quick cleanup wrapper for delete_files.py
#
# This script provides shortcuts for common cleanup operations

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DELETE_SCRIPT="${SCRIPT_DIR}/delete_files.py"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    cat << EOF
${GREEN}clean.sh${NC} - Quick cleanup wrapper for file deletion

${YELLOW}Usage:${NC}
    ./scripts/clean.sh [command] [options]

${YELLOW}Commands:${NC}
    logs                Delete all .log files
    tmp                 Delete all .tmp files  
    interactive         Interactive file selection
    pattern <PATTERN>   Delete files matching pattern
    help                Show this help

${YELLOW}Options:${NC}
    --dry-run          Preview without deleting
    --recursive        Search in subdirectories
    --directory <DIR>  Specify directory (default: current)

${YELLOW}Examples:${NC}
    # Preview log file deletion
    ./scripts/clean.sh logs --dry-run
    
    # Delete all log files
    ./scripts/clean.sh logs
    
    # Interactive cleanup
    ./scripts/clean.sh interactive
    
    # Delete files by custom pattern
    ./scripts/clean.sh pattern "PHASE*.md" --dry-run
    
    # Delete tmp files recursively
    ./scripts/clean.sh tmp --recursive

${YELLOW}Full documentation:${NC} docs/FILE_DELETION_UTILITY.md

EOF
}

if [ ! -f "$DELETE_SCRIPT" ]; then
    echo -e "${RED}Error: delete_files.py not found at ${DELETE_SCRIPT}${NC}"
    exit 1
fi

# No arguments - show help
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

COMMAND="$1"
shift

case "$COMMAND" in
    logs)
        echo -e "${BLUE}Cleaning log files (.log)...${NC}"
        python "$DELETE_SCRIPT" --pattern "*.log" "$@"
        ;;
    
    tmp)
        echo -e "${BLUE}Cleaning temporary files (.tmp, .temp)...${NC}"
        python "$DELETE_SCRIPT" --pattern "*.tmp" "$@"
        python "$DELETE_SCRIPT" --pattern "*.temp" "$@"
        ;;
    
    interactive|i)
        echo -e "${BLUE}Starting interactive file selection...${NC}"
        python "$DELETE_SCRIPT" --interactive "$@"
        ;;
    
    pattern|p)
        if [ -z "$1" ]; then
            echo -e "${RED}Error: Pattern required${NC}"
            echo "Usage: ./scripts/clean.sh pattern <PATTERN> [options]"
            exit 1
        fi
        PATTERN="$1"
        shift
        echo -e "${BLUE}Deleting files matching pattern: ${PATTERN}${NC}"
        python "$DELETE_SCRIPT" --pattern "$PATTERN" "$@"
        ;;
    
    help|--help|-h)
        show_help
        ;;
    
    *)
        echo -e "${RED}Unknown command: ${COMMAND}${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
