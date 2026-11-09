#!/usr/bin/env bash
# ==============================================================================
# Script: build.sh
# Purpose: Compile the modular LaTeX thesis report into main.pdf
# Works with: Tectonic (recommended) or XeLaTeX fallback
# ==============================================================================

set -euo pipefail

# Determine script directory (thesis_report/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " 🎓 FL-IoT-IDS Thesis Report PDF Builder"
echo "=========================================================="
echo "Working directory: $SCRIPT_DIR"

# 1. Detect LaTeX compiler
COMPILER=""
if command -v tectonic >/dev/null 2>&1; then
    COMPILER="tectonic"
elif [[ -x "$HOME/.local/bin/tectonic" ]]; then
    COMPILER="$HOME/.local/bin/tectonic"
elif command -v xelatex >/dev/null 2>&1; then
    COMPILER="xelatex"
else
    echo "❌ Error: Neither 'tectonic' nor 'xelatex' was found in PATH."
    echo "Please ensure tectonic is installed at $HOME/.local/bin/tectonic or in your PATH."
    exit 1
fi

COMPILER_VER=$("$COMPILER" --version 2>&1 | head -n 1)
echo "Using compiler: $COMPILER ($COMPILER_VER)"

# 2. Check essential modular section files
REQUIRED_SECTIONS=(
    "main.tex"
    "references.bib"
    "00_frontmatter/cover.tex"
    "00_frontmatter/abstract.tex"
    "00_frontmatter/acronyms.tex"
    "01_introduction/introduction.tex"
    "02_literature_review/literature_review.tex"
    "03_methodology/methodology.tex"
    "04_implementation/implementation.tex"
    "05_results_discussions/results_discussions.tex"
    "06_conclusions/conclusions.tex"
    "07_appendices/appendix_a.tex"
    "07_appendices/appendix_b.tex"
)

echo "Verifying modular section files..."
MISSING=0
for sec in "${REQUIRED_SECTIONS[@]}"; do
    if [[ ! -f "$sec" ]]; then
        echo " ⚠️  Missing: $sec"
        MISSING=1
    fi
done

if [[ $MISSING -eq 1 ]]; then
    echo "❌ Error: One or more required section files are missing. Aborting."
    exit 1
fi
echo "✅ All modular sections verified."

# 3. Compile
echo "Compiling main.tex..."
START_TIME=$(date +%s)

if [[ "$COMPILER" == *"tectonic"* ]]; then
    "$COMPILER" main.tex
else
    # Fallback multi-pass for standard xelatex + bibtex
    xelatex -interaction=nonstopmode main.tex
    bibtex main
    xelatex -interaction=nonstopmode main.tex
    xelatex -interaction=nonstopmode main.tex
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# 4. Verify output PDF
if [[ -f "main.pdf" ]]; then
    SIZE_BYTES=$(stat -c%s "main.pdf")
    SIZE_MB=$(awk -v b="$SIZE_BYTES" 'BEGIN { printf "%.2f MB", b / 1048576 }')
    
    # Inspect page count using PyPDF2 if available
    PAGES="N/A"
    if python3 -c "import PyPDF2" >/dev/null 2>&1; then
        PAGES=$(python3 -c "import PyPDF2; print(len(PyPDF2.PdfReader('main.pdf').pages))" 2>/dev/null || echo "N/A")
    fi

    echo "=========================================================="
    echo " 🎉 Build Successful!"
    echo " Output: $SCRIPT_DIR/main.pdf"
    echo " Size:   $SIZE_MB"
    echo " Pages:  $PAGES pages"
    echo " Time:   ${ELAPSED}s"
    echo "=========================================================="
else
    echo "❌ Error: main.pdf was not generated."
    exit 1
fi
