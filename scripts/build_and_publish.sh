#!/usr/bin/env bash
#
# Build and publish turkicnlp to PyPI.
#
# Usage:
#   ./scripts/build_and_publish.sh          # publish to PyPI (production)
#   ./scripts/build_and_publish.sh --test   # publish to TestPyPI first
#
# Prerequisites:
#   pip install build twine
#
# The script will prompt for your PyPI credentials (or API token) at upload time.
# To use an API token, enter "__token__" as the username and the token as the password.
#
# Alternatively, set up a ~/.pypirc file or use TWINE_USERNAME / TWINE_PASSWORD env vars.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
info "Checking prerequisites..."

command -v python3 >/dev/null 2>&1 || error "python3 is required but not found."

python3 -c "import build" 2>/dev/null || {
    warn "'build' package not found. Installing..."
    pip install build
}

python3 -c "import twine" 2>/dev/null || {
    warn "'twine' package not found. Installing..."
    pip install twine
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
USE_TEST_PYPI=false
if [[ "${1:-}" == "--test" ]]; then
    USE_TEST_PYPI=true
    info "Will publish to TestPyPI."
fi

# ---------------------------------------------------------------------------
# Extract version
# ---------------------------------------------------------------------------
VERSION=$(python3 -c "
import re
with open('turkicnlp/__init__.py') as f:
    match = re.search(r'__version__\s*=\s*[\"'\''](.*?)[\"'\''']', f.read())
    print(match.group(1) if match else 'unknown')
")
info "Package version: $VERSION"

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
info "Running tests..."
if python3 -m pytest turkicnlp/tests/ -q; then
    info "All tests passed."
else
    error "Tests failed. Fix them before publishing."
fi

# ---------------------------------------------------------------------------
# Clean previous builds
# ---------------------------------------------------------------------------
info "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info turkicnlp.egg-info/

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
info "Building source distribution and wheel..."
python3 -m build

info "Build artifacts:"
ls -lh dist/

# ---------------------------------------------------------------------------
# Verify the package
# ---------------------------------------------------------------------------
info "Checking package with twine..."
python3 -m twine check dist/*

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
echo ""
if $USE_TEST_PYPI; then
    info "Uploading to TestPyPI..."
    echo "You will be prompted for your TestPyPI credentials."
    echo "  Username: __token__"
    echo "  Password: your TestPyPI API token"
    echo ""
    python3 -m twine upload --repository testpypi dist/*

    echo ""
    info "Published to TestPyPI!"
    info "Install with: pip install --index-url https://test.pypi.org/simple/ turkicnlp==$VERSION"
else
    info "Uploading to PyPI..."
    echo "You will be prompted for your PyPI credentials."
    echo "  Username: __token__"
    echo "  Password: your PyPI API token"
    echo ""

    read -rp "Proceed with upload to PyPI? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        warn "Upload cancelled."
        info "Build artifacts are in dist/. Upload manually with: python3 -m twine upload dist/*"
        exit 0
    fi

    python3 -m twine upload dist/*

    echo ""
    info "Published to PyPI!"
    info "Install with: pip install turkicnlp==$VERSION"
fi

echo ""
info "Done."
