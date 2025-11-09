#!/bin/bash
# Production setup script for DeepFake Detector
# This script sets up the development and production environment

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}DeepFake Detector - Production Setup${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION is installed${NC}"
else
    echo -e "${RED}✗ Python 3.8+ is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo ""

# Install package in development mode
echo -e "${YELLOW}[2/6] Installing package in development mode...${NC}"
python3 -m pip install -e ".[dev]" --quiet
echo -e "${GREEN}✓ Package installed successfully${NC}"
echo ""

# Install pre-commit hooks
echo -e "${YELLOW}[3/6] Installing pre-commit hooks...${NC}"
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
else
    echo -e "${YELLOW}⚠ pre-commit not found, installing...${NC}"
    python3 -m pip install pre-commit --quiet
    pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
fi
echo ""

# Run validation script
echo -e "${YELLOW}[4/6] Running validation checks...${NC}"
python3 scripts/validate.py
echo ""

# Run tests
echo -e "${YELLOW}[5/6] Running test suite...${NC}"
if python3 -m pytest tests/ --tb=short --quiet -v; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo -e "${YELLOW}This may be due to missing TensorFlow. Install it with:${NC}"
    echo -e "${YELLOW}  pip install tensorflow${NC}"
fi
echo ""

# Summary
echo -e "${YELLOW}[6/6] Setup summary${NC}"
echo -e "${GREEN}✓ Development environment configured${NC}"
echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
echo -e "${GREEN}✓ Validation checks passed${NC}"
echo ""

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}Setup Complete!${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo -e "  1. Run tests:          ${BLUE}hatch run test${NC}"
echo -e "  2. Run tests (parallel): ${BLUE}hatch run test-fast${NC}"
echo -e "  3. Check coverage:      ${BLUE}hatch run test-cov${NC}"
echo -e "  4. Format code:         ${BLUE}hatch run fmt${NC}"
echo -e "  5. Lint code:           ${BLUE}hatch run lint${NC}"
echo -e "  6. Run all checks:      ${BLUE}hatch run check${NC}"
echo ""
echo -e "${GREEN}CLI commands:${NC}"
echo -e "  - ${BLUE}deepfake-detector --help${NC}"
echo -e "  - ${BLUE}dfd --help${NC}"
echo ""
