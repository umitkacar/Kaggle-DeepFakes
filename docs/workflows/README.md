# GitHub Actions Workflows

This directory contains CI/CD workflows for the project.

## ⚠️ Note on Workflow Files

The workflow files (`ci.yml` and `release.yml`) are included in the repository but cannot be pushed via GitHub Apps without the `workflows` permission.

### Manual Setup Required

To enable GitHub Actions:

1. **Navigate to your repository on GitHub**
2. **Create workflows manually** or enable GitHub Actions in Settings
3. **Copy the workflow files** from this directory to `.github/workflows/` on GitHub

## Available Workflows

### 📋 CI Workflow (`ci.yml`)

Automated testing and quality checks on every push and pull request.

**Features:**
- Multi-version Python testing (3.8, 3.9, 3.10, 3.11)
- Code quality checks (pre-commit hooks)
- Test coverage reporting
- Package building and validation

**Triggers:**
- Push to main/master/develop branches
- Pull requests to main/master/develop branches

### 🚀 Release Workflow (`release.yml`)

Automated PyPI publishing on new releases.

**Features:**
- Builds package with Hatch
- Publishes to PyPI using trusted publishing

**Triggers:**
- New GitHub release is published

## Workflow Files Location

The complete workflow files are available in this directory:

- `ci.yml` - Continuous Integration workflow
- `release.yml` - Release and PyPI publishing workflow

## Setting Up Workflows

### Option 1: Manual Creation

1. Go to your repository on GitHub
2. Click on "Actions" tab
3. Click "New workflow"
4. Click "set up a workflow yourself"
5. Copy content from the respective `.yml` file
6. Commit the file

### Option 2: Git Push (After Granting Permissions)

If you have admin access to the repository:

1. Grant workflow permissions in repository settings
2. Push the files using git

## Testing Locally

You can test the workflow steps locally:

```bash
# Install dependencies
pip install -e ".[dev]"

# Run quality checks (same as CI)
pre-commit run --all-files

# Run tests (same as CI)
hatch run test

# Build package (same as CI)
hatch build
```

## Required Secrets

For the release workflow to work, configure these in repository settings:

- **PyPI Publishing**: Set up trusted publishing or add `PYPI_API_TOKEN`

## More Information

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Hatch Documentation](https://hatch.pypa.io/)
- [pre-commit Documentation](https://pre-commit.com/)
