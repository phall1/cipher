# Cipher Project Guidelines

## Build & Run Commands
- Setup environment: `python -m venv venv && source venv/bin/activate` (macOS/Linux)
- Install dependencies: `pip install pydantic`
- Run ingestion script: `python ingestion/main.py`
- Run single test: `pytest -xvs tests/path_to_test.py::test_function_name`
- Type checking: `mypy ingestion/`
- Lint code: `flake8 ingestion/`

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports with a blank line between groups
- **Formatting**: Use 4 spaces for indentation, 88-character line length
- **Types**: Use type hints for all function parameters and return values
- **Naming**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
- **Error Handling**: Use explicit exception handling with specific exception types
- **Documentation**: Docstrings for modules, classes, and functions using triple quotes
- **Organization**: Group related functionality into modules, follow Python package structure

This project processes news reports and organizes them by topic using Pydantic models.