"""Test suite for the DSL (Domain-Specific Language) interpreter.

This package contains all unit and integration tests for the DSL interpreter,
including tokenizer, parser, AST nodes, interpreter, context, providers,
manifest validation, and the high-level evaluate functions.

Test modules:
- test_tokenizer: Lexical analysis of DSL expressions.
- test_parser: Syntactic analysis and AST construction.
- test_ast_serialization: JSON serialization/deserialization of AST nodes.
- test_interpreter: Direct evaluation of AST nodes without parsing.
- test_integration: End-to-end tests with full pipeline and mock providers.
- test_evaluate: High-level evaluate_dsl and evaluate_dsl_async functions.
- test_manifest: Manifest structure and validator tests.
- test_providers: Provider implementations (InProcess, HTTP, AsyncHTTP).

Fixtures:
- conftest.py: Shared fixtures for mock providers, contexts, and interpreters.

The test suite uses pytest with markers:
- unit: Fast, isolated tests (tokenizer, parser, AST, interpreter).
- integration: Full pipeline tests with providers.
- async_test: Asynchronous tests using pytest-asyncio.
- with_providers / without_providers: Tests that require or avoid providers.
- error: Tests that verify error handling.
- tokenizer, parser, interpreter, manifest, provider,
    ast: Component-specific markers.
- evaluate: Tests for the evaluate module.

To run all tests:
    pytest

To run specific markers:
    pytest -m unit
    pytest -m integration
    pytest -m async_test

For more details, see the project documentation and pytest.ini.
"""
