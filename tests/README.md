# Tests Directory

This directory contains all test files for the MemGen project.

## 📁 Test Files

### Memory System Tests
- **test_basic_memory.py** - Basic memory operations
  - Tests memory storage and retrieval
  - Validates memory format

- **test_memory_injection.py** - Memory injection into model
  - Tests memory token injection
  - Validates memory integration with model forward pass

- **test_memory_retrieval.py** - Memory retrieval functionality
  - Tests similarity search
  - Validates retrieval quality

- **test_memory_standalone.py** - Standalone memory tests
  - Independent memory system tests
  - No model dependencies

- **test_model_memory.py** - Model-level memory tests
  - Tests MemGenModel with memory
  - End-to-end memory flow

### Component Tests
- **test_self_evolving_components.py** - Self-evolving RAG components
  - Tests Trigger and Weaver modules
  - Validates component interactions

- **test_save_load_fix.py** - Model save/load functionality
  - Tests checkpoint saving
  - Validates model loading

### Quick Tests
- **test_quick.py** - Quick smoke tests
  - Fast sanity checks
  - Basic functionality validation

- **test_small_training.py** - Small-scale training test
  - Tests training loop with minimal data
  - Validates training pipeline

## 🚀 Running Tests

### Run All Tests
```bash
# From project root
python -m pytest tests/

# Or run individually
python tests/test_basic_memory.py
```

### Run Specific Test Categories
```bash
# Memory tests only
python -m pytest tests/test_*memory*.py

# Component tests
python -m pytest tests/test_self_evolving_components.py

# Quick smoke test
python tests/test_quick.py
```

## 📝 Test Coverage

### Memory System ✅
- [x] Basic memory operations
- [x] Memory injection
- [x] Memory retrieval
- [x] Standalone memory functionality
- [x] Model-level memory integration

### Training Pipeline ✅
- [x] Small-scale training
- [x] Component interactions
- [x] Save/load functionality

### Quick Validation ✅
- [x] Smoke tests
- [x] Basic sanity checks

## 🔧 Adding New Tests

When adding new tests:
1. Follow the naming convention: `test_*.py`
2. Include docstrings explaining what is being tested
3. Use descriptive test function names
4. Add assertions with clear error messages

Example:
```python
def test_memory_retrieval():
    """Test that memory retrieval returns relevant experiences."""
    # Setup
    memory_store = MemoryStore()

    # Test
    results = memory_store.retrieve(query="test query", top_k=5)

    # Assert
    assert len(results) <= 5, "Should return at most top_k results"
    assert all(isinstance(r, dict) for r in results), "Results should be dicts"
```

## 🐛 Debugging Failed Tests

If tests fail:
1. Check the error message and stack trace
2. Run the test in isolation: `python tests/test_name.py`
3. Add print statements or use debugger
4. Check if dependencies are installed: `pip install -r requirements.txt`
5. Verify environment setup: `source scripts/setup_env.sh`

## 📞 Related Directories

- [../scripts/](../scripts/) - Executable scripts
- [../memgen/](../memgen/) - Core library code
- [../docs/](../docs/) - Documentation

## 🔗 For More Information

See the main [README](../README.md) for project overview and setup instructions.
