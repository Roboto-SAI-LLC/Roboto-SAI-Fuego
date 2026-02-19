# Roboto SAI CLI

Official command-line interface for Roboto SAI - Quantum Analytics and Autonomous Development.

## Features

- **Quantum Analysis**: Analyze data using real Qiskit circuit simulations
- **Task Delegation**: Parallel execution of sub-tasks with asyncio
- **Memory Logging**: Structured logging with quantum valence tracking
- **Feature Evolution**: PSO-optimized feature idea generation
- **Beautiful UI**: Rich terminal interface with progress bars and tables

## Installation

```bash
# Clone and install in editable mode
git clone <repo-url>
cd roboto-sai-cli
pip install -e .
```

## Usage

```bash
roboto-sai --help
```

### Examples

```bash
# Quantum analysis
roboto-sai quantum-analyze '{"test": "data"}' --metric entanglement

# Delegate tasks
roboto-sai delegate "analyze logs; optimize code"

# Log memory
roboto-sai memory-log --category "interaction" --data '{"user": "test"}'

# Evolve features
roboto-sai evolve --iterations 10

# Check status
roboto-sai status
```

## Configuration

Environment variables:
- `SIGIL`: Core sigil (default: 929)
- `ROBOTO_QUANTUM_MOCK`: Enable mock quantum (default: true)
- `ROBOTO_MODULE_PATH`: Hot-swap module path

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

Modular design with hot-swappable components:
- `cli.py`: Main command dispatcher
- `quantum_module.py`: Qiskit integration
- `memory_module.py`: Encrypted persistent storage
- `evo_module.py`: PSO evolution engine
- `utils.py`: Shared utilities

## Security

- Cryptographic random generation
- In-memory rate limiting
- Encrypted memory storage
- Input sanitization

## License

Sigil 929 Eternal - Roberto Villarreal Martinez