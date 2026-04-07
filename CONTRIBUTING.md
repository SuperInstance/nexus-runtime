# Contributing to NEXUS

Thank you for your interest in contributing to NEXUS! This guide covers everything you need to get started.

## Prerequisites

- **Python 3.11+** — Required for all Python-based modules, tests, and tooling
- **ESP-IDF** (optional) — Only needed for building and flashing ESP32 firmware
- **gcc + cmake** (optional) — Needed for building and running C firmware tests on host

## Setup

```bash
# Clone the repository
git clone https://github.com/<org>/nexus-runtime.git
cd nexus-runtime

# Install Python dependencies
pip install pytest ruff mypy

# Verify the setup
python -m pytest --tb=short -q
```

To build and run firmware tests (requires gcc and cmake):

```bash
mkdir -p tests/firmware/build
cd tests/firmware/build
cmake .. -DCMAKE_SOURCE_DIR=../..
make -j$(nproc)
./test_firmware
```

## Code Style

We use **ruff** for linting and **mypy** for type checking:

```bash
# Lint Python code
ruff check nexus/ jetson/

# Type check Python code
mypy nexus/ jetson/
```

Please fix any linting or type errors before submitting a PR.

## Project Structure

| Directory | Description |
|-----------|-------------|
| `nexus/` | Core Python runtime — VM, wire protocol, trust engine, orchestrator |
| `jetson/` | Jetson companion modules — navigation, vision, swarm, RL, security, compliance |
| `hardware/` | Board configuration profiles for 11+ platform families (50+ boards) |
| `firmware/` | C firmware source — VM, wire protocol, drivers, safety state machine |
| `shared/` | Shared opcodes and schemas used by both Python and C sides |
| `schemas/` | JSON Schema definitions for configuration and protocol validation |
| `tests/` | Test suites — Python unit/integration tests, C firmware tests (Unity) |
| `docs/` | Architecture plans, ADRs, build guides, planning docs |

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Make changes** following the code style guidelines above
3. **Test** — ensure all tests pass, no regressions introduced
4. **Document** — update docs and CHANGELOG.md if applicable
5. **Submit** a PR with a clear description using our [PR template](.github/PULL_REQUEST_TEMPLATE.md)
6. **Review** — address reviewer feedback; CI must pass before merge

## Reporting Issues

Please use our [bug report](.github/ISSUE_TEMPLATE/bug_report.md) and [feature request](.github/ISSUE_TEMPLATE/feature_request.md) templates when opening issues. Include your board type, firmware version, and steps to reproduce.

## Code of Conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
