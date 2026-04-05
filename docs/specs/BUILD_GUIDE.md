> **MIGRATION STATUS**: Copied from Edge-Native reference repo
> **LAST RECONCILED**: 2026-04-05
> **IMPLEMENTATION STATUS**: CMakeLists.txt root exists; jetson/ Python layer present; firmware/ C build active; pytest tests in tests/jetson/

# Senior Engineer Build Guide

## Prerequisites
- CMake 3.16+
- Python 3.11
- Ninja build system
- JetPack SDK (for target builds)

## Quick Start

```bash
mkdir build && cd build
cmake .. -DNEXUS_HOST_TEST=ON
cmake --build . --parallel
ctest --output-on-failure
```

## Python Layer Setup

```bash
cd jetson
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -v
```

## Architecture Notes
- The C runtime (`nexus_core`) compiles to a shared library
- Python layer loads it via ctypes
- All hardware access goes through the C abstraction layer
- Configuration files validated against JSON Schema before use

## Debug Builds
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DNEXUS_HOST_TEST=ON
```

## Release Builds
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DNEXUS_JETSON_TARGET=ON
```

## Testing Strategy
1. Unit tests via pytest for Python layer
2. CTest for C runtime unit tests
3. Integration tests require Jetson hardware
