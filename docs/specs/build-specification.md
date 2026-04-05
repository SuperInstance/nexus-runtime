> **MIGRATION STATUS**: Copied from Edge-Native reference repo
> **LAST RECONCILED**: 2026-04-05
> **IMPLEMENTATION STATUS**: Directory layout matches (jetson/, docs/, firmware/); schemas/ not yet populated; CI workflows being added in Sprint 2A.3

# Build Specification

## Build Targets

### Host (Development)
- Platform: Ubuntu 22.04 / macOS
- Compiler: GCC 11+ / Clang 14+
- Python: 3.11
- CMake flags: `-DNEXUS_HOST_TEST=ON`

### Target (Jetson)
- Platform: Jetson Orin NX
- Compiler: aarch64-linux-gnu-gcc
- JetPack: 5.1.2+
- CMake flags: `-DNEXUS_JETSON_TARGET=ON`

## Directory Layout
```
nexus-runtime/
├── CMakeLists.txt          # Root cmake
├── jetson/                 # Python control layer
│   ├── test_*.py
│   └── *.py
├── schemas/                # JSON Schema definitions
├── docs/                   # Documentation
├── src/                    # C runtime source
│   └── nexus_core/
└── build/                  # Build output (gitignored)
```

## Dependencies
- jsonschema (Python)
- pytest (Python)
- pydantic (Python, optional)

## Configuration
All runtime configuration uses JSON files validated against schemas in `schemas/`.
The Python layer reads config at startup and validates before any hardware interaction.
