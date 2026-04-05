> **MIGRATION STATUS**: Copied from Edge-Native reference repo
> **LAST RECONCILED**: 2026-04-05
> **IMPLEMENTATION STATUS**: ADR-001 (CMake) implemented; ADR-002 (Python layer) implemented in jetson/; ADR-003 (JSON Schema) pending; ADR-004 (monorepo) in effect

# Architecture Decision Records (ADR)

## ADR-001: Choice of CMake as Build System
- **Status**: Accepted
- **Context**: Need cross-platform build support for Jetson and x86 targets
- **Decision**: Use CMake 3.16+ with Ninja generator
- **Consequences**: Standardized build, good IDE support, cross-compilation friendly

## ADR-002: Python-based Jetson Control Layer
- **Status**: Accepted
- **Context**: Need rapid iteration on GPIO/PWM control for edge devices
- **Decision**: Python 3.11 with ctypes bindings to C runtime
- **Consequences**: Faster development cycle, type-safe interfaces via pydantic

## ADR-003: JSON Schema for Configuration Validation
- **Status**: Accepted
- **Context**: Runtime configuration must be validated before hardware interaction
- **Decision**: JSON Schema Draft-7 for all config files under `schemas/`
- **Consequences**: Language-agnostic validation, good tooling support

## ADR-004: Monorepo Structure
- **Status**: Proposed
- **Context**: Keep C runtime, Python control layer, and docs in sync
- **Decision**: Single repository with `jetson/`, `schemas/`, `docs/` top-level dirs
- **Consequences**: Atomic commits across layers, simpler dependency management
