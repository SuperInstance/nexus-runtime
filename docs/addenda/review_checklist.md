> **MIGRATION STATUS**: Copied from Edge-Native reference repo
> **LAST RECONCILED**: 2026-04-05
> **IMPLEMENTATION STATUS**: Reference document — active for all PRs

# Code Review Checklist

## General
- [ ] Code follows existing project conventions
- [ ] No hardcoded values that should be configurable
- [ ] Error handling covers all failure modes
- [ ] No debug print statements left in code

## Python Layer
- [ ] Type annotations on public functions
- [ ] JSON Schema validation for all config inputs
- [ ] Async patterns used for I/O operations
- [ ] Unit tests added for new functionality
- [ ] No blocking calls in control loops

## C Runtime
- [ ] Memory leaks checked (valgrind on host tests)
- [ ] Return values checked by callers
- [ ] Thread safety considered for shared state
- [ ] Platform-specific code properly guarded with `#ifdef`

## Configuration & Schemas
- [ ] Schema changes are backward compatible
- [ ] Default values documented in schema
- [ ] New config fields have sensible defaults

## Safety
- [ ] No bypass of safety checks
- [ ] Failsafe behavior unchanged or explicitly improved
- [ ] Thermal and current limits respected
- [ ] Watchdog still armed after changes

## Documentation
- [ ] Public API documented
- [ ] Architecture decisions recorded if applicable
- [ ] Migration notes provided for breaking changes
