> **MIGRATION STATUS**: Copied from Edge-Native reference repo
> **LAST RECONCILED**: 2026-04-05
> **IMPLEMENTATION STATUS**: Reference document — applicable to all sprints

# Engineering Pitfalls

## Common Mistakes in Edge-Native Development

### 1. Ignoring Thermal Constraints
Jetson devices throttle under sustained load. Always monitor thermal zones
and implement graceful degradation.

### 2. Blocking the Main Control Loop
Never perform I/O in the main PWM/GPIO control loop. Use async patterns
or dedicated I/O threads.

### 3. Schema Drift
When modifying JSON schemas, always update both the schema file and the
code that consumes it in the same commit. Schema validation is your first
line of defense.

### 4. Hardcoded Paths
Use environment variables or config files for all file paths. The runtime
must work across different mount points on Jetson vs dev machines.

### 5. Skipping Host Tests
Always run tests with `-DNEXUS_HOST_TEST=ON` before deploying to hardware.
Host tests catch logic errors without requiring physical devices.

## Anti-Patterns
- **Global mutable state in Python control layer**: Use dependency injection
- **Synchronous sleep in control loops**: Use event-driven timers
- **Ignoring error codes from C runtime**: Always check return values
