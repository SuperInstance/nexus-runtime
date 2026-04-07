# Security Policy

## Supported Versions

The following versions of NEXUS are currently receiving security updates:

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2.0 | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in NEXUS, please report it privately
to maintain the safety of the project and its users.

**Please do not open a public issue.**

Instead, send a report to [INSERT SECURITY CONTACT EMAIL] with the following:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** — what an attacker could do
4. **Affected versions** — which versions are impacted
5. **Suggested fix** (optional) — any ideas for remediation

We will acknowledge receipt within 48 hours and aim to provide a resolution
timeline within 72 hours. Security fixes will be prioritized and shipped as
patch releases.

## Security Model Overview

NEXUS implements a layered security architecture designed for autonomous
robotics systems operating in safety-critical environments:

### 4-Tier Safety Architecture

1. **Hardware Safety Layer** — Watchdog timers, hardware interlocks, and
   heartbeat monitoring at the firmware level. The system can safely halt
   all actuators independent of software state.

2. **Firmware Safety State Machine** — A deterministic state machine in C
   that enforces safety invariants at the lowest software level. Transitions
   between safety states (e.g., IDLE, RUNNING, FAULT, E-STOP) follow strict
   validation rules.

3. **Trust Engine** — A Python-based trust scoring system that gates autonomy
   levels based on accumulated evidence. Nodes must earn trust before
   being granted higher autonomy. Trust can be revoked at any time.

4. **Policy Engine** — High-level safety automation policies that define
   per-domain constraints, audit trails, and bytecode-gated operation
   boundaries.

### Trust-Gated Autonomy

NEXUS uses a trust-gated autonomy model where:

- All nodes start at the lowest autonomy level
- Autonomy is earned through demonstrated safe behavior
- Trust increments are recorded and auditable
- Safety state machine transitions are always respected regardless of trust
   level
- Emergency stops bypass all trust mechanisms for immediate safety

### Wire Protocol Security

- All serial communication uses COBS framing with CRC-16 integrity checks
- Message dispatch validates opcode ranges and payload boundaries
- No authentication bypass is possible at the wire protocol level

## Responsible Disclosure

We follow coordinated disclosure practices. Once a vulnerability is confirmed:

1. We develop and test a fix
2. We release a security advisory and patched version
3. We credit the reporter (with their permission)

Thank you for helping keep NEXUS safe.
