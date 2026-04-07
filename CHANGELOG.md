# Changelog

All notable changes to the NEXUS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - Unreleased

### Added

- **50+ board configurations** across 11 platform families:
  - Arduino (UNO, Mega, Nano, Nano 33 IoT, Due, MKR WiFi)
  - ESP32, ESP32-S3, ESP32-C3, ESP32-C6, ESP32-H2
  - Raspberry Pi (3B, 4, 5, CM4, Pi 400, Pi Zero W, Pico 2)
  - Teensy 3.6, 4.0, 4.1
  - STM32F4, STM32H7
  - Jetson Nano, TX2, Xavier NX, Orin Nano, Orin NX, AGX Orin
  - NRF52810, NRF52832, NRF52840, NRF5340
  - IMX RT1050, RT1060, RT1064, RT1170
  - ESP8266, D1 Mini
  - RP2040, Pico W
  - BeagleBone Black, AI64
- **Hardware Discovery API** for automatic board detection and configuration
- **Trust Engine** with incremental trust scoring, trust levels, and attestation
- **Safety State Machine** with deterministic state transitions for autonomous operation
- **Wire Protocol** with COBS framing, CRC-16 integrity, and structured message dispatch
- **Swarm module** — flocking, formation control, path planning, task allocation, and metrics
- **RL module** — multi-agent reinforcement learning, curriculum learning, reward shaping, replay buffer
- **Vision module** — sonar processing, depth estimation, odometry, marine detectors, preprocessing
- **Navigation module** — collision avoidance, geospatial positioning, path following, waypoint management, pilot, situational awareness
- **Adaptive Autonomy** — trust-gated autonomy levels, learning, override control, transition management
- **Security module** — authentication, safety monitoring, byzantine fault tolerance, attack detection, trust boundaries
- **Compliance module** — IEC 61508, EU AI Act, COLREGS, regulatory audit trails
- **Self-Healing** — fault detection, diagnosis, recovery, adaptation, resilience
- **Digital Twin** — physics simulation, state mirroring, fault simulation, scenario modeling
- **Knowledge Graph** — graph reasoning, NLP, embedding, query engine, marine knowledge base
- **MPC module** — trajectory planning, constraint formulation, multi-objective optimization, adaptive control
- **Sensor Fusion** — Kalman filtering, Bayesian fusion, particle filtering, associative fusion, calibration
- **Explainability** — decision logging, attribution, counterfactual analysis, interpret reports
- **Fleet Coordination** — CRDT-based state sync, consensus, communication, task orchestration, resource management
- **Energy Management** — battery modeling, power budgets, load shedding, energy harvesting, forecasting
- **Cooperative Perception** — shared perception fusion, quality assessment, consensus perception
- **Agent system** — intent parsing, bytecode generation, fleet simulation, MQTT bridge, edge heartbeat, LLM pipeline, Rosetta Stone intent compiler, emergency protocols
- **Data Pipeline** — streaming, transformation, aggregation, storage
- **API Gateway** — REST routing, authentication, rate limiting, OpenAPI spec, middleware
- **Comms Hub** — protocol handling, encoding, routing, compression, QoS
- **Decision Engine** — multi-objective decisions, uncertainty quantification, utility theory, group decisions
- **Runtime Verification** — temporal logic, invariant checking, contracts, watchdog monitoring
- **Maintenance** — health scoring, failure prediction, degradation modeling, remaining life estimation, scheduling
- **Performance** — benchmarking, profiling, resource monitoring, caching, optimization
- **A/B Testing** with statistical engine and experiment management

### Fixed

- Various review feedback items from expert code review
- Improved CI/CD pipeline with parallel jobs for Python tests, firmware tests, and linting
- Added community governance files (CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md)
- Added issue and PR templates for structured contribution workflows
