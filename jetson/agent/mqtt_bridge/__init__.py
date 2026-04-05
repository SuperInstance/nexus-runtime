"""NEXUS MQTT Bridge — Real-time telemetry between edge vessels and cloud services.

Provides:
  - 13+ MQTT topics in the Edge-Native topic hierarchy
  - Abstract MQTT client interface with mock implementation
  - Telemetry bridge: UnifiedObservation <-> JSON <-> MQTT
  - Message router: inbound dispatch with dead letter queue
  - Connection manager: multi-broker, health monitoring, graceful shutdown
"""
