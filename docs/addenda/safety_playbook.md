> **MIGRATION STATUS**: Copied from Edge-Native reference repo
> **LAST RECONCILED**: 2026-04-05
> **IMPLEMENTATION STATUS**: safety_sm.c and watchdog.c implemented in firmware/src/safety/; Python safety_validator.py in jetson/reflex/

# Safety Validation Playbook

## Pre-Deployment Checklist

### Hardware Safety
- [ ] GPIO pin mapping verified against schematic
- [ ] PWM frequency within motor driver specifications
- [ ] Current limits configured in hardware protection circuit
- [ ] Emergency stop mechanism tested

### Software Safety
- [ ] All JSON schemas validated
- [ ] Configuration sanity checks pass
- [ ] Watchdog timer configured
- [ ] Graceful shutdown handler registered

### Integration Safety
- [ ] Control loop latency measured and within bounds
- [ ] Thermal monitoring active
- [ ] Failsafe behaviors tested (loss of sensor, loss of comms)
- [ ] Logging captures all safety-critical events

## Runtime Safety Monitors
1. **Thermal Monitor**: Shuts down actuators if junction temp > 85°C
2. **Current Monitor**: Triggers overcurrent protection
3. **Watchdog**: Hardware watchdog resets system if software hangs
4. **Heartbeat**: External system can detect unresponsive runtime

## Incident Response
1. Capture logs from `/var/log/nexus/`
2. Check thermal and current telemetry
3. Review configuration diffs from last known good state
4. Run host test suite to reproduce software issues
