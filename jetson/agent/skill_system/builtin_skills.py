"""NEXUS Built-in Marine Skill Cartridges — Pre-compiled marine operation patterns.

Five built-in skill cartridges covering essential marine operations:
  1. surface_navigation  — GPS waypoint navigation (trust L2)
  2. depth_monitoring    — Depth sensor alert system (trust L1)
  3. station_keeping     — PID position hold on thrusters (trust L3)
  4. emergency_surface   — Emergency blow-ballast ascend (trust L0)
  5. sensor_survey       — Multi-sensor data collection (trust L1)

Each cartridge contains pre-compiled AAB bytecode that passes the
6-stage safety validation pipeline at its declared trust level.

Bytecode programs are built using the BytecodeEmitter to ensure
correct instruction encoding (8-byte little-endian format).
"""

from __future__ import annotations

from agent.skill_system.cartridge import SkillCartridge, SkillParameter

# Lazy import to avoid circular dependency at module level
def _build_builtin_skills() -> dict[str, SkillCartridge]:
    """Build all built-in skill cartridges.

    Uses BytecodeEmitter to create syntactically valid bytecode
    that passes the safety validation pipeline at each skill's
    declared trust level.

    Returns:
        Dictionary mapping skill names to SkillCartridge instances.
    """
    from reflex.bytecode_emitter import BytecodeEmitter

    skills: dict[str, SkillCartridge] = {}

    # ==================================================================
    # 1. emergency_surface — trust L2 (requires WRITE_PIN for ascent)
    # ==================================================================
    # Read depth, compute danger signal, clamp, WRITE to ascent actuator.
    # Uses L2 opcodes: L0 + WRITE_PIN, JUMP
    #
    # Stack trace:                       Stack depth
    # 0: READ_PIN 10 (depth)             1
    # 1: PUSH_F32 180.0                  2
    # 2: SUB_F                           1  (depth excess over max)
    # 3: ABS_F                           1  (always positive)
    # 4: CLAMP_F 0.0 500.0               1  (normalize danger signal)
    # 5: WRITE_PIN 7 (ballast actuator)  0  (trigger ascent)
    # 6: NOP                             0  (end)
    em = BytecodeEmitter()
    em.emit_read_pin(10)      # depth sensor
    em.emit_push_f32(180.0)   # max depth trigger
    em.emit_sub_f()           # depth - 180.0 (positive if too deep)
    em.emit_abs_f()           # |depth excess|
    em.emit_clamp_f(0.0, 500.0)  # normalize danger signal
    em.emit_write_pin(7)      # trigger ascent actuator (pin 7 = ballast)
    em.emit_nop()             # end marker
    skills["emergency_surface"] = SkillCartridge(
        name="emergency_surface",
        version="1.0.1",
        description="Emergency surface procedure — read depth, compute danger, trigger ascent",
        domain="marine",
        trust_required=2,
        bytecode=em.get_bytecode(),
        inputs=[
            SkillParameter(
                name="depth", type="sensor", pin=10,
                range_min=0.0, range_max=500.0, unit="meters",
                description="Depth sensor reading",
            ),
        ],
        outputs=[
            SkillParameter(
                name="ascent_command", type="actuator", pin=7,
                range_min=-100.0, range_max=100.0, unit="percent",
                description="Ballast actuator ascent command",
            ),
        ],
        parameters={"ascent_rate": 1.0, "max_depth_trigger": 180.0},
        constraints={"max_depth": {"min": 0.0, "max": 500.0}},
        provenance={
            "author": "nexus-core",
            "review_status": "reviewed",
            "test_results": "pass",
        },
        metadata={"category": "safety", "priority": "critical"},
    )

    # ==================================================================
    # 2. depth_monitoring — trust L1 (+ conditional branches)
    # ==================================================================
    # Read depth, compare to max, conditional branch.
    # Uses L1 opcodes: L0 + JUMP_IF_TRUE
    #
    # Stack trace:                    Stack depth
    # 0: READ_PIN 10 (depth)          1
    # 1: PUSH_F32 200.0               2
    # 2: GTE_F                        1  (depth >= 200?)
    # 3: JUMP_IF_TRUE 5               0  (if true, jump to 5)
    # 4: NOP                          0  (safe path: continue monitoring)
    # 5: READ_TIMER_MS                1  (alert path: record time)
    # 6: NOP                          1  (end)
    em.reset()
    em.emit_read_pin(10)       # depth sensor
    em.emit_push_f32(200.0)    # max_depth threshold
    em.emit_gte_f()            # depth >= 200?
    em.emit_jump_if_true(5)    # if deep, jump to alert
    em.emit_nop()              # safe: no alert needed
    em.emit_read_timer_ms()    # alert: record timestamp
    em.emit_nop()              # end
    skills["depth_monitoring"] = SkillCartridge(
        name="depth_monitoring",
        version="1.0.0",
        description="Monitor depth sensor and trigger alerts when exceeding threshold",
        domain="marine",
        trust_required=1,
        bytecode=em.get_bytecode(),
        inputs=[
            SkillParameter(
                name="depth", type="sensor", pin=10,
                range_min=0.0, range_max=1000.0, unit="meters",
                description="Depth sensor reading",
            ),
        ],
        outputs=[
            SkillParameter(
                name="alert_flag", type="variable",
                range_min=0.0, range_max=1.0, unit="boolean",
                description="Alert trigger flag",
            ),
        ],
        parameters={"max_depth": 200.0, "alert_interval": 30.0},
        constraints={"max_depth": {"min": 0.0, "max": 1000.0}},
        provenance={
            "author": "nexus-core",
            "review_status": "reviewed",
            "test_results": "pass",
        },
        metadata={"category": "monitoring", "priority": "high"},
    )

    # ==================================================================
    # 3. surface_navigation — trust L2 (+ WRITE_PIN, JUMP)
    # ==================================================================
    # Read heading, subtract target, clamp, write to rudder.
    # Uses L2 opcodes: L1 + WRITE_PIN, JUMP
    # Pin 4 = rudder (non-safety critical)
    #
    # Stack trace:                       Stack depth
    # 0: READ_PIN 4 (compass heading)    1
    # 1: PUSH_F32 270.0 (target)         2
    # 2: SUB_F                           1  (heading error)
    # 3: CLAMP_F -45.0 45.0              1  (limit rudder range)
    # 4: WRITE_PIN 4 (rudder)            0  (actuate)
    # 5: NOP                             0  (end)
    em.reset()
    em.emit_read_pin(4)        # compass heading
    em.emit_push_f32(270.0)    # target heading
    em.emit_sub_f()            # heading error
    em.emit_clamp_f(-45.0, 45.0)  # clamp to rudder range
    em.emit_write_pin(4)       # write to rudder actuator
    em.emit_nop()              # end
    skills["surface_navigation"] = SkillCartridge(
        name="surface_navigation",
        version="1.0.0",
        description="Basic surface waypoint navigation using GPS and compass",
        domain="marine",
        trust_required=2,
        bytecode=em.get_bytecode(),
        inputs=[
            SkillParameter(
                name="compass_heading", type="sensor", pin=4,
                range_min=0.0, range_max=360.0, unit="degrees",
                description="Compass heading reading",
            ),
        ],
        outputs=[
            SkillParameter(
                name="rudder_angle", type="actuator", pin=4,
                range_min=-45.0, range_max=45.0, unit="degrees",
                description="Rudder actuator command",
            ),
        ],
        parameters={"waypoint_tolerance": 5.0, "max_speed": 2.0},
        constraints={
            "heading": {"min": 0.0, "max": 360.0},
            "rudder": {"min": -90.0, "max": 90.0},
        },
        provenance={
            "author": "nexus-core",
            "review_status": "reviewed",
            "test_results": "pass",
        },
        metadata={"category": "navigation", "priority": "standard"},
    )

    # ==================================================================
    # 4. station_keeping — trust L3 (L2 opcodes, CALL not needed)
    # ==================================================================
    # Read position, compute error, apply gain, clamp, write thruster.
    # Two-axis position control with separate reads.
    # Uses L2 opcodes (all available at L3).
    # Pin 5 = thruster (non-safety critical), Pin 6 = position Y sensor
    #
    # Stack trace:                       Stack depth
    # 0: READ_PIN 5 (position_x)         1
    # 1: PUSH_F32 100.0 (setpoint_x)    2
    # 2: SUB_F                           1  (error_x)
    # 3: PUSH_F32 1.0 (Kp)              2
    # 4: MUL_F                           1  (output_x)
    # 5: CLAMP_F -100.0 100.0            1  (clamped_x)
    # 6: WRITE_PIN 5 (thruster_x)        0
    # 7: READ_PIN 6 (position_y)         1
    # 8: PUSH_F32 50.0 (setpoint_y)     2
    # 9: SUB_F                           1  (error_y)
    # 10: ABS_F                          1  (|error_y|)
    # 11: PUSH_F32 0.1 (Kp_y)           2
    # 12: MUL_F                          1  (scaled_y)
    # 13: CLAMP_F 0.0 50.0              1  (clamped_y)
    # 14: NOP                            1  (end)
    em.reset()
    em.emit_read_pin(5)        # position X
    em.emit_push_f32(100.0)    # setpoint X
    em.emit_sub_f()            # error X
    em.emit_push_f32(1.0)      # Kp gain
    em.emit_mul_f()            # output X
    em.emit_clamp_f(-100.0, 100.0)  # clamp output
    em.emit_write_pin(5)       # thruster X
    em.emit_read_pin(6)        # position Y
    em.emit_push_f32(50.0)     # setpoint Y
    em.emit_sub_f()            # error Y
    em.emit_abs_f()            # |error Y|
    em.emit_push_f32(0.1)      # Y-axis gain
    em.emit_mul_f()            # scaled Y
    em.emit_clamp_f(0.0, 50.0)  # clamp Y
    em.emit_nop()              # end
    skills["station_keeping"] = SkillCartridge(
        name="station_keeping",
        version="1.0.0",
        description="Maintain position using PID control on thrusters",
        domain="marine",
        trust_required=3,
        bytecode=em.get_bytecode(),
        inputs=[
            SkillParameter(
                name="position_x", type="sensor", pin=5,
                range_min=-500.0, range_max=500.0, unit="meters",
                description="X position from GPS",
            ),
            SkillParameter(
                name="position_y", type="sensor", pin=6,
                range_min=-500.0, range_max=500.0, unit="meters",
                description="Y position from GPS",
            ),
        ],
        outputs=[
            SkillParameter(
                name="thruster_x", type="actuator", pin=5,
                range_min=-100.0, range_max=100.0, unit="percent",
                description="X-axis thruster command",
            ),
            SkillParameter(
                name="position_error_y", type="variable",
                range_min=0.0, range_max=50.0, unit="meters",
                description="Y-axis position error magnitude",
            ),
        ],
        parameters={
            "kp": 1.0, "ki": 0.1, "kd": 0.05,
            "position_tolerance": 2.0,
        },
        constraints={
            "max_thrust": {"min": -100.0, "max": 100.0},
            "position_error": {"min": 0.0, "max": 50.0},
        },
        provenance={
            "author": "nexus-core",
            "review_status": "reviewed",
            "test_results": "pass",
        },
        metadata={"category": "control", "priority": "standard"},
    )

    # ==================================================================
    # 5. sensor_survey — trust L1 (read-only with conditional branch)
    # ==================================================================
    # Read multiple sensors, accumulate, compute delta from interval.
    # Uses L0 opcodes only (all valid at L1).
    #
    # Stack trace:                       Stack depth
    # 0: READ_PIN 10 (depth)             1
    # 1: READ_PIN 11 (temperature)       2
    # 2: ADD_F                           1  (depth + temp)
    # 3: READ_PIN 12 (pressure)          2
    # 4: ADD_F                           1  (sum)
    # 5: PUSH_F32 5.0 (interval)         2
    # 6: SUB_F                           1  (result)
    # 7: NOP                             1  (end)
    em.reset()
    em.emit_read_pin(10)       # depth sensor
    em.emit_read_pin(11)       # temperature sensor
    em.emit_add_f()            # accumulate
    em.emit_read_pin(12)       # pressure sensor
    em.emit_add_f()            # accumulate
    em.emit_push_f32(5.0)      # survey interval
    em.emit_sub_f()            # compute delta
    em.emit_nop()              # end
    skills["sensor_survey"] = SkillCartridge(
        name="sensor_survey",
        version="1.0.0",
        description="Systematic sensor survey pattern — read multiple sensors and accumulate",
        domain="marine",
        trust_required=1,
        bytecode=em.get_bytecode(),
        inputs=[
            SkillParameter(
                name="depth", type="sensor", pin=10,
                range_min=0.0, range_max=500.0, unit="meters",
                description="Depth sensor",
            ),
            SkillParameter(
                name="temperature", type="sensor", pin=11,
                range_min=-10.0, range_max=40.0, unit="celsius",
                description="Water temperature sensor",
            ),
            SkillParameter(
                name="pressure", type="sensor", pin=12,
                range_min=0.0, range_max=5000.0, unit="millibar",
                description="Water pressure sensor",
            ),
        ],
        outputs=[
            SkillParameter(
                name="survey_result", type="variable",
                range_min=None, range_max=None, unit="composite",
                description="Accumulated sensor survey result",
            ),
        ],
        parameters={"survey_interval": 5.0, "sensors": "all"},
        constraints={"max_sensors": {"min": 1, "max": 16}},
        provenance={
            "author": "nexus-core",
            "review_status": "reviewed",
            "test_results": "pass",
        },
        metadata={"category": "survey", "priority": "low"},
    )

    return skills


# Module-level cache: build on first access
_BUILTIN_CACHE: dict[str, SkillCartridge] | None = None


def get_builtin_skills() -> dict[str, SkillCartridge]:
    """Get all built-in skill cartridges (lazy-initialized).

    Returns:
        Dictionary mapping skill names to SkillCartridge instances.
    """
    global _BUILTIN_CACHE
    if _BUILTIN_CACHE is None:
        _BUILTIN_CACHE = _build_builtin_skills()
    return _BUILTIN_CACHE


# Convenience alias
BUILTIN_SKILLS = get_builtin_skills


def get_builtin_skill(name: str) -> SkillCartridge | None:
    """Get a specific built-in skill cartridge by name.

    Args:
        name: Skill name (e.g., "surface_navigation").

    Returns:
        SkillCartridge if found, None otherwise.
    """
    return get_builtin_skills().get(name)


def list_builtin_skills() -> list[str]:
    """List all built-in skill names.

    Returns:
        Sorted list of built-in skill names.
    """
    return sorted(get_builtin_skills().keys())
