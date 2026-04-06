"""6-DOF vessel dynamics simulation for NEXUS digital twin engine."""

import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VesselState:
    """Complete 6-DOF vessel state."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0

    def copy(self) -> 'VesselState':
        return VesselState(
            x=self.x, y=self.y, z=self.z,
            vx=self.vx, vy=self.vy, vz=self.vz,
            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
            wx=self.wx, wy=self.wy, wz=self.wz,
        )

    def distance_to(self, other: 'VesselState') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def speed(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    def angular_speed(self) -> float:
        return math.sqrt(self.wx**2 + self.wy**2 + self.wz**2)


@dataclass
class Force:
    """Force and torque vector in 3D."""
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    torque_x: float = 0.0
    torque_y: float = 0.0
    torque_z: float = 0.0

    def magnitude(self) -> float:
        return math.sqrt(self.fx**2 + self.fy**2 + self.fz**2)

    def torque_magnitude(self) -> float:
        return math.sqrt(self.torque_x**2 + self.torque_y**2 + self.torque_z**2)

    def __add__(self, other: 'Force') -> 'Force':
        return Force(
            fx=self.fx + other.fx,
            fy=self.fy + other.fy,
            fz=self.fz + other.fz,
            torque_x=self.torque_x + other.torque_x,
            torque_y=self.torque_y + other.torque_y,
            torque_z=self.torque_z + other.torque_z,
        )

    def __mul__(self, scalar: float) -> 'Force':
        return Force(
            fx=self.fx * scalar,
            fy=self.fy * scalar,
            fz=self.fz * scalar,
            torque_x=self.torque_x * scalar,
            torque_y=self.torque_y * scalar,
            torque_z=self.torque_z * scalar,
        )

    def zero(self) -> bool:
        return (self.fx == 0.0 and self.fy == 0.0 and self.fz == 0.0
                and self.torque_x == 0.0 and self.torque_y == 0.0 and self.torque_z == 0.0)


@dataclass
class DragCoefficients:
    """Hydrodynamic drag coefficients."""
    linear_x: float = 50.0
    linear_y: float = 100.0
    linear_z: float = 200.0
    angular_x: float = 10.0
    angular_y: float = 20.0
    angular_z: float = 15.0
    quadratic_x: float = 20.0
    quadratic_y: float = 40.0
    quadratic_z: float = 80.0


@dataclass
class VesselProperties:
    """Physical properties of the vessel."""
    mass: float = 100.0
    inertia_x: float = 50.0
    inertia_y: float = 100.0
    inertia_z: float = 80.0


class VesselPhysics:
    """6-DOF rigid body physics simulation for marine vessels."""

    DEFAULT_DRAG = DragCoefficients()
    DEFAULT_PROPS = VesselProperties()

    def __init__(self, properties: VesselProperties = None, drag: DragCoefficients = None):
        self.props = properties or self.DEFAULT_PROPS
        self.drag = drag or self.DEFAULT_DRAG

    def apply_force(self, state: VesselState, force: Force, dt: float) -> VesselState:
        """Apply a single force to the vessel and return new state."""
        forces = [force]
        return self.update_state(state, forces, dt)

    def compute_drag(self, velocity: Tuple[float, float, float],
                     angular_vel: Tuple[float, float, float],
                     drag_coefficients: DragCoefficients = None) -> Force:
        """Compute hydrodynamic drag force from velocity."""
        dc = drag_coefficients or self.drag
        vx, vy, vz = velocity
        wx, wy, wz = angular_vel

        # Linear drag: F = -c * v
        drag_fx = -(dc.linear_x * vx + dc.quadratic_x * vx * abs(vx))
        drag_fy = -(dc.linear_y * vy + dc.quadratic_y * vy * abs(vy))
        drag_fz = -(dc.linear_z * vz + dc.quadratic_z * vz * abs(vz))

        # Angular drag
        torque_x = -dc.angular_x * wx
        torque_y = -dc.angular_y * wy
        torque_z = -dc.angular_z * wz

        return Force(fx=drag_fx, fy=drag_fy, fz=drag_fz,
                    torque_x=torque_x, torque_y=torque_y, torque_z=torque_z)

    def compute_thrust(self, throttle: float, heading: float,
                       max_thrust: float = 500.0) -> Force:
        """Compute thrust force from throttle and heading."""
        magnitude = throttle * max_thrust
        fx = magnitude * math.cos(heading)
        fy = magnitude * math.sin(heading)
        return Force(fx=fx, fy=fy, fz=0.0)

    def compute_thrust_with_depth(self, throttle: float, heading: float,
                                   pitch_angle: float, max_thrust: float = 500.0) -> Force:
        """Compute 3D thrust force from throttle, heading, and pitch."""
        magnitude = throttle * max_thrust
        horizontal = magnitude * math.cos(pitch_angle)
        fx = horizontal * math.cos(heading)
        fy = horizontal * math.sin(heading)
        fz = -magnitude * math.sin(pitch_angle)
        return Force(fx=fx, fy=fy, fz=fz)

    def compute_buoyancy(self, submersion: float, displacement: float = 1000.0) -> Force:
        """Compute buoyancy force. submersion is fraction 0..1+."""
        g = 9.81
        buoyancy_force = submersion * displacement * g
        gravity = self.props.mass * g
        net_fz = buoyancy_force - gravity
        return Force(fx=0.0, fy=0.0, fz=net_fz)

    def compute_gravity(self) -> Force:
        """Compute gravitational force."""
        g = 9.81
        return Force(fx=0.0, fy=0.0, fz=-self.props.mass * g)

    def compute_derivatives(self, state: VesselState, forces: List[Force]) -> dict:
        """Compute state derivatives (rates of change) for given forces."""
        total = Force()
        for f in forces:
            total = total + f

        dx = state.vx
        dy = state.vy
        dz = state.vz
        dvx = total.fx / self.props.mass
        dvy = total.fy / self.props.mass
        dvz = total.fz / self.props.mass
        droll = state.wx
        dpitch = state.wy
        dyaw = state.wz
        dwx = total.torque_x / self.props.inertia_x
        dwy = total.torque_y / self.props.inertia_y
        dwz = total.torque_z / self.props.inertia_z

        return {
            'dx': dx, 'dy': dy, 'dz': dz,
            'dvx': dvx, 'dvy': dvy, 'dvz': dvz,
            'droll': droll, 'dpitch': dpitch, 'dyaw': dyaw,
            'dwx': dwx, 'dwy': dwy, 'dwz': dwz,
        }

    def update_state(self, state: VesselState, forces: List[Force], dt: float) -> VesselState:
        """Euler integration: update state given forces and time step."""
        derivs = self.compute_derivatives(state, forces)
        return VesselState(
            x=state.x + derivs['dx'] * dt,
            y=state.y + derivs['dy'] * dt,
            z=state.z + derivs['dz'] * dt,
            vx=state.vx + derivs['dvx'] * dt,
            vy=state.vy + derivs['dvy'] * dt,
            vz=state.vz + derivs['dvz'] * dt,
            roll=state.roll + derivs['droll'] * dt,
            pitch=state.pitch + derivs['dpitch'] * dt,
            yaw=state.yaw + derivs['dyaw'] * dt,
            wx=state.wx + derivs['dwx'] * dt,
            wy=state.wy + derivs['dwy'] * dt,
            wz=state.wz + derivs['dwz'] * dt,
        )

    def _derivs_from_state(self, state: VesselState) -> dict:
        """Extract derivatives packed into a VesselState (used by RK4)."""
        return {
            'dx': state.vx, 'dy': state.vy, 'dz': state.vz,
            'dvx': state.roll, 'dvy': state.pitch, 'dvz': state.yaw,
            'droll': state.wx, 'dpitch': state.wy, 'dyaw': state.wz,
            'dwx': 0.0, 'dwy': 0.0, 'dwz': 0.0,  # torques stored separately
        }

    def rk4_step(self, state: VesselState, forces: List[Force], dt: float) -> VesselState:
        """4th-order Runge-Kutta integration step."""
        # k1
        k1 = self.compute_derivatives(state, forces)

        # Build intermediate state for k2
        s2 = VesselState(
            x=state.x + 0.5*dt*k1['dx'], y=state.y + 0.5*dt*k1['dy'],
            z=state.z + 0.5*dt*k1['dz'],
            vx=state.vx + 0.5*dt*k1['dvx'], vy=state.vy + 0.5*dt*k1['dvy'],
            vz=state.vz + 0.5*dt*k1['dvz'],
            roll=state.roll + 0.5*dt*k1['droll'], pitch=state.pitch + 0.5*dt*k1['dpitch'],
            yaw=state.yaw + 0.5*dt*k1['dyaw'],
            wx=state.wx + 0.5*dt*k1['dwx'], wy=state.wy + 0.5*dt*k1['dwy'],
            wz=state.wz + 0.5*dt*k1['dwz'],
        )
        k2 = self.compute_derivatives(s2, forces)

        # k3
        s3 = VesselState(
            x=state.x + 0.5*dt*k2['dx'], y=state.y + 0.5*dt*k2['dy'],
            z=state.z + 0.5*dt*k2['dz'],
            vx=state.vx + 0.5*dt*k2['dvx'], vy=state.vy + 0.5*dt*k2['dvy'],
            vz=state.vz + 0.5*dt*k2['dvz'],
            roll=state.roll + 0.5*dt*k2['droll'], pitch=state.pitch + 0.5*dt*k2['dpitch'],
            yaw=state.yaw + 0.5*dt*k2['dyaw'],
            wx=state.wx + 0.5*dt*k2['dwx'], wy=state.wy + 0.5*dt*k2['dwy'],
            wz=state.wz + 0.5*dt*k2['dwz'],
        )
        k3 = self.compute_derivatives(s3, forces)

        # k4
        s4 = VesselState(
            x=state.x + dt*k3['dx'], y=state.y + dt*k3['dy'],
            z=state.z + dt*k3['dz'],
            vx=state.vx + dt*k3['dvx'], vy=state.vy + dt*k3['dvy'],
            vz=state.vz + dt*k3['dvz'],
            roll=state.roll + dt*k3['droll'], pitch=state.pitch + dt*k3['dpitch'],
            yaw=state.yaw + dt*k3['dyaw'],
            wx=state.wx + dt*k3['dwx'], wy=state.wy + dt*k3['dwy'],
            wz=state.wz + dt*k3['dwz'],
        )
        k4 = self.compute_derivatives(s4, forces)

        # Combine
        return VesselState(
            x=state.x + (dt/6.0)*(k1['dx'] + 2*k2['dx'] + 2*k3['dx'] + k4['dx']),
            y=state.y + (dt/6.0)*(k1['dy'] + 2*k2['dy'] + 2*k3['dy'] + k4['dy']),
            z=state.z + (dt/6.0)*(k1['dz'] + 2*k2['dz'] + 2*k3['dz'] + k4['dz']),
            vx=state.vx + (dt/6.0)*(k1['dvx'] + 2*k2['dvx'] + 2*k3['dvx'] + k4['dvx']),
            vy=state.vy + (dt/6.0)*(k1['dvy'] + 2*k2['dvy'] + 2*k3['dvy'] + k4['dvy']),
            vz=state.vz + (dt/6.0)*(k1['dvz'] + 2*k2['dvz'] + 2*k3['dvz'] + k4['dvz']),
            roll=state.roll + (dt/6.0)*(k1['droll'] + 2*k2['droll'] + 2*k3['droll'] + k4['droll']),
            pitch=state.pitch + (dt/6.0)*(k1['dpitch'] + 2*k2['dpitch'] + 2*k3['dpitch'] + k4['dpitch']),
            yaw=state.yaw + (dt/6.0)*(k1['dyaw'] + 2*k2['dyaw'] + 2*k3['dyaw'] + k4['dyaw']),
            wx=state.wx + (dt/6.0)*(k1['dwx'] + 2*k2['dwx'] + 2*k3['dwx'] + k4['dwx']),
            wy=state.wy + (dt/6.0)*(k1['dwy'] + 2*k2['dwy'] + 2*k3['dwy'] + k4['dwy']),
            wz=state.wz + (dt/6.0)*(k1['dwz'] + 2*k2['dwz'] + 2*k3['dwz'] + k4['dwz']),
        )

    def simulate(self, state: VesselState, forces: List[Force],
                 dt: float, steps: int, method: str = 'euler') -> List[VesselState]:
        """Run simulation for multiple steps. method='euler' or 'rk4'."""
        trajectory = [state.copy()]
        current = state.copy()
        step_fn = self.rk4_step if method == 'rk4' else self.update_state
        for _ in range(steps):
            current = step_fn(current, forces, dt)
            trajectory.append(current.copy())
        return trajectory

    def kinetic_energy(self, state: VesselState) -> float:
        """Compute translational kinetic energy. KE = 0.5 * m * v^2"""
        return 0.5 * self.props.mass * (state.vx**2 + state.vy**2 + state.vz**2)

    def rotational_energy(self, state: VesselState) -> float:
        """Compute rotational kinetic energy."""
        return (0.5 * self.props.inertia_x * state.wx**2 +
                0.5 * self.props.inertia_y * state.wy**2 +
                0.5 * self.props.inertia_z * state.wz**2)

    def momentum(self, state: VesselState) -> Tuple[float, float, float]:
        """Compute linear momentum vector."""
        return (self.props.mass * state.vx,
                self.props.mass * state.vy,
                self.props.mass * state.vz)

    def total_force(self, forces: List[Force]) -> Force:
        """Sum all forces into a single resultant."""
        total = Force()
        for f in forces:
            total = total + f
        return total
