"""Tests for particle.py — Particle, ParticleFilter."""

import math
import pytest
import random

from jetson.sensor_fusion.particle import Particle, ParticleFilter


# ===================================================================
# Particle dataclass tests
# ===================================================================

class TestParticle:
    def test_creation_defaults(self):
        p = Particle(state=[1.0, 2.0])
        assert p.state == [1.0, 2.0]
        assert p.weight == 1.0

    def test_creation_custom_weight(self):
        p = Particle(state=[0.0], weight=0.5)
        assert p.weight == 0.5

    def test_mutable(self):
        p = Particle(state=[1.0, 2.0])
        p.state[0] = 99.0
        assert p.state[0] == 99.0


# ===================================================================
# ParticleFilter tests
# ===================================================================

class TestParticleFilterInit:
    def test_default_init(self):
        pf = ParticleFilter(num_particles=100, state_dim=2)
        assert len(pf.particles) == 100
        for p in pf.particles:
            assert len(p.state) == 2
            assert p.weight == pytest.approx(0.01, abs=1e-12)

    def test_initial_weights_sum_to_one(self):
        pf = ParticleFilter(num_particles=50, state_dim=3)
        total = sum(p.weight for p in pf.particles)
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_state_dim(self):
        pf = ParticleFilter(num_particles=10, state_dim=5)
        for p in pf.particles:
            assert len(p.state) == 5


class TestParticleFilterInitialize:
    def test_initialize_particles(self):
        pf = ParticleFilter(num_particles=4, state_dim=2)
        pf.initialize_particles([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        assert pf.particles[0].state == [1.0, 2.0]
        assert pf.particles[3].state == [7.0, 8.0]

    def test_initialize_wrong_count_raises(self):
        pf = ParticleFilter(num_particles=3, state_dim=2)
        with pytest.raises(ValueError):
            pf.initialize_particles([[0.0, 0.0]])

    def test_initialize_wrong_dim_raises(self):
        pf = ParticleFilter(num_particles=2, state_dim=2)
        with pytest.raises(ValueError):
            pf.initialize_particles([[1.0], [2.0]])


class TestParticleFilterPredict:
    def test_predict_no_process_fn(self):
        pf = ParticleFilter(num_particles=100, state_dim=1)
        pf.initialize_particles([[0.0]] * 100)
        pf.set_process_noise([0.0])  # zero noise
        pf.predict(dt=1.0)
        # All particles should still be at 0 (no process fn, zero noise)
        for p in pf.particles:
            assert abs(p.state[0]) < 1e-12

    def test_predict_with_process_fn(self):
        def drift(state, dt):
            return [s + dt for s in state]

        pf = ParticleFilter(num_particles=10, state_dim=2)
        pf.set_process_noise([0.0, 0.0])
        pf.predict(dt=1.0, process_fn=drift)
        for p in pf.particles:
            assert p.state[0] == pytest.approx(1.0, abs=1e-10)
            assert p.state[1] == pytest.approx(1.0, abs=1e-10)

    def test_predict_adds_noise(self):
        random.seed(42)
        pf = ParticleFilter(num_particles=1000, state_dim=1)
        pf.initialize_particles([[0.0]] * 1000)
        pf.set_process_noise([1.0])
        pf.predict(dt=1.0, process_fn=lambda s, dt: s)
        states = [p.state[0] for p in pf.particles]
        mean = sum(states) / len(states)
        var = sum((s - mean) ** 2 for s in states) / len(states)
        assert abs(mean) < 0.2  # centered at 0
        assert var > 0.5  # has spread

    def test_predict_dt_zero(self):
        pf = ParticleFilter(num_particles=10, state_dim=1)
        pf.initialize_particles([[5.0]] * 10)
        pf.set_process_noise([1.0])
        pf.predict(dt=0.0)
        for p in pf.particles:
            assert abs(p.state[0] - 5.0) < 1e-10

    def test_predict_dt_negative(self):
        pf = ParticleFilter(num_particles=10, state_dim=1)
        pf.initialize_particles([[0.0]] * 10)
        pf.set_process_noise([0.0])
        pf.predict(dt=-1.0)
        # sqrt(max(dt, 0)) = 0 for negative dt, so no noise
        for p in pf.particles:
            assert abs(p.state[0]) < 1e-10


class TestParticleFilterUpdate:
    def test_update_uniform_likelihood(self):
        pf = ParticleFilter(num_particles=10, state_dim=1)
        pf.initialize_particles([[float(i)] for i in range(10)])
        pf.update(measurement=[5.0], likelihood_fn=lambda s, z: 1.0)
        # All weights should remain uniform
        for p in pf.particles:
            assert p.weight == pytest.approx(0.1, abs=1e-10)

    def test_update_gaussian_likelihood(self):
        def gauss_likelihood(s, z):
            return math.exp(-0.5 * ((s[0] - z[0]) ** 2) / 1.0)

        pf = ParticleFilter(num_particles=100, state_dim=1)
        pf.initialize_particles([[float(i)] for i in range(100)])
        pf.update(measurement=[50.0], likelihood_fn=gauss_likelihood)
        # Find the particle with max weight
        max_weight_p = max(pf.particles, key=lambda p: p.weight)
        # Should be the one closest to 50
        assert abs(max_weight_p.state[0] - 50.0) < 1.0

    def test_update_weights_sum_to_one(self):
        pf = ParticleFilter(num_particles=10, state_dim=1)
        pf.initialize_particles([[float(i)] for i in range(10)])
        pf.update(measurement=[5.0], likelihood_fn=lambda s, z: max(0.01, 10.0 - abs(s[0] - z[0])))
        total = sum(p.weight for p in pf.particles)
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_update_zero_likelihood_degeneracy_guard(self):
        pf = ParticleFilter(num_particles=5, state_dim=1)
        pf.update(measurement=[0.0], likelihood_fn=lambda s, z: 0.0)
        # Should redistribute uniformly
        for p in pf.particles:
            assert p.weight == pytest.approx(0.2, abs=1e-10)


class TestParticleFilterResample:
    def test_systematic_resample(self):
        random.seed(99)
        pf = ParticleFilter(num_particles=100, state_dim=1)
        pf.initialize_particles([[float(i)] for i in range(100)])
        # Make particle 50 have all the weight
        for p in pf.particles:
            p.weight = 1e-10
        pf.particles[50].weight = 1.0
        pf.normalize_weights()
        pf.resample(method='systematic')
        # After resample, most particles should be at state 50
        unique_states = set(p.state[0] for p in pf.particles)
        assert 50.0 in unique_states
        # Count how many are at 50
        count_50 = sum(1 for p in pf.particles if abs(p.state[0] - 50.0) < 1e-10)
        assert count_50 >= 50  # at least half should be at 50

    def test_multinomial_resample(self):
        random.seed(42)
        pf = ParticleFilter(num_particles=100, state_dim=1)
        pf.initialize_particles([[float(i)] for i in range(100)])
        for p in pf.particles:
            p.weight = 1e-10
        pf.particles[50].weight = 1.0
        pf.normalize_weights()
        pf.resample(method='multinomial')
        count_50 = sum(1 for p in pf.particles if abs(p.state[0] - 50.0) < 1e-10)
        assert count_50 >= 30

    def test_stratified_resample(self):
        random.seed(7)
        pf = ParticleFilter(num_particles=100, state_dim=1)
        pf.initialize_particles([[float(i)] for i in range(100)])
        for p in pf.particles:
            p.weight = 1e-10
        pf.particles[50].weight = 1.0
        pf.normalize_weights()
        pf.resample(method='stratified')
        count_50 = sum(1 for p in pf.particles if abs(p.state[0] - 50.0) < 1e-10)
        assert count_50 >= 30

    def test_resample_unknown_method_raises(self):
        pf = ParticleFilter(num_particles=10, state_dim=1)
        with pytest.raises(ValueError):
            pf.resample(method='unknown')

    def test_resample_weights_uniform_after(self):
        random.seed(0)
        pf = ParticleFilter(num_particles=50, state_dim=1)
        pf.update(measurement=[0.0], likelihood_fn=lambda s, z: max(0.01, 1.0 - abs(s[0])))
        pf.resample(method='systematic')
        for p in pf.particles:
            assert p.weight == pytest.approx(1.0 / 50, abs=1e-12)


class TestParticleFilterGetStateEstimate:
    def test_mean_uniform(self):
        pf = ParticleFilter(num_particles=4, state_dim=1)
        pf.initialize_particles([[2.0], [4.0], [6.0], [8.0]])
        est = pf.get_state_estimate()
        assert est["mean"][0] == pytest.approx(5.0, abs=1e-10)

    def test_mean_weighted(self):
        pf = ParticleFilter(num_particles=2, state_dim=1)
        pf.particles[0].state = [0.0]
        pf.particles[0].weight = 0.75
        pf.particles[1].state = [10.0]
        pf.particles[1].weight = 0.25
        est = pf.get_state_estimate()
        assert est["mean"][0] == pytest.approx(2.5, abs=1e-10)

    def test_covariance_shape(self):
        pf = ParticleFilter(num_particles=10, state_dim=3)
        est = pf.get_state_estimate()
        cov = est["covariance"]
        assert len(cov) == 3
        for row in cov:
            assert len(row) == 3

    def test_covariance_zero_for_identical(self):
        pf = ParticleFilter(num_particles=10, state_dim=1)
        pf.initialize_particles([[5.0]] * 10)
        est = pf.get_state_estimate()
        assert est["covariance"][0][0] == pytest.approx(0.0, abs=1e-10)


class TestParticleFilterEffectiveParticles:
    def test_uniform_particles(self):
        pf = ParticleFilter(num_particles=100, state_dim=1)
        assert pf.get_effective_particles() == pytest.approx(100.0, abs=1e-6)

    def test_degenerate_particles(self):
        pf = ParticleFilter(num_particles=100, state_dim=1)
        for p in pf.particles:
            p.weight = 1e-10
        pf.particles[0].weight = 1.0
        pf.normalize_weights()
        n_eff = pf.get_effective_particles()
        assert n_eff < 10.0

    def test_effective_particles_range(self):
        pf = ParticleFilter(num_particles=50, state_dim=1)
        n_eff = pf.get_effective_particles()
        assert 1.0 <= n_eff <= 50.0


class TestParticleFilterSetProcessNoise:
    def test_set_process_noise(self):
        pf = ParticleFilter(num_particles=10, state_dim=3)
        pf.set_process_noise([0.1, 0.2, 0.3])
        assert pf.process_noise_stddev == [0.1, 0.2, 0.3]

    def test_set_process_noise_wrong_dim_raises(self):
        pf = ParticleFilter(num_particles=10, state_dim=2)
        with pytest.raises(ValueError):
            pf.set_process_noise([0.1])


class TestParticleFilterNormalizeWeights:
    def test_normalize(self):
        pf = ParticleFilter(num_particles=3, state_dim=1)
        pf.particles[0].weight = 2.0
        pf.particles[1].weight = 3.0
        pf.particles[2].weight = 5.0
        pf.normalize_weights()
        assert pf.particles[0].weight == pytest.approx(0.2, abs=1e-10)
        assert pf.particles[1].weight == pytest.approx(0.3, abs=1e-10)
        assert pf.particles[2].weight == pytest.approx(0.5, abs=1e-10)

    def test_normalize_zero_weights(self):
        pf = ParticleFilter(num_particles=3, state_dim=1)
        for p in pf.particles:
            p.weight = 0.0
        pf.normalize_weights()
        for p in pf.particles:
            assert p.weight == pytest.approx(1.0 / 3, abs=1e-10)


class TestParticleFilterPredictUpdateResample:
    def test_full_cycle(self):
        random.seed(123)
        pf = ParticleFilter(num_particles=200, state_dim=1)
        pf.initialize_particles([[random.gauss(0, 5)] for _ in range(200)])
        pf.set_process_noise([0.5])

        true_val = 10.0
        for i in range(10):
            pf.predict(dt=0.1, process_fn=lambda s, dt: s)
            pf.update(
                measurement=[true_val + random.gauss(0, 0.5)],
                likelihood_fn=lambda s, z: math.exp(-0.5 * ((s[0] - z[0]) ** 2) / 2.0),
            )
            pf.resample(method='systematic')

        est = pf.get_state_estimate()
        assert abs(est["mean"][0] - true_val) < 5.0
