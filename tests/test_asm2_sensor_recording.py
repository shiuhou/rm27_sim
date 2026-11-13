"""Sensor-generator boundary tests on analytical inputs, no native replay."""
from copy import deepcopy
import json

import numpy as np
import pytest

from rm27_sim.experiments.asm2_sensor_recording import SensorRecorder, packet
from rm27_sim.experiments.asm2_contact import ROOT


def config():
    specification = json.loads((ROOT/'configs/asm2_recovery_validation.json').read_text())
    result = deepcopy(specification['sensors'])
    result['imu'] = json.loads((ROOT/specification['imu_config']).read_text())
    result['gyro_bias_std_rad_s'] = 0.
    return result


def test_packets_keep_measurement_and_delivery_separate_without_truth_state():
    r = SensorRecorder(config(), 918, start_s=0., native_dt_s=.0001, held_thrust_N=.142*9.81, mass_kg=.142)
    for tick in range(401):
        r.observe(tick, np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3),
                  **({} if tick == 0 else dict(before_velocity=np.zeros(3), before_rotation=np.eye(3))))
    external = [p for p in r.packets if p['kind'] == 'external']
    assert [p['acquisition_s'] for p in external] == [0., .02, .04]
    assert all(p['delivery_s'] == p['acquisition_s']+.015 for p in external)
    assert all(set(p['value']) == {'position_m', 'velocity_m_s'} for p in external)
    assert all(p['source'] == 'TRUTH_GENERATED_SYNTHETIC' for p in r.packets)
    assert not any('rotation' in p or 'truth' in p or 'contact' in p for p in r.packets)
    assert not any(p['kind'] == 'attitude' for p in r.packets)
    assert all(p['acquisition_s'] == p['delivery_s'] for p in r.packets if p['kind'] in ('accel', 'gyro'))
    gyro = [p for p in r.packets if p['kind'] == 'gyro']
    assert len(gyro) > len(external)
    assert all(len(p['value']) == 3 for p in gyro)
    assert all(set(p) == {'kind', 'acquisition_s', 'delivery_s', 'value', 'source', 'valid', 'clipped'} for p in r.packets)


def test_impact_interval_is_sensor_conservative_not_native_contact_label():
    r = SensorRecorder(config(), 919, start_s=0., native_dt_s=.0001, held_thrust_N=.142*9.81, mass_kg=.142)
    previous = np.zeros(3)
    for tick in range(401):
        velocity = np.array([0., 0., tick*.1])
        r.observe(tick, np.zeros(3), velocity, np.eye(3), np.zeros(3),
                  **({} if tick == 0 else dict(before_velocity=previous, before_rotation=np.eye(3))))
        previous = velocity
    events = [p for p in r.packets if p['kind'] == 'impact']
    assert len(events) == 1
    event = events[0]
    assert event['value'] == {'lower_s': 0., 'upper_s': event['acquisition_s']}
    assert event['delivery_s'] == event['acquisition_s']+.0025
    assert any(p['clipped'] for p in r.packets if p['kind'] == 'accel')


def test_native_ticks_cannot_skip_and_derivative_cannot_use_missing_state():
    r = SensorRecorder(config(), 920, start_s=0., native_dt_s=.0001, held_thrust_N=1.4, mass_kg=.142)
    args = (np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match='consecutive'):
        r.observe(1, *args)
    r.observe(0, *args)
    with pytest.raises(ValueError, match='preceding'):
        r.observe(1, *args)


def test_seed_reproduces_packets_without_initial_attitude_observation():
    outputs = []
    for _ in range(2):
        r = SensorRecorder(config(), 921, start_s=0., native_dt_s=.0001, held_thrust_N=1.4, mass_kg=.142)
        r.observe(0, np.zeros(3), np.zeros(3), np.eye(3), np.ones(3))
        outputs.append(r.packets)
    assert outputs[0] == outputs[1]
    assert {p['kind'] for p in outputs[0]} == {'accel', 'gyro', 'external'}
