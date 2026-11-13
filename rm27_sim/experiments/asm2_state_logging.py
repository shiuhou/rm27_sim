"""Offline state-log clocks and strict time alignment, independent of physics.

This module records provenance for saved estimates. It does not estimate state,
create covariance, infer missing sensor vectors or change channel validity.
"""
from __future__ import annotations

import numpy as np


def historical_estimator_generation():
    """Reset generation was not recorded; numerical state time is separate.

    Contact events, query timestamps and channel age cannot establish an
    estimator reset generation, so no generation is inferred from them.
    """
    return dict(estimator_epoch_id=None, estimator_epoch_provenance='NOT_RECORDED')


def translation_clock(query_s, acquisition_s, *, latency_s, valid,
                      source='SYNTHETIC_EXTERNAL_TRANSLATION'):
    """Distinguish capture, assumed delivery, and per-field state epochs.

    The saved RecoverySensors contract extrapolates position with constant
    measured velocity to query time, while leaving the velocity sample unchanged.
    A modeled delivery time is never labelled as a recorded transport event.
    """
    if (not np.isfinite([query_s, latency_s]).all() or query_s < 0
            or latency_s < 0 or not isinstance(source, str) or not source):
        raise ValueError('Invalid translation log clock or source')
    if acquisition_s is None:
        if valid:
            raise ValueError('Valid translation requires an acquisition timestamp')
        return dict(**historical_estimator_generation(), source=source, modality='SYNTHETIC_EXTERNAL_POSITION_VELOCITY',
                    query_time_s=float(query_s), acquisition_time_s=None,
                    modeled_delivery_time_s=None, delivery_provenance='UNAVAILABLE',
                    age_s=None, valid=False, position_state_epoch_s=None,
                    velocity_state_epoch_s=None, position_propagated=False,
                    velocity_propagated=False, covariance=None)
    if not np.isfinite(acquisition_s) or acquisition_s < 0 or acquisition_s > query_s + 1e-9:
        raise ValueError('Invalid acquisition timestamp')
    delivery = float(acquisition_s + latency_s)
    if valid and delivery > query_s + 1e-9:
        raise ValueError('Valid translation cannot precede declared delivery')
    age = float(query_s - acquisition_s)
    return dict(**historical_estimator_generation(), source=source, modality='SYNTHETIC_EXTERNAL_POSITION_VELOCITY',
                query_time_s=float(query_s), acquisition_time_s=float(acquisition_s),
                modeled_delivery_time_s=delivery,
                delivery_provenance='DECLARED_FIXED_LATENCY_MODEL_NOT_TRANSPORT_LOG',
                age_s=age, valid=bool(valid),
                position_state_epoch_s=float(query_s),
                velocity_state_epoch_s=float(acquisition_s),
                position_propagated=bool(age > 1e-9),
                velocity_propagated=False,
                position_propagation='CONSTANT_MEASURED_VELOCITY',
                covariance=None)


def exact_time_index(times_s, query_s, *, tolerance_s=1e-9):
    """Align a saved row only when a real timestamp exists, never interpolate."""
    times = np.asarray(times_s, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or not np.isfinite(query_s) or np.any(np.diff(times) <= 0)
            or not np.isfinite(tolerance_s) or tolerance_s < 0):
        raise ValueError('Invalid archived time axis')
    right = int(np.searchsorted(times, query_s))
    candidates = [i for i in (right-1, right) if 0 <= i < len(times)]
    nearest = min(candidates, key=lambda i: abs(times[i] - query_s))
    if abs(times[nearest] - query_s) > tolerance_s:
        raise ValueError('No saved state exists at this estimate epoch')
    return nearest


def vector_error(estimate, truth):
    """Offline vector error only; truth never enters a control measurement."""
    estimate, truth = np.asarray(estimate, dtype=float), np.asarray(truth, dtype=float)
    if (estimate.shape != (3,) or truth.shape != (3,)
            or not np.isfinite(estimate).all() or not np.isfinite(truth).all()):
        raise ValueError('Expected finite three-component state vectors')
    delta = estimate - truth
    return dict(components=delta.tolist(), norm=float(np.linalg.norm(delta)))
