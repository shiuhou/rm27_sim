"""Local SIM_JSON.h wire contract (little-endian SITL host)."""

from dataclasses import dataclass
import json
import struct


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ServoPacket:
    frame_rate: int
    frame_count: int
    pwm: tuple[int, ...]


def decode_servos(payload: bytes) -> ServoPacket:
    layouts = {40: (18458, "<HHI16H"), 72: (29569, "<HHI32H")}
    if len(payload) not in layouts:
        raise ProtocolError(f"invalid servo packet size {len(payload)}")
    magic, layout = layouts[len(payload)]
    received, rate, count, *pwm = struct.unpack(layout, payload)
    if received != magic or rate == 0:
        raise ProtocolError("invalid magic or zero frame_rate")
    return ServoPacket(rate, count, tuple(pwm))


def encode_state(state: dict) -> bytes:
    return (
        "\n" + json.dumps(state, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("ascii")
