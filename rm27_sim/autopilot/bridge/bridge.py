"""One new servo frame → one MuJoCo step → one sensor response."""

import argparse
import json
from pathlib import Path
import socket
from .plant_adapter import Plant
from .protocol import ProtocolError, decode_servos, encode_state


class Bridge:
    def __init__(self, plant=None):
        self.plant = Plant() if plant is None else plant
        self.last_count = None
        self.cached_reply = None
        self.epoch = 0
        self.stats = {
            "frames": 0,
            "duplicates": 0,
            "missing": 0,
            "resets": 0,
            "invalid": 0,
        }
        self.rates = set()

    def process(self, payload):
        try:
            packet = decode_servos(payload)
            # SIM_Aircraft starts at 1200 Hz before the first timestamp
            # negotiates 500 Hz. Also applies to restarted SITL and retries.
            if packet.frame_rate != 500 and not (
                packet.frame_count == 0 and packet.frame_rate == 1200
            ):
                raise ProtocolError(
                    f"expected 500 Hz, received {packet.frame_rate}; configure SIM_RATE_HZ=500"
                )
        except ProtocolError:
            self.stats["invalid"] += 1
            raise
        self.rates.add(packet.frame_rate)
        count = packet.frame_count
        if count == self.last_count:
            self.stats["duplicates"] += 1
            return self.cached_reply, None
        if self.last_count is not None:
            wrapped = self.last_count == 0xFFFFFFFF and count == 0
            if count < self.last_count and not wrapped:
                self.plant.reset()
                self.cached_reply = None
                self.epoch += 1
                self.stats["resets"] += 1
            else:
                self.stats["missing"] += (count - self.last_count - 1) % (2**32)
        state = self.plant.step(packet.pwm)
        reply = encode_state(state)
        self.last_count, self.cached_reply = count, reply
        self.stats["frames"] += 1
        row = {
            "frame_count": count,
            "frame_rate": packet.frame_rate,
            "epoch": self.epoch,
            "pwm": list(packet.pwm[:4]),
            "state_sent": state,
            "truth": self.plant.truth(),
        }
        return reply, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--log", type=Path, default=Path("bridge.jsonl"))
    args = parser.parse_args()
    bridge = Bridge()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, args.log.open(
        "w"
    ) as log:
        sock.bind(("127.0.0.1", args.port))
        print(f"Listening on 127.0.0.1:{args.port}, expected 500 Hz", flush=True)
        try:
            while True:
                payload, address = sock.recvfrom(1024)
                try:
                    reply, row = bridge.process(payload)
                except ProtocolError as error:
                    print(str(error), flush=True)
                    continue
                sock.sendto(reply, address)
                if row is not None:
                    log.write(json.dumps(row, allow_nan=False) + "\n")
        except KeyboardInterrupt:
            print(json.dumps(bridge.stats), flush=True)


if __name__ == "__main__":
    main()
