"""Legacy native API; task orchestration belongs to RM27 Non-RL."""
from rm27_sim.simulation.asm2 import *


def _compiled_contact_geometry(*args, **kwargs):
    from rm27_non_rl.geometry import _compiled_contact_geometry as implementation
    return implementation(*args, **kwargs)


def simulate(*args, **kwargs):
    from rm27_non_rl.asm2_contact import simulate as implementation
    return implementation(*args, **kwargs)


def _bounded_plan(*args, **kwargs):
    from rm27_non_rl.asm2_contact import _bounded_plan as implementation
    return implementation(*args, **kwargs)


def main():
    from rm27_non_rl.asm2_contact import main as implementation
    implementation()


if __name__ == '__main__':
    main()
