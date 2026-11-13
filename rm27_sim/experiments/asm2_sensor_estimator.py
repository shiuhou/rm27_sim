"""Compatibility entry; implementation belongs to RM27 Non-RL."""
import importlib
import sys

if __name__ == '__main__':
    importlib.import_module('rm27_non_rl.asm2_sensor_estimator').main()
else:
    sys.modules[__name__] = importlib.import_module('rm27_non_rl.asm2_sensor_estimator')
