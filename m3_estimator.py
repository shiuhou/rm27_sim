"""Legacy M3 benchmark API; canonical estimator is owned by RM27 Non-RL."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module('rm27_non_rl.m3_estimator')
