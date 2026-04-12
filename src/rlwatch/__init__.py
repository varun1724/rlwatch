"""rlwatch - Real-time GRPO/PPO training instability detection."""

__version__ = "0.3.1"

from rlwatch.core import attach, log_step, get_monitor, RLWatch
from rlwatch.config import RLWatchConfig, load_config

__all__ = ["attach", "log_step", "get_monitor", "RLWatch", "RLWatchConfig", "load_config"]
