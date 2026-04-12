"""Framework-specific integration code for rlwatch.

Each submodule provides the glue between an RL training framework and
rlwatch's ``RLWatch.log_step()`` pipeline. The integrations are imported
lazily by ``core.py::_attach_<framework>()`` so users who never touch a
given framework never pay the import cost.
"""
