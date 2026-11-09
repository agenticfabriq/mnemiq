"""The starter profiles this engine ships: one TOML file per model build, each holding one profile named for its file.

A profile in a deployment's profiles file starts from one by name, `starter = "<name>"`, and its own settings go on
top. A starter holds answer-time settings and mode sections, never an endpoint, a key or roles. docs/profiles.md says
where a starter sits among a profile's layers, and what it can and cannot change.
"""
