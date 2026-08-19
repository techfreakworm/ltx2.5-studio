"""Typed exception hierarchy. Every error carries an actionable detail.

PLAN §5.2 (missing file -> name the file AND its URL), §5.3 (a generation that
produces nothing raises; it never returns an empty path), §4.3 (each validator
stage fails with a message that names what was actually found).

Errors cross layers as exceptions. The UI is the only layer that turns one into
a string for a human (§7.2).

Imports nothing else from this package.
"""
