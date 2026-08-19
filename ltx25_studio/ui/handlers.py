"""Thin adapters: UI event -> one call into ``generate``/``lora`` -> UI update.

PLAN §7.2: no business logic lives here. Every handler marshals inputs, calls
exactly one function, and marshals outputs. That rule is what keeps the app
testable without a GPU and without Gradio.
"""
