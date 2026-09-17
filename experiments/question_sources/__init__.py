"""Adapters that turn inspected external sources into candidate questions.

Each adapter reads a source this thesis did not author, and emits candidate
questions whose reference answer is traceable back to a named record with a
date and a stable identifier. No adapter invents a question, an answer or a
citation.

Source files are NOT vendored. They carry their own licences and some are
large; each adapter takes a path and fails loudly if it is absent.
"""
