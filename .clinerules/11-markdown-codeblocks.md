# Markdown Code Block Formatting Guidelines

This document establishes formatting standards for bash commands in markdown files and Jupyter notebooks.

## Key Requirements

The guidelines mandate that "each flag or argument on its own line" using backslash continuations. Commands exceeding two arguments should follow this multi-line format with two-space indentation.

## Special Cases

Long `curl` commands containing JSON bodies must be placed in dedicated scripts rather than embedded directly in documentation or conversation.

## Purpose

These rules enhance readability and maintainability of complex command-line instructions by breaking them into logical, easy-to-scan components rather than presenting them as unwieldy single lines.
