#!/usr/bin/env python
"""
Entry point for running experiments as a module.
Enables: python -m experiments.generate_run
"""

from .generate_run import main

if __name__ == "__main__":
    main()