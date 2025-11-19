#!/usr/bin/env python3
"""Test script to verify module imports work from docs directory."""

import sys
import os

# Add project root to path (same as conf.py)
sys.path.insert(0, os.path.abspath(".."))

try:
    import backtester
    print("✓ Successfully imported backtester")
    print(f"  Module file: {backtester.__file__}")
    
    import backtester.strategy
    print("✓ Successfully imported backtester.strategy")
    print(f"  BaseStrategy docstring: {backtester.strategy.BaseStrategy.__doc__[:100]}...")
    
    import backtester.data
    print("✓ Successfully imported backtester.data")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print(f"Python path: {sys.path}")