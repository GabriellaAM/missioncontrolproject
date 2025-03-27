#!/usr/bin/env python3
"""
Simple test script to verify that the imports work correctly.
"""
import sys
import os

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    print("\nTrying to import TrendAnalyzer...")
    from main import TrendAnalyzer
    print("✅ Successfully imported TrendAnalyzer from main")
except Exception as e:
    print(f"❌ Failed to import TrendAnalyzer from main: {e}")

try:
    print("\nTrying to import TrendAnalyzer with absolute import...")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from soros_system.main import TrendAnalyzer
    print("✅ Successfully imported TrendAnalyzer from soros_system.main")
except Exception as e:
    print(f"❌ Failed to import TrendAnalyzer with absolute import: {e}")

print("\nImport test complete.") 