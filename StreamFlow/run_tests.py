#!/usr/bin/env python3
"""
Test runner script for StreamFlow.
"""

import subprocess
import sys
import os

def run_tests():
    """Run the StreamFlow test suite."""
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir  # We're already in the project root
    os.chdir(project_root)

    print("🚀 Running StreamFlow Test Suite")
    print("=" * 50)

    # Check if tests directory exists
    if not os.path.exists("tests"):
        print("❌ Tests directory not found!")
        print("Available directories:", [d for d in os.listdir(".") if os.path.isdir(d)])
        return 1

    # Run unit tests
    print("\n📋 Running Unit Tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short"
    ], capture_output=False)

    if result.returncode != 0:
        print("❌ Unit tests failed!")
        return result.returncode

    # Run integration tests
    print("\n🔗 Running Integration Tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v",
        "--tb=short",
        "-m", "integration"
    ], capture_output=False)

    if result.returncode != 0:
        print("❌ Integration tests failed!")
        return result.returncode

    print("\n✅ All tests passed!")
    print("🎉 StreamFlow test suite completed successfully!")

    return 0

def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("📊 Running Tests with Coverage...")

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "--cov=streamflow",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ], capture_output=False)

    return result.returncode

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--coverage":
        exit_code = run_tests_with_coverage()
    else:
        exit_code = run_tests()

    sys.exit(exit_code)
