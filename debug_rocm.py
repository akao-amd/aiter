#!/usr/bin/env python3
"""Debug script to test ROCm detection."""

import os
import sys
import shutil

print("=" * 60)
print("ROCm Detection Debug Script")
print("=" * 60)

# Check environment variables
print("\n1. Environment Variables:")
print(f"   ROCM_HOME: {os.environ.get('ROCM_HOME', 'NOT SET')}")
print(f"   ROCM_PATH: {os.environ.get('ROCM_PATH', 'NOT SET')}")
print(f"   DEBUG_ROCM: {os.environ.get('DEBUG_ROCM', 'NOT SET')}")

# Check for ROCm tools in PATH
print("\n2. ROCm Tools in PATH:")
print(f"   hipcc: {shutil.which('hipcc') or 'NOT FOUND'}")
print(f"   hipconfig: {shutil.which('hipconfig') or 'NOT FOUND'}")

# Check common ROCm installation paths
print("\n3. Common ROCm Installation Paths:")
common_paths = ["/opt/rocm", "/opt/rocm-5.7.0", "/usr/local/rocm"]
for path in common_paths:
    exists = os.path.exists(path)
    hipconfig = os.path.join(path, "bin", "hipconfig")
    hipconfig_exists = os.path.exists(hipconfig)
    print(f"   {path}: {'EXISTS' if exists else 'NOT FOUND'}")
    if exists:
        print(f"     - hipconfig: {'EXISTS' if hipconfig_exists else 'NOT FOUND'}")

# Try importing the module
print("\n4. Attempting to import aiter.jit.utils.cpp_extension:")
try:
    from aiter.jit.utils import cpp_extension
    print("   SUCCESS")
    print(f"   IS_HIP_EXTENSION: {cpp_extension.IS_HIP_EXTENSION}")
    print(f"   ROCM_HOME: {cpp_extension.ROCM_HOME}")
    print(f"   HIP_VERSION: {cpp_extension.HIP_VERSION}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Recommendations:")
print("=" * 60)

if not shutil.which('hipconfig'):
    print("• hipconfig not found in PATH")
    print("  - If ROCm is installed, add it to PATH or set ROCM_HOME")
    print("  - Ensure ROCm is properly installed")

if not any(os.path.exists(p) for p in common_paths):
    print("• ROCm installation not found in common paths")
    print("  - Install ROCm or set ROCM_HOME to installation directory")

print("\nTo enable debug output, run:")
print("  export DEBUG_ROCM=1")
print("  python your_script.py")
