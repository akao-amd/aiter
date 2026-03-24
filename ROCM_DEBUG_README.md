# ROCm Detection Debugging Guide

## Summary of Changes

The `aiter/jit/utils/cpp_extension.py` file has been modified to better handle ROCm installation detection issues and provide better debugging capabilities.

### Key Improvements

1. **Graceful Failure Handling**: Instead of raising a `RuntimeError` when ROCm detection fails, the code now returns `None` and continues, allowing the module to import even without ROCm.

2. **Better Error Handling for hipconfig**: The `get_hip_version()` function now handles cases where:
   - `hipconfig` returns a non-zero exit code but still outputs the version
   - `hipconfig` outputs to stderr instead of stdout
   - `hipconfig` is missing entirely

3. **Debug Environment Variable**:
   - `DEBUG_ROCM=1` - Enable verbose debugging output

## Usage

### Enable Debugging Output

To see detailed ROCm detection information:

```bash
export DEBUG_ROCM=1
python your_script.py
```

This will show:
- ROCm environment variables (ROCM_HOME, ROCM_PATH)
- Location of hipcc and hipconfig in PATH
- Detected ROCM_HOME path
- Any errors encountered during detection

## Debug Script

A debugging script is provided at `/sgl-workspace/aiter/debug_rocm.py`:

```bash
python debug_rocm.py
```

This script will:
1. Check ROCm environment variables
2. Look for ROCm tools in PATH
3. Check common ROCm installation paths
4. Test importing the module
5. Provide recommendations based on findings

## Common Issues and Solutions

### Issue: hipconfig returns non-zero exit code

**Symptom**: Error like `ModuleNotFoundError: No module named 'rocm_sdk_core'`

**Solution**: The updated code now handles this automatically by capturing output from both stdout and stderr, and accepting version strings even when the command fails.

### Issue: ROCm installed in non-standard location

**Solution**: Set the ROCM_HOME environment variable:
```bash
export ROCM_HOME=/path/to/rocm
python your_script.py
```

## Technical Details

### ROCm Detection Flow

1. Call `get_hip_version()`:
   - Find `hipconfig` using `executable_path()`
   - Run `hipconfig --version`
   - Capture output from both stdout and stderr
   - Return version string or `None` on failure
2. Call `_find_rocm_home()`:
   - Check ROCM_HOME and ROCM_PATH environment variables
   - Find hipcc in PATH and derive ROCM_HOME
   - Check `/opt/rocm` as fallback
3. Set global variables:
   - `HIP_VERSION`: Version string or None
   - `ROCM_HOME`: Path to ROCm installation or None
   - `IS_HIP_EXTENSION`: True only if both HIP_VERSION and ROCM_HOME are found
   - `ROCM_VERSION`: Tuple of (major, minor) version or None

### Modified Functions

- `get_hip_version()`: Now returns `None` instead of raising RuntimeError
- Module-level initialization: Now handles `None` values gracefully
