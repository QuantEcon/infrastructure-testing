# GPU Testing for JAX

A comprehensive diagnostic script that validates NVIDIA GPU environments for JAX-based computational workloads.

## Overview

`test_jax_gpu.py` performs systematic checks of the GPU stack from drivers through to application-level computation, ensuring your environment is properly configured for JAX-based computational economics work.

## What it Tests

- ✅ **NVIDIA Drivers** - Installation and version verification
- ✅ **CUDA Toolkit** - CUDA compiler and runtime detection
- ✅ **CUDNN Library** - CUDNN version and library availability
- ✅ **Hardware Detection** - GPU availability across frameworks (PyTorch, TensorFlow)
- ✅ **JAX GPU Functionality** - Actual GPU computations with performance timing

## Quick Start

### Prerequisites

**System Requirements:**
- Linux operating system (Ubuntu/Debian recommended)
- NVIDIA GPU with compute capability 3.5 or higher
- NVIDIA drivers (version 450.80.02 or higher)
- CUDA toolkit (version 11.1 or higher)
- CUDNN library (version 8.0 or higher)

**Python Requirements:**
- Python 3.8 or higher
- JAX with GPU support

### Installation

1. **Install NVIDIA Drivers**

   ```bash
   # Check for recommended driver
   ubuntu-drivers devices
   
   # Install recommended driver
   sudo ubuntu-drivers autoinstall
   ```

2. **Install CUDA Toolkit**

   ```bash
   # Example for CUDA 12.x on Ubuntu
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

   **Tip: Clean up older CUDA versions**
   
   If you have multiple CUDA versions installed and want to clean up older ones (i.e. such as CUDA==12.3):
   
   ```bash
   dpkg -l | grep -i cuda | grep "12-3"
   sudo apt-get --purge remove $(dpkg -l | grep -i cuda | grep "12-3" | awk '{print $2}')
   sudo apt-get autoremove
   sudo apt-get autoclean
   ```

3. **Install CUDNN**

   ```bash
   # For CUDA 12.x
   sudo apt-get install libcudnn9-cuda-12 libcudnn9-dev-cuda-12
   
   # For CUDA 11.x
   sudo apt-get install libcudnn8-cuda-11 libcudnn8-dev-cuda-11
   
   # For CUDA 13.x
   sudo apt-get install libcudnn9-cuda-13 libcudnn9-dev-cuda-13
   ```

4. **Install JAX with GPU Support**

   ```bash
   # For CUDA 12
   pip install -U "jax[cuda12]"
   
   # For CUDA 11
   pip install -U "jax[cuda11]"
   ```

### Usage

```bash
python3 test_jax_gpu.py
```

### Expected Output

When everything is working correctly:

```
==================================================================
 TEST SUMMARY
==================================================================
NVIDIA Drivers.......................................... ✅ PASS
CUDA/CUDNN.............................................. ✅ PASS
Hardware Availability................................... ✅ PASS
JAX GPU................................................. ✅ PASS

🎉 All tests passed! Your GPU setup is working correctly.
```

**Detailed output includes:**

**Test 2: CUDA and CUDNN Versions**
```
📦 CUDA Toolkit Versions:
   • 12.9 (current active)
   • 12.4
   • 12.3

   💡 Tip: Multiple CUDA versions detected. To remove older versions:
      # List all CUDA packages
      dpkg -l | grep cuda

      # Remove specific version (e.g., CUDA 12.3)
      sudo apt-get --purge remove '*cuda-12.3*'

🔧 CUDA Driver Version: 13.0 (max supported)

   ℹ️  Note: Driver supports CUDA 13.0, but highest toolkit installed is 12.9
      You could install CUDA 13.0 if needed

📦 CUDNN Version:
   • 8.9.7

   ✅ CUDNN 8.9.7 is compatible with CUDA 12.9

   💡 Tip: CUDNN 9.x is available for CUDA 12.x
      Newer version: sudo apt-get install libcudnn9-cuda-12 libcudnn9-dev-cuda-12
      (CUDNN 8.9.7 will continue to work)
```

**Test 4: JAX GPU Functionality**
```
✅ JAX installed successfully
JAX version: 0.8.0

📋 JAX/CUDA Compatibility:
   JAX backend: GPU (auto-detected compatibility)
   System CUDA: 12.9
   ✅ JAX is successfully using CUDA 12.9
```

## Script Architecture

The script is organized into five main components:

1. **Utility Functions** - Output formatting and command execution helpers
2. **Driver Testing** - NVIDIA driver validation via `nvidia-smi`
3. **CUDA/CUDNN Testing** - Toolkit and library version detection
4. **Hardware Testing** - Cross-framework GPU availability checks
5. **JAX Testing** - JAX-specific GPU functionality with actual computations

### Test Details

#### Test 1: NVIDIA Driver Information

Validates driver installation and reports:
- Driver version
- GPU model name
- Total GPU memory
- Full `nvidia-smi` status

#### Test 2: CUDA and CUDNN Versions

**New Features:**
- Detects all installed CUDA toolkit versions
- Shows which version is currently active (symlink target)
- Compares driver CUDA support with installed toolkits
- Provides cleanup tips for multiple installations
- Displays full CUDNN version (e.g., 8.9.7)
- Checks CUDNN/CUDA compatibility
- Recommends newer CUDNN versions when available
- Uses CUDA-specific package names (e.g., `libcudnn9-cuda-12`)

**What it reports:**
- All CUDA toolkit installations (with active version marked)
- Driver CUDA version (maximum supported)
- Compatibility status between driver and toolkits
- Full CUDNN version from header files
- CUDNN/CUDA compatibility check
- Cleanup commands for old versions
- Upgrade recommendations with exact install commands

#### Test 3: Hardware Availability

Tests GPU detection across frameworks:
- **PyTorch** - CUDA availability and device information
- **TensorFlow** - GPU device enumeration

*Note: PyTorch and TensorFlow are optional dependencies.*

#### Test 4: JAX GPU Functionality

**New Features:**
- Detects JAX/CUDA version compatibility
- Auto-detects compatibility for newer JAX versions (0.8.0+)
- Warns about version mismatches
- Provides specific reinstall commands

**Performs actual GPU computations:**

1. **Installation verification** - JAX version and available devices
2. **CUDA compatibility check** - Validates JAX works with system CUDA
3. **GPU device detection** - Filters and counts GPU devices
4. **Matrix multiplication test** - 4096×4096 matrices with JIT compilation
5. **Gradient computation** - Automatic differentiation on GPU

**Performance metrics:**
- Execution time
- Result shape and dtype
- Device placement verification

## Troubleshooting

### Common Issues

#### `nvidia-smi not found`

**Cause:** NVIDIA drivers not installed or not in PATH

**Solution:** Install NVIDIA drivers and reboot

```bash
sudo apt install nvidia-driver-535
sudo reboot
```

#### `JAX not installed`

**Solution:** Install JAX with GPU support

```bash
pip install -U "jax[cuda12]"
```

#### `No GPU devices found in JAX`

**Possible causes:**
- CPU-only JAX installed
- CUDA/CUDNN version mismatch
- Library path issues

**Solution:**

```bash
# Reinstall JAX with GPU support
pip uninstall jax jaxlib
pip install -U "jax[cuda12]"

# Ensure CUDA is in your path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```

#### `CUDNN library not found`

**Solution:** Install CUDNN for your CUDA version

```bash
# For CUDA 12.x
sudo apt-get install libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# For CUDA 11.x
sudo apt-get install libcudnn8-cuda-11 libcudnn8-dev-cuda-11

# For CUDA 13.x
sudo apt-get install libcudnn9-cuda-13 libcudnn9-dev-cuda-13
```

#### `CUDA version mismatch`

**Symptoms:** JAX finds no GPU despite working nvidia-smi

**Solution:** Match JAX CUDA version with installed CUDA:

```bash
# For CUDA 12.x
pip install "jax[cuda12]"

# For CUDA 11.x
pip install "jax[cuda11]"
```

#### `Multiple CUDA versions detected`

**Symptoms:** Test shows multiple CUDA library versions installed

**Solution:** Clean up older CUDA versions to avoid conflicts:

```bash
# List all CUDA packages
dpkg -l | grep cuda

# Remove specific version (e.g., CUDA 11.8)
sudo apt-get --purge remove "*cuda-11-8*"

# Update alternatives if using CUDA from /usr/local
sudo update-alternatives --config cuda

# Verify only desired version remains
ldconfig -p | grep libcuda
```

### Debug Commands

```bash
# Check JAX backend
python3 -c "import jax; print(jax.default_backend())"

# Check library path
echo $LD_LIBRARY_PATH

# Verify NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version
```

## Environment Configuration

Add these to your `~/.bashrc` or `~/.bash_profile`:

```bash
# CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Optional: Force JAX to use GPU
export JAX_PLATFORMS=gpu
```

## Customization

### Adjust Matrix Size

For GPUs with less memory, reduce the test matrix size:

```python
# In test_jax_gpu() function
size = 2048  # Reduced from 4096
```

### Add Custom CUDNN Paths

```python
# In test_cuda_cudnn() function
cudnn_paths = [
    "/usr/include/cudnn_version.h",
    "/usr/local/cuda/include/cudnn_version.h",
    "/usr/include/x86_64-linux-gnu/cudnn_version.h",
    "/your/custom/path/cudnn_version.h"
]
```

### Modify Command Timeout

```python
# In run_command() function
timeout=30  # Changed from 10 seconds
```

## Dependencies

### Required
- Python 3.8+
- Standard library: `sys`, `subprocess`, `os`, `time`

### For Full Functionality
- NVIDIA drivers
- CUDA toolkit
- CUDNN library
- JAX with GPU support

### Optional
- PyTorch (enhances hardware detection test)
- TensorFlow (enhances hardware detection test)

## Best Practices

1. **Run after fresh installation** - Validate setup immediately
2. **Run before major computations** - Ensure GPU is available
3. **Save output logs** - Keep records of system configuration
4. **Compare across systems** - Benchmark different hardware

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX Installation Guide](https://github.com/google/jax#installation)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [QuantEcon Website](https://quantecon.org/)

## Support

For issues with this script:
- Open an issue in this repository
- Include full script output
- Include `nvidia-smi` output
- Include `pip list | grep jax` output

## License

MIT License - See repository root for details
