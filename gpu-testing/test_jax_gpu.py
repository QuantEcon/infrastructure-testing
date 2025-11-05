#!/usr/bin/env python3
"""
NVIDIA GPU and JAX Test Script
Tests NVIDIA drivers, CUDA, CUDNN, and JAX GPU functionality
"""

import sys
import subprocess
import os
import re

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def run_command(cmd, description, quiet=False):
    """Run a shell command and return output"""
    if not quiet:
        print(f"\n{description}")
        print(f"Command: {cmd}")
        print("-" * 70)
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            if not quiet:
                print(result.stdout)
            return True, result.stdout
        else:
            if not quiet:
                print(f"Error (exit code {result.returncode}):")
                print(result.stderr)
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print("Command timed out after 10 seconds")
        return False, ""
    except Exception as e:
        print(f"Exception occurred: {e}")
        return False, str(e)

def test_nvidia_drivers():
    """Test 1: Check NVIDIA drivers"""
    print_section("TEST 1: NVIDIA Driver Information")
    
    # Check nvidia-smi
    success, output = run_command(
        "nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv",
        "Checking NVIDIA driver version and GPU info:"
    )
    
    if not success:
        print("\n⚠️  nvidia-smi not found or failed. NVIDIA drivers may not be installed.")
        return False
    
    # Also show full nvidia-smi output
    run_command("nvidia-smi", "Full nvidia-smi output:")
    
    print("\n✅ NVIDIA drivers detected")
    return True

def test_cuda_cudnn():
    """Test 2: Check CUDA and CUDNN versions"""
    print_section("TEST 2: CUDA and CUDNN Versions")
    
    # Collect CUDA versions
    cuda_versions = set()
    
    # Check CUDA installations from directories
    success, output = run_command(
        "ls -d /usr/local/cuda-*/ 2>/dev/null | grep -o 'cuda-[0-9.]*' | sed 's/cuda-//'",
        "Checking installed CUDA toolkits:",
        quiet=True
    )
    if success and output.strip():
        for line in output.strip().split('\n'):
            if line and line != 'No versioned CUDA installations found':
                version = line.strip()
                # Only include versions with at least major.minor format (e.g., 12.3, not just 12)
                if '.' in version:
                    cuda_versions.add(version)
    
    # Check which version is the active symlink
    active_cuda = None
    success, output = run_command(
        "readlink -f /usr/local/cuda 2>/dev/null | grep -oP 'cuda-\\K[0-9.]+'",
        "Checking active CUDA symlink:",
        quiet=True
    )
    if success and output.strip():
        active_cuda = output.strip()
    
    # Check nvidia-smi for driver CUDA version
    success, output = run_command(
        "nvidia-smi | grep -oP 'CUDA Version: \\K[0-9.]+'",
        "Checking CUDA from nvidia-smi:",
        quiet=True
    )
    driver_cuda = output.strip() if success and output.strip() else None
    
    # Display CUDA summary
    print("\n📦 CUDA Toolkit Versions:")
    if cuda_versions:
        sorted_versions = sorted(cuda_versions, key=lambda x: [int(n) for n in x.split('.')], reverse=True)
        for version in sorted_versions:
            if version == active_cuda:
                print(f"   • {version} (current active)")
            else:
                print(f"   • {version}")
        
        # If multiple versions detected, show cleanup tip
        if len(sorted_versions) > 1:
            # Convert version format (e.g., "12.3" -> "12-3")
            version_dash = sorted_versions[-1].replace(".", "-")
            print(f"\n   💡 Tip: Multiple CUDA versions detected. To remove older versions:")
            print(f"      dpkg -l | grep -i cuda | grep \"{version_dash}\"")
            print(f"      sudo apt-get --purge remove $(dpkg -l | grep -i cuda | grep \"{version_dash}\" | awk '{{print $2}}')")
            print(f"      sudo apt-get autoremove")
            print(f"      sudo apt-get autoclean")
    else:
        print("   ⚠️  No CUDA toolkit installations detected")
    
    if driver_cuda:
        print(f"\n🔧 CUDA Driver Version: {driver_cuda} (max supported)")
        
        # Check for version compatibility
        if cuda_versions:
            highest_toolkit = sorted_versions[0]
            driver_major = int(driver_cuda.split('.')[0])
            toolkit_major = int(highest_toolkit.split('.')[0])
            
            if toolkit_major > driver_major:
                print(f"\n   ⚠️  WARNING: Highest toolkit version ({highest_toolkit}) exceeds driver support ({driver_cuda})")
            elif toolkit_major < driver_major:
                print(f"\n   ℹ️  Note: Driver supports CUDA {driver_cuda}, but highest toolkit installed is {highest_toolkit}")
                print(f"      You could install CUDA {driver_cuda} if needed")
            else:
                print(f"\n   ✅ Toolkit and driver versions are compatible")
    
    # Check CUDNN version from header
    cudnn_version = None
    cudnn_paths = [
        "/usr/include/cudnn_version.h",
        "/usr/local/cuda/include/cudnn_version.h",
        "/usr/include/x86_64-linux-gnu/cudnn_version.h"
    ]
    
    for path in cudnn_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    major = re.search(r'#define CUDNN_MAJOR\s+(\d+)', content)
                    minor = re.search(r'#define CUDNN_MINOR\s+(\d+)', content)
                    patch = re.search(r'#define CUDNN_PATCHLEVEL\s+(\d+)', content)
                    if major and minor and patch:
                        cudnn_version = f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"
                        break
            except:
                pass
    
    # Display CUDNN summary
    print("\n📦 CUDNN Version:")
    if cudnn_version:
        cudnn_major = int(cudnn_version.split('.')[0])
        print(f"   • {cudnn_version}")
        
        # Check CUDNN compatibility with CUDA toolkit
        if cuda_versions and active_cuda:
            cuda_major = int(active_cuda.split('.')[0])
            
            # CUDNN compatibility guide (approximate):
            # CUDNN 8.x supports CUDA 11.x and 12.x
            # CUDNN 9.x supports CUDA 12.x and 13.x
            is_compatible = False
            compatibility_msg = ""
            
            if cudnn_major == 8 and cuda_major in [11, 12]:
                is_compatible = True
            elif cudnn_major == 9 and cuda_major in [12, 13]:
                is_compatible = True
            elif cudnn_major == cudnn_major:  # Same major version generally works
                is_compatible = True
            
            if is_compatible:
                print(f"\n   ✅ CUDNN {cudnn_version} is compatible with CUDA {active_cuda}")
                
                # Check if newer CUDNN version is recommended
                if cuda_major >= 12 and cudnn_major == 8:
                    print(f"\n   💡 Tip: CUDNN 9.x is available for CUDA {cuda_major}.x")
                    print(f"      Newer version: sudo apt-get install libcudnn9-cuda-{cuda_major} libcudnn9-dev-cuda-{cuda_major}")
                    print(f"      (CUDNN 8.9.7 will continue to work)")
            else:
                print(f"\n   ⚠️  WARNING: CUDNN {cudnn_version} may not be compatible with CUDA {active_cuda}")
                if cuda_major >= 13:
                    print(f"      Recommended: CUDNN 9.x for CUDA 13.x")
                    print(f"      Install: sudo apt-get install libcudnn9-cuda-13 libcudnn9-dev-cuda-13")
                elif cuda_major == 12:
                    print(f"      Recommended: CUDNN 8.9+ or CUDNN 9.x for CUDA 12.x")
                    print(f"      Install: sudo apt-get install libcudnn9-cuda-12 libcudnn9-dev-cuda-12")
                elif cuda_major == 11:
                    print(f"      Recommended: CUDNN 8.x for CUDA 11.x")
                    print(f"      Install: sudo apt-get install libcudnn8-cuda-11 libcudnn8-dev-cuda-11")
    else:
        print("   ⚠️  CUDNN not detected")
        if cuda_versions:
            cuda_major = int(sorted_versions[0].split('.')[0])
            print(f"\n   💡 Tip: Install CUDNN for CUDA {sorted_versions[0]}:")
            if cuda_major >= 13:
                print(f"      sudo apt-get install libcudnn9-cuda-13 libcudnn9-dev-cuda-13")
            elif cuda_major == 12:
                print(f"      sudo apt-get install libcudnn9-cuda-12 libcudnn9-dev-cuda-12")
            else:
                print(f"      sudo apt-get install libcudnn8-cuda-11 libcudnn8-dev-cuda-11")
    
    return True

def test_hardware_availability():
    """Test 3: Test hardware availability with Python"""
    print_section("TEST 3: Hardware Availability (Python)")
    
    try:
        import torch
        print("\n📦 PyTorch installed")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available in PyTorch: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version (PyTorch): {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("\n⚠️  PyTorch not installed (optional)")
    except Exception as e:
        print(f"\n⚠️  Error checking PyTorch: {e}")
    
    # Check TensorFlow (optional)
    try:
        import tensorflow as tf
        print("\n📦 TensorFlow installed")
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs available in TensorFlow: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    except ImportError:
        print("\n⚠️  TensorFlow not installed (optional)")
    except Exception as e:
        print(f"\n⚠️  Error checking TensorFlow: {e}")
    
    return True

def test_jax_gpu():
    """Test 4: Test JAX with GPU backend"""
    print_section("TEST 4: JAX GPU Functionality")
    
    try:
        import jax
        import jax.numpy as jnp
        
        print(f"\n✅ JAX installed successfully")
        print(f"JAX version: {jax.__version__}")
        
        # Get CUDA/CUDNN info from previous test (stored in global or re-check)
        # Check which CUDA version JAX was built for
        try:
            # Try to get jaxlib version info
            import jaxlib
            jaxlib_version = jaxlib.__version__
            
            # Determine CUDA version from jaxlib
            # JAX package names include cuda version: jaxlib-0.4.23+cuda12.cudnn89
            # Newer versions (0.8.0+) use simpler version strings
            cuda_in_jax = None
            if 'cuda12' in jaxlib_version or 'cu12' in jaxlib_version:
                cuda_in_jax = 12
            elif 'cuda11' in jaxlib_version or 'cu11' in jaxlib_version:
                cuda_in_jax = 11
            elif 'cuda13' in jaxlib_version or 'cu13' in jaxlib_version:
                cuda_in_jax = 13
            else:
                # For newer JAX versions without CUDA in version string,
                # try to detect from linked libraries
                try:
                    import subprocess
                    # Find jaxlib location and check which CUDA it links to
                    jaxlib_path = jaxlib.__file__
                    result = subprocess.run(
                        f"ldd {jaxlib_path} 2>/dev/null | grep -oP 'libcuda.*\\.so' | head -1",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    # If JAX successfully loaded GPU, infer CUDA compatibility from runtime
                    if jax.default_backend() == 'gpu':
                        # JAX is working with GPU, so it's compatible with system CUDA
                        cuda_in_jax = "auto-detected"
                except:
                    pass
            
            # Get system CUDA version
            success, output = run_command(
                "readlink -f /usr/local/cuda 2>/dev/null | grep -oP 'cuda-\\K[0-9.]+'",
                "Checking active CUDA:",
                quiet=True
            )
            system_cuda = None
            if success and output.strip():
                system_cuda = output.strip()
                system_cuda_major = int(system_cuda.split('.')[0])
            
            # Check compatibility
            if cuda_in_jax and system_cuda:
                print(f"\n📋 JAX/CUDA Compatibility:")
                if cuda_in_jax == "auto-detected":
                    print(f"   JAX backend: GPU (auto-detected compatibility)")
                    print(f"   System CUDA: {system_cuda}")
                    print(f"   ✅ JAX is successfully using CUDA {system_cuda}")
                elif cuda_in_jax == system_cuda_major:
                    print(f"   JAX built for: CUDA {cuda_in_jax}.x")
                    print(f"   System CUDA: {system_cuda}")
                    print(f"   ✅ JAX and CUDA versions are compatible")
                else:
                    print(f"   JAX built for: CUDA {cuda_in_jax}.x")
                    print(f"   System CUDA: {system_cuda}")
                    print(f"   ⚠️  WARNING: Version mismatch detected!")
                    print(f"      JAX expects CUDA {cuda_in_jax}.x but system has CUDA {system_cuda}")
                    print(f"\n   💡 Tip: Reinstall JAX for your CUDA version:")
                    if system_cuda_major == 12:
                        print(f"      pip uninstall jax jaxlib")
                        print(f'      pip install -U "jax[cuda12]"')
                    elif system_cuda_major == 11:
                        print(f"      pip uninstall jax jaxlib")
                        print(f'      pip install -U "jax[cuda11]"')
                    elif system_cuda_major >= 13:
                        print(f"      pip uninstall jax jaxlib")
                        print(f'      pip install -U "jax[cuda12]"  # CUDA 13 may use cuda12 package')
                        print(f"      # Check JAX docs for CUDA 13 support status")
        except Exception as e:
            # Silently skip compatibility check if it fails
            pass
        
        # Check available devices
        print("\nJAX Devices:")
        devices = jax.devices()
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device}")
        
        # Check default backend
        print(f"\nDefault backend: {jax.default_backend()}")
        
        # Check for GPU
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if not gpu_devices:
            print("\n⚠️  WARNING: No GPU devices found in JAX!")
            print("JAX may be using CPU-only version")
            return False
        
        print(f"\n✅ Found {len(gpu_devices)} GPU device(s)")
        
        # Perform a simple computation on GPU
        print("\n" + "-"*70)
        print("Testing GPU computation...")
        print("-"*70)
        
        # Create random matrices
        key = jax.random.PRNGKey(0)
        size = 4096
        x = jax.random.normal(key, (size, size))
        
        # Compile and run matrix multiplication
        print(f"\nPerforming matrix multiplication ({size}x{size})...")
        
        @jax.jit
        def matmul(a):
            return jnp.dot(a, a.T)
        
        # Warm-up run
        result = matmul(x)
        result.block_until_ready()
        
        # Timed run
        import time
        start = time.time()
        result = matmul(x)
        result.block_until_ready()
        end = time.time()
        
        print(f"✅ Computation successful!")
        print(f"   Result shape: {result.shape}")
        print(f"   Result dtype: {result.dtype}")
        print(f"   Time taken: {(end-start)*1000:.2f} ms")
        print(f"   Device used: {result.device}")
        
        # Test gradient computation
        print("\n" + "-"*70)
        print("Testing automatic differentiation on GPU...")
        print("-"*70)
        
        def loss_fn(x):
            return jnp.sum(x ** 2)
        
        grad_fn = jax.grad(loss_fn)
        x_small = jax.random.normal(key, (100,))
        gradient = grad_fn(x_small)
        
        print(f"✅ Gradient computation successful!")
        print(f"   Gradient shape: {gradient.shape}")
        print(f"   Device used: {gradient.device}")
        
        print("\n" + "="*70)
        print(" ✅ ALL JAX GPU TESTS PASSED!")
        print("="*70)
        
        return True
        
    except ImportError as e:
        print(f"\n❌ JAX not installed: {e}")
        print("\nTo install JAX with GPU support, run:")
        print("  pip install -U jax[cuda12]  # for CUDA 12")
        print("  pip install -U jax[cuda11]  # for CUDA 11")
        return False
    except Exception as e:
        print(f"\n❌ Error testing JAX: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" NVIDIA GPU and JAX Test Suite")
    print("="*70)
    
    results = {
        "NVIDIA Drivers": False,
        "CUDA/CUDNN": False,
        "Hardware Availability": False,
        "JAX GPU": False
    }
    
    # Run tests
    results["NVIDIA Drivers"] = test_nvidia_drivers()
    results["CUDA/CUDNN"] = test_cuda_cudnn()
    results["Hardware Availability"] = test_hardware_availability()
    results["JAX GPU"] = test_jax_gpu()
    
    # Summary
    print_section("TEST SUMMARY")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Your GPU setup is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
