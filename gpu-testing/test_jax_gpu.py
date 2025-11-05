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

def run_command(cmd, description):
    """Run a shell command and return output"""
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
            print(result.stdout)
            return True, result.stdout
        else:
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
    
    # Check CUDA version from nvcc
    print("\nChecking CUDA version (nvcc):")
    run_command("nvcc --version", "NVCC compiler version:")
    
    # Check CUDA version from nvidia-smi
    run_command(
        "nvidia-smi | grep 'CUDA Version'",
        "CUDA version from nvidia-smi:"
    )
    
    # Check CUDA runtime version
    print("\nChecking for CUDA libraries:")
    success, output = run_command("ldconfig -p | grep libcuda", "CUDA libraries in system:")
    if success:
        # Extract unique CUDA library versions
        versions = set()
        for line in output.split('\n'):
            # Look for patterns like libcudart.so.12.0 or libcuda.so.1
            match = re.search(r'libcuda\S+\.so\.(\d+(?:\.\d+)*)', line)
            if match:
                versions.add(match.group(1))
        if versions:
            print("\nDetected CUDA library versions:")
            for version in sorted(versions, reverse=True):
                print(f"  - {version}")
    
    # Check CUDNN
    print("\nChecking for CUDNN libraries:")
    success, output = run_command("ldconfig -p | grep libcudnn", "CUDNN libraries in system:")
    if success:
        # Extract CUDNN library versions
        versions = set()
        for line in output.split('\n'):
            # Look for patterns like libcudnn.so.8
            match = re.search(r'libcudnn\S*\.so\.(\d+(?:\.\d+)*)', line)
            if match:
                versions.add(match.group(1))
        if versions:
            print("\nDetected CUDNN library versions:")
            for version in sorted(versions, reverse=True):
                print(f"  - {version}")
    
    # Try to find CUDNN version from header file
    cudnn_paths = [
        "/usr/include/cudnn_version.h",
        "/usr/local/cuda/include/cudnn_version.h",
        "/usr/include/x86_64-linux-gnu/cudnn_version.h"
    ]
    
    print("\nSearching for CUDNN version header:")
    for path in cudnn_paths:
        if os.path.exists(path):
            print(f"Found: {path}")
            run_command(
                f"cat {path} | grep '#define CUDNN_MAJOR\\|#define CUDNN_MINOR\\|#define CUDNN_PATCHLEVEL'",
                f"CUDNN version from {path}:"
            )
            break
    else:
        print("CUDNN version header not found in common locations")
    
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
