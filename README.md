# QuantEcon Infrastructure Testing

A collection of diagnostic and testing scripts for validating computational environments used in quantitative economics research and teaching.

## Overview

This repository contains testing utilities to verify that computational environments are properly configured for running QuantEcon materials, including GPU-accelerated computations, neural network training, and other computational economics tasks.

## Available Tools

### [GPU Testing](./gpu-testing/)

**Script:** `test_jax_gpu.py`

Comprehensive diagnostic tool for NVIDIA GPU environments running JAX programs. Validates drivers, CUDA, CUDNN, and JAX GPU functionality with actual computations.

**Quick usage:**
```bash
cd gpu-testing
python3 test_jax_gpu.py
```

**Tests:**
- NVIDIA driver installation and version
- CUDA toolkit and CUDNN library versions
- GPU hardware detection across frameworks
- JAX GPU backend functionality with performance benchmarks

[→ See full documentation](./gpu-testing/README.md)

---

## Contributing

Contributions are welcome! If you have additional test scripts or improvements to existing ones, please submit a pull request.

When adding new tools:
1. Create a descriptive folder for the tool
2. Include a comprehensive README in the folder
3. Update this main README with a brief description and link

## Support

For issues related to:
- **Test scripts:** Open an issue in this repository
- **QuantEcon materials:** See [QuantEcon Discourse](https://discourse.quantecon.org/)

## Authors

QuantEcon Team

## License

MIT License

---

**Note:** This repository supports computational economics research and education using modern numerical methods and hardware acceleration.
