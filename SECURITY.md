# Security Policy

## Supported Versions

We actively support the following versions of vgpu.rs with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

Security is critical for vgpu.rs, especially given its role as a GPU hypervisor and CUDA API interceptor. If you discover a security vulnerability, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via:

- **Primary**: security@tensor-fusion.com
- **Alternative**: support@tensor-fusion.com
- **Encrypted**: Use our PGP key for sensitive reports (contact us for the public key)

### What to Include

Please provide the following information:

- **Vulnerability Type**: Memory safety issue, privilege escalation, information disclosure, etc.
- **Affected Components**: 
  - `libcuda_limiter.so` (CUDA API interception)
  - `libadd_path.so` (Environment variable manipulation)
  - Core hypervisor logic
  - Build system or dependencies
- **Affected Versions**: Specific version numbers or commit hashes
- **Environment Details**: 
  - Operating system and version
  - CUDA driver version
  - GPU model and driver
  - Rust toolchain version
- **Reproduction Steps**: Clear, step-by-step instructions
- **Proof of Concept**: Code or commands demonstrating the issue
- **Impact Assessment**: Potential security implications
- **Suggested Fix**: If you have ideas for mitigation

### Response Timeline

- **Acknowledgment**: Within 24 hours
- **Initial Assessment**: Within 3 business days
- **Status Updates**: Weekly until resolution
- **Critical Issues**: Addressed within 7 days
- **High/Medium Issues**: Addressed within 30 days
- **Low Issues**: Addressed within 90 days
