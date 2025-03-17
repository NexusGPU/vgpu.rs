FROM ubuntu:22.04
WORKDIR /home/app

# Default path for the binary, can be overridden by build-arg
ARG BINARY_PATH=target/release/tensor-fusion-hypervisor

# Copy the pre-built binary from the host
COPY ${BINARY_PATH} ./tensor-fusion-hypervisor
