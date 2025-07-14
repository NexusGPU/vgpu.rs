FROM ubuntu:24.04

# Default path for the binary, can be overridden by build-arg
ARG BINARY_PATH=target/release/hypervisor

# Copy the pre-built binary from the host
COPY ${BINARY_PATH} /usr/local/bin/hypervisor

ENTRYPOINT ["/usr/local/bin/hypervisor", "daemon"]
