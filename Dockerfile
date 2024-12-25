FROM rust:1.79.0-bookworm as builder

WORKDIR /home/app
COPY . .
RUN cargo build --release

FROM ubuntu:22.04
WORKDIR /home/app

COPY --from=builder /home/app/target/release/tensor-fusion-hypervisor ./
