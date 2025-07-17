//! HTTP API for querying pod resource information
//!
//! This module provides a REST API for clients to query pod resource allocations
//! including TFLOPS and VRAM usage. Authentication is handled via JWT tokens
//! containing Kubernetes pod and namespace information.
//!
//! # API Endpoints
//!
//! - `GET /api/v1/pods` - Get resource information for the authenticated pod
//!
//! # Authentication
//!
//! All requests must include a JWT token in the Authorization header:
//! ```
//! Authorization: Bearer <JWT_TOKEN>
//! ```
//!
//! The JWT payload must contain Kubernetes information in the following format:
//! ```json
//! {
//!   "kubernetes.io": {
//!     "namespace": "default",
//!     "pod": {
//!       "name": "my-pod",
//!       "uid": "pod-uuid"
//!     },
//!     "node": {
//!       "name": "node-name",
//!       "uid": "node-uuid"
//!     },
//!     "serviceaccount": {
//!       "name": "default",
//!       "uid": "sa-uuid"
//!     }
//!   },
//!   "nbf": 1751311081,
//!   "sub": "user-subject"
//! }
//! ```

pub mod auth;
pub mod errors;
pub mod handlers;
pub mod server;
pub mod types;

// Re-export main types and functions for easy access
