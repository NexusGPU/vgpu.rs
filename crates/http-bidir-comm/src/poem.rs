//! Poem integration for `http-bidir-comm`.
//!
//! This module provides helper functions to expose the [`HttpServer`] as
//! Poem routes, including:
//!
//! * `GET {base_path}/events/:client_id` – SSE stream that pushes queued
//!   tasks to the client.
//! * `POST {base_path}/result`       – endpoint for clients to submit
//!   [`TaskResult`] values back to the server.
//!
//! With this wrapper, users can effortlessly embed the bidirectional
//! communication server into their existing Poem applications:
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use http_bidir_comm::poem::create_routes;
//! use http_bidir_comm::HttpServer;
//! use poem::Route;
//! use poem::Server;
//!
//! type MyTask = String;
//! type MyResult = String;
//!
//! #[tokio::main]
//! async fn main() {
//!     let server = Arc::new(HttpServer::<MyTask, MyResult>::new());
//!     let routes = create_routes(server.clone(), "/api");
//!
//!     // add the generated routes into your existing router tree
//!     let app = Route::new().nest("/api", routes);
//!     let listener = poem::listener::TcpListener::bind("0.0.0.0:8080");
//!     Server::new(listener).run(app).await.unwrap();
//! }
//! ```

use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use poem::endpoint::make;
use poem::get;
use poem::http::StatusCode;
use poem::post;
use poem::web::sse::Event;
use poem::web::sse::SSE;
use poem::web::FromRequest;
use poem::web::Path;
use poem::IntoResponse;
pub use poem::Route;

use crate::server::HttpServer;
use crate::types::TaskResult;

/// Create Poem `Route`s for the given `HttpServer`.
pub fn create_routes<T, R>(server: Arc<HttpServer<T, R>>, base_path: &str) -> Route
where
    T: Clone
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>
        + Send
        + Sync
        + 'static
        + core::fmt::Debug,
    R: Clone
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>
        + Send
        + Sync
        + 'static
        + core::fmt::Debug,
{
    // SSE events endpoint
    let srv_events = Arc::clone(&server);
    let events_ep = make(move |req: poem::Request| {
        let server = Arc::clone(&srv_events);
        async move {
            // Extract client_id from path
            let Path(client_id) = Path::<String>::from_request_without_body(&req).await?;

            // Build an async stream of events
            let sse_stream = stream! {
                loop {
                    match server.poll_task_internal(&client_id).await {
                        Ok(Some(task)) => {
                            if let Ok(json) = serde_json::to_string(&task) {
                                yield Event::message(json);
                            }
                        }
                        Ok(None) => {
                            tokio::time::sleep(Duration::from_millis(300)).await;
                        }
                        Err(err) => {
                            yield Event::message(err.to_string()).event_type("error");
                            break;
                        }
                    }
                }
            };

            let response = SSE::new(sse_stream)
                .keep_alive(Duration::from_secs(15))
                .into_response();
            Ok::<poem::Response, poem::Error>(response)
        }
    });

    // Result submission endpoint
    let srv_submit = Arc::clone(&server);
    let result_ep = make(move |req: poem::Request| {
        let server = Arc::clone(&srv_submit);
        async move {
            // Extract body as bytes
            let body = req.into_body();
            let bytes = body.into_vec().await.unwrap_or_default();
            let result: TaskResult<R> = match serde_json::from_slice(&bytes) {
                Ok(r) => r,
                Err(_) => {
                    return Ok::<poem::Response, poem::Error>(
                        StatusCode::BAD_REQUEST.into_response(),
                    );
                }
            };

            let status = match server.submit_result_internal(result).await {
                Ok(_) => StatusCode::OK,
                Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            Ok::<poem::Response, poem::Error>(status.into_response())
        }
    });

    Route::new()
        .at(format!("{base_path}/events/:client_id"), get(events_ep))
        .at(format!("{base_path}/result"), post(result_ep))
}
