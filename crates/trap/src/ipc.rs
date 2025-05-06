use crate::{Trap, TrapAction, TrapFrame, TrapHandler};
use ipc_channel::ipc::{self, IpcOneShotServer, IpcReceiver, IpcSender};
use std::sync::Mutex;

/// IpcTrap: IPC implementation of Trap (client side, sends TrapFrame and waits for TrapAction)
#[derive(Debug)]
pub struct IpcTrap {
    sender: Mutex<IpcSender<TrapFrame>>,
    receiver: Mutex<IpcReceiver<TrapAction>>,
}

impl IpcTrap {
    pub fn new(sender: IpcSender<TrapFrame>, receiver: IpcReceiver<TrapAction>) -> Self {
        Self {
            sender: Mutex::new(sender),
            receiver: Mutex::new(receiver),
        }
    }

    /// Creates a client-side IpcTrap by connecting to a server using the provided server name.
    /// This should be used in a different process than the one that called `create_server`.
    pub fn connect<S: ToString>(server_name: S) -> Result<Self, crate::TrapError> {
        // Connect to the server
        let tx = IpcSender::connect(server_name.to_string())?;

        // Create channels for sending frames and receiving actions
        let (frame_sender, frame_receiver) = ipc::channel()?;
        let (action_sender, action_receiver) = ipc::channel()?;

        // Send our channel endpoints to the server
        tx.send((action_sender, frame_receiver))?;

        // Return a new IpcTrap with our side of the channels
        Ok(Self::new(frame_sender, action_receiver))
    }
}

impl Trap for IpcTrap {
    fn enter_trap_and_wait(&self, frame: TrapFrame) -> Result<TrapAction, crate::TrapError> {
        self.sender
            .lock()
            .unwrap()
            .send(frame)
            .map_err(crate::TrapError::Ipc)?;
        self.receiver
            .lock()
            .unwrap()
            .recv()
            .map_err(crate::TrapError::IpcRecv)
    }
}

/// IpcTrapHandler: IPC implementation of TrapHandler (server side, receives TrapFrame, processes and returns TrapAction)
pub struct IpcTrapHandler<H: TrapHandler + Send + Sync + 'static> {
    handler: H,
    frame_receiver: IpcReceiver<TrapFrame>,
    action_sender: IpcSender<TrapAction>,
}

impl<H: TrapHandler + Send + Sync + 'static> IpcTrapHandler<H> {
    pub fn new(
        handler: H,
        frame_receiver: IpcReceiver<TrapFrame>,
        action_sender: IpcSender<TrapAction>,
    ) -> Self {
        Self {
            handler,
            frame_receiver,
            action_sender,
        }
    }

    /// Creates a server endpoint that a client can connect to.
    /// Returns the server and the connection name to share with the client.
    pub fn create_server(handler: H) -> Result<(Self, String), Box<ipc_channel::ErrorKind>> {
        // Create a one-shot server to receive the client's channel endpoints
        let (server, server_name) = IpcOneShotServer::new()?;

        // Clone the server name before moving the server into accept_connection
        let name_to_return = server_name.clone();

        // Accept a connection and create the handler (this will block until a client connects)
        let handler = Self::accept_connection(handler, server)?;

        Ok((handler, name_to_return))
    }

    /// Accept a connection from a client and create an IpcTrapHandler.
    /// This should be run in a different process than the one that calls `IpcTrap::connect`.
    pub fn accept_connection(
        handler: H,
        server: IpcOneShotServer<(IpcSender<TrapAction>, IpcReceiver<TrapFrame>)>,
    ) -> Result<Self, Box<ipc_channel::ErrorKind>> {
        // Accept the client connection and get their channel endpoints
        let (_, (client_action_sender, client_frame_receiver)) = server.accept()?;

        // Create the handler with the received channels
        Ok(Self::new(
            handler,
            client_frame_receiver,
            client_action_sender,
        ))
    }

    /// Start the event loop: receive TrapFrame, call handler, and send TrapAction
    pub fn start(&self) {
        while let Ok(frame) = self.frame_receiver.recv() {
            let sender = self.action_sender.clone();
            self.handler.handle_trap(
                &frame,
                Box::new(move |result| {
                    // here if send fails, it can't be fed back to the client, can only ignore or log
                    let _ = sender.send(result.unwrap_or_else(|e| {
                        crate::TrapAction::Fatal(format!("TrapHandler error: {e}"))
                    }));
                }),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use ipc_channel::ipc::channel;

    use super::*;
    use crate::{TrapAction, TrapFrame, TrapHandler};
    use std::{io, thread};

    /// Helper function: create a pair of IPC channels, returning (Trap side, Handler side)
    pub fn create_ipc_trap_pair<H: TrapHandler + Send + Sync + 'static>(
        handler: H,
    ) -> Result<(IpcTrap, IpcTrapHandler<H>), io::Error> {
        let (frame_sender, frame_receiver) = channel::<TrapFrame>()?;
        let (action_sender, action_receiver) = channel::<TrapAction>()?;
        Ok((
            IpcTrap::new(frame_sender, action_receiver),
            IpcTrapHandler::new(handler, frame_receiver, action_sender),
        ))
    }

    struct DummyHandler;
    impl TrapHandler for DummyHandler {
        fn handle_trap(
            &self,
            frame: &TrapFrame,
            waker: Box<dyn FnOnce(Result<TrapAction, crate::TrapError>) + Send>,
        ) {
            match frame {
                TrapFrame::OutOfMemory { requested_bytes } if *requested_bytes < 4096 => {
                    waker(Ok(TrapAction::Resume));
                }
                TrapFrame::OutOfMemory { requested_bytes } => {
                    waker(Ok(TrapAction::Fatal(format!("OOM: {}", requested_bytes))));
                }
            }
        }
    }

    #[test]
    fn test_ipc_trap_resume_and_fatal() {
        let (trap, handler_side) =
            create_ipc_trap_pair(DummyHandler).expect("create_ipc_trap_pair");
        let handler_thread = thread::spawn(move || {
            handler_side.start();
        });

        // Test Resume
        let frame = TrapFrame::OutOfMemory {
            requested_bytes: 1024,
        };
        let action = trap
            .enter_trap_and_wait(frame)
            .expect("trap enter_trap_and_wait");
        assert!(matches!(action, TrapAction::Resume));

        // Test Fatal
        let frame = TrapFrame::OutOfMemory {
            requested_bytes: 9999,
        };
        let action = trap
            .enter_trap_and_wait(frame)
            .expect("trap enter_trap_and_wait");
        assert!(matches!(action, TrapAction::Fatal(msg) if msg == "OOM: 9999"));

        // Drop trap to close the channel and exit the handler thread
        drop(trap);
        handler_thread.join().ok();
    }
}
