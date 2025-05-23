use crate::{Trap, TrapAction, TrapError, TrapFrame, TrapHandler};
use ipc_channel::ipc::{self, IpcOneShotServer, IpcReceiver, IpcReceiverSet, IpcSender};
use signal_hook::consts::signal::SIGUSR1;
use signal_hook::iterator::Signals;
use std::collections::HashMap;
use std::io::Error as IoError;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::{fs, thread, time::Duration};

/// Represents a pending trap request waiting for a response
#[derive(Debug)]
struct PendingTrap {
    action: Option<TrapAction>,
}

/// Type alias for a mutex and condvar pair containing a pending trap
type PendingTrapPair = Arc<(Mutex<PendingTrap>, Condvar)>;

/// Type alias for a map of trap IDs to pending trap pairs
type PendingTrapsMap = HashMap<u64, PendingTrapPair>;

/// IpcTrap: IPC implementation of Trap (client side, sends TrapFrame and waits for TrapAction)
#[derive(Debug, Clone)]
pub struct IpcTrap {
    sender: IpcSender<(u64, TrapFrame)>,
    pending_traps: Arc<Mutex<PendingTrapsMap>>,
    next_trap_id: Arc<AtomicU64>,
}

impl IpcTrap {
    pub fn new(
        sender: IpcSender<(u64, TrapFrame)>,
        receiver: IpcReceiver<(u64, TrapAction)>,
    ) -> Self {
        let pending_traps = Arc::new(Mutex::new(
            HashMap::<u64, Arc<(Mutex<PendingTrap>, Condvar)>>::new(),
        ));
        let pending_traps_clone = Arc::clone(&pending_traps);

        // Start a dedicated thread to handle incoming messages
        thread::spawn(move || {
            loop {
                match receiver.recv() {
                    Ok((id, action)) => {
                        let maybe_trap = pending_traps_clone.lock().expect("poisoning").remove(&id);
                        if let Some(trap) = maybe_trap {
                            let (mutex, condvar) = &*trap;
                            let mut pending = mutex.lock().expect("poisoning");
                            pending.action = Some(action);
                            condvar.notify_one();
                        }
                        // If trap not found, it might have timed out or been cancelled
                    }
                    Err(e) => {
                        // Channel closed or error occurred
                        tracing::error!("IPC receiver error: {:?}", e);
                        break;
                    }
                }
            }
        });

        Self {
            sender,
            pending_traps,
            next_trap_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Creates a client-side IpcTrap by connecting to a server.
    /// This should be used in a different process than the one that called `wait_client`.
    /// This method waits for a SIGUSR1 signal or the server name file to appear, with a timeout.
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self, crate::TrapError> {
        // Get our process ID
        let pid = unsafe { libc::getpid() } as u32;

        // Construct the expected filename where the server name is stored
        let filename = path.as_ref().join(format!("trap_server_{}.addr", pid));

        // Wait for the file to appear with a timeout
        let poll_interval = Duration::from_millis(300); // How often to check if no signal

        if !filename.exists() {
            // Register a signal handler for SIGUSR1.
            // The `Signals` instance handles registration and unregistration on drop.
            let mut signals = Signals::new([SIGUSR1]).map_err(TrapError::Io)?;

            loop {
                // Priority 1: Check if file exists
                if filename.exists() {
                    break;
                }

                // Priority 3: Check for pending signals (non-blocking)
                let mut signal_received_this_iteration = false;
                if let Some(_signal) = signals.pending().next() {
                    // SIGUSR1 received, loop will immediately check filename.exists()
                    signal_received_this_iteration = true;
                }

                // If no signal was pending and file still doesn't exist, sleep.
                if !signal_received_this_iteration {
                    thread::sleep(poll_interval);
                }
                // If a signal was received, we loop immediately to re-check filename.
            }
        }

        // Read the server name from the file
        let server_name = fs::read_to_string(&filename).map_err(TrapError::Io)?;

        // Clean up the file after reading it
        let _ = fs::remove_file(&filename); // Ignore error if removal fails

        // Connect to the server
        let tx = IpcSender::connect(server_name)?;

        // Create channels for sending frames and receiving actions
        let (frame_sender, frame_receiver): (IpcSender<(u64, TrapFrame)>, _) = ipc::channel()?;
        let (action_sender, action_receiver): (IpcSender<(u64, TrapAction)>, _) = ipc::channel()?;

        // Send our channel endpoints to the server
        tx.send((action_sender, frame_receiver))?;

        // Return a new IpcTrap with our side of the channels
        Ok(Self::new(frame_sender, action_receiver))
    }

    pub fn dummy() -> Self {
        let (frame_sender, _): (IpcSender<(u64, TrapFrame)>, _) =
            ipc::channel().expect("poisoning");
        let (_, action_receiver): (_, IpcReceiver<(u64, TrapAction)>) =
            ipc::channel().expect("poisoning");
        Self::new(frame_sender, action_receiver)
    }
}

impl Trap for IpcTrap {
    fn enter_trap_and_wait(&self, frame: TrapFrame) -> Result<TrapAction, crate::TrapError> {
        // Generate a unique ID for this trap request
        let trap_id = self.next_trap_id.fetch_add(1, Ordering::SeqCst);

        // Create a new pending trap entry
        let pending_trap = PendingTrap { action: None };

        // Create a mutex and condvar pair for this trap
        let pair = Arc::new((Mutex::new(pending_trap), Condvar::new()));

        // Register this trap in our pending traps map
        {
            let mut traps = self.pending_traps.lock().expect("poisoning");
            traps.insert(trap_id, Arc::clone(&pair));
        }

        // Send the frame with its ID to the server
        self.sender
            .send((trap_id, frame))
            .map_err(crate::TrapError::Ipc)?;

        // Wait for the response using the condvar
        let (mutex, condvar) = &*pair;
        let mut pending = mutex.lock().expect("poisoning");

        // Wait until the action is set by the receiver thread
        while pending.action.is_none() {
            pending = condvar.wait(pending).expect("poisoning");
        }

        // Return the action
        Ok(pending.action.take().expect("poisoning"))
    }
}

struct Client {
    sender: IpcSender<(u64, TrapAction)>,
    pid: u32,
}

/// IpcTrapServer: IPC implementation of TrapHandler (server side, receives TrapFrame, processes and returns TrapAction)
pub struct IpcTrapServer<H: TrapHandler + Send + Sync + 'static> {
    handler: H,
    // ReceiverId -> Client
    clients: Mutex<HashMap<u64, Client>>,
    ipc_receiver_set: IpcReceiverSet,
}

impl<H: TrapHandler + Send + Sync + 'static> IpcTrapServer<H> {
    /// Create a new IpcTrapServer with the given handler
    pub fn new(handler: H) -> Result<Self, TrapError> {
        let ipc_receiver_set = IpcReceiverSet::new()?;

        Ok(Self {
            handler,
            clients: Mutex::new(HashMap::new()),
            ipc_receiver_set,
        })
    }

    /// Wait for a client with the specified PID to connect
    pub fn wait_client<P: AsRef<Path>>(&mut self, path: P, pid: u32) -> Result<(), TrapError> {
        // Create a one-shot server that will receive the initial connection
        let (server, server_name) = IpcOneShotServer::new()?;

        // Wait for the client to connect and send its channels
        let (receiver, sender) = server.accept()?;

        // Add the frame receiver to our receiver set
        let receiver_id = self.ipc_receiver_set.add(receiver)?;

        // Store the client information
        let client = Client { sender, pid };

        self.clients
            .lock()
            .expect("poisoned")
            .insert(receiver_id, client);

        let filename = path.as_ref().join(format!("trap_server_{}.addr", pid));
        if let Ok(mut file) = fs::File::create(&filename) {
            use std::io::Write;
            let _ = file.write_all(server_name.as_bytes());
        }

        // Send a signal to notify the process that the server name is available
        // Using libc kill function to send SIGUSR1 (signal 10)
        unsafe {
            let ret = libc::kill(pid as libc::pid_t, libc::SIGUSR1);
            if ret == -1 {
                Err(TrapError::Io(IoError::last_os_error()))
            } else {
                Ok(())
            }
        }
    }

    pub fn run(&mut self) -> Result<(), crate::TrapError> {
        loop {
            let events = self.ipc_receiver_set.select()?;
            for event in events {
                match event {
                    ipc::IpcSelectionResult::MessageReceived(id, msg) => {
                        // Extract the trap ID and frame from the message
                        if let Ok((trap_id, frame)) = msg.to::<(u64, TrapFrame)>() {
                            // Get the client associated with this receiver ID
                            let clients = self.clients.lock().expect("poisoning");
                            if let Some(client) = clients.get(&id) {
                                // Handle the trap using the provided handler
                                // The handler will now need to include the trap_id when sending the response
                                self.handler.handle_trap(
                                    client.pid,
                                    trap_id,
                                    &frame,
                                    client.sender.clone(),
                                );
                            }
                        }
                    }
                    ipc::IpcSelectionResult::ChannelClosed(id) => {
                        self.clients.lock().expect("poisoned").remove(&id);
                    }
                }
            }
        }
    }

    /// Runs the server until the provided stop function returns true
    /// This is primarily useful for testing when you need to stop the server
    pub fn run_with_stop<F>(&mut self, mut should_stop: F) -> Result<(), crate::TrapError>
    where
        F: FnMut() -> bool,
    {
        loop {
            // Check if we should stop
            if should_stop() {
                return Ok(());
            }

            // Try to get events without blocking (using select would block indefinitely)
            let events_result = self.ipc_receiver_set.select();

            match events_result {
                Ok(events) => {
                    for event in events {
                        match event {
                            ipc::IpcSelectionResult::MessageReceived(id, msg) => {
                                // Extract the trap ID and frame from the message
                                if let Ok((trap_id, frame)) = msg.to::<(u64, TrapFrame)>() {
                                    // Get the client associated with this receiver ID
                                    let clients = self.clients.lock().expect("poisoning");
                                    if let Some(client) = clients.get(&id) {
                                        // Handle the trap using the provided handler
                                        self.handler.handle_trap(
                                            client.pid,
                                            trap_id,
                                            &frame,
                                            client.sender.clone(),
                                        );
                                    }
                                }
                            }
                            ipc::IpcSelectionResult::ChannelClosed(id) => {
                                self.clients.lock().expect("poisoned").remove(&id);
                            }
                        }
                    }
                }
                Err(_) => {
                    // If no events, sleep a little bit to avoid busy waiting
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }
}
