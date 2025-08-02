use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use crate::TrapError;
use ipc_channel::ipc::{self, IpcOneShotServer, IpcReceiver, IpcReceiverSet, IpcSender};

use crate::Trap;
use crate::TrapAction;
use crate::TrapFrame;
use crate::TrapHandler;
use crate::Waker;

/// IPC implementation of Waker trait
struct IpcWaker {
    sender: Arc<Mutex<IpcSender<(u64, TrapAction)>>>,
}

#[async_trait::async_trait]
impl Waker for IpcWaker {
    async fn send(&self, trap_id: u64, action: TrapAction) -> Result<(), TrapError> {
        let sender = self.sender.lock().expect("poisoned");
        sender.send((trap_id, action)).map_err(TrapError::Ipc)
    }
}

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
#[derive(Clone)]
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
        std::thread::spawn(move || {
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
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self, TrapError> {
        // Get our process ID
        let pid = std::process::id();

        // Construct the expected filename where the server name is stored
        let filename = path.as_ref().join(format!("trap_server_{pid}.addr"));

        // Wait for the file to appear with a timeout
        let poll_interval = Duration::from_millis(300); // How often to check if no signal

        if !filename.exists() {
            loop {
                //  Check if file exists
                if filename.exists() {
                    break;
                }
                thread::sleep(poll_interval);
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
}

impl Trap for IpcTrap {
    fn enter_trap_and_wait(&self, frame: TrapFrame) -> Result<TrapAction, TrapError> {
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
        self.sender.send((trap_id, frame)).map_err(TrapError::Ipc)?;

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

#[derive(Debug)]
struct Client {
    sender: IpcSender<(u64, TrapAction)>,
    pid: u32,
}

#[derive(Debug)]
struct AddressFileGuard {
    ipc_path: PathBuf,
    pid: u32,
}

impl Drop for AddressFileGuard {
    fn drop(&mut self) {
        let filename = self.ipc_path.join(format!("trap_server_{}.addr", self.pid));
        if filename.exists() {
            if let Err(e) = fs::remove_file(&filename) {
                tracing::warn!("Failed to remove IPC addr file for PID {}: {}", self.pid, e);
            }
        }
    }
}

/// IpcTrapServer: IPC implementation of TrapHandler (server side, receives TrapFrame, processes and returns TrapAction)
pub struct IpcTrapServer<H: TrapHandler + Send + Sync + 'static> {
    handler: H,
    // ReceiverId -> Client
    clients: Mutex<HashMap<u64, Client>>,
    ipc_receiver_set: Mutex<IpcReceiverSet>,
    addr_guards: Mutex<Vec<AddressFileGuard>>,
}

impl<H: TrapHandler + Send + Sync + 'static> IpcTrapServer<H> {
    /// Create a new IpcTrapServer with the given handler
    pub fn new(handler: H) -> Result<Self, TrapError> {
        let ipc_receiver_set = IpcReceiverSet::new()?;

        Ok(Self {
            handler,
            clients: Mutex::new(HashMap::new()),
            ipc_receiver_set: ipc_receiver_set.into(),
            addr_guards: Mutex::new(Vec::new()),
        })
    }

    /// Wait for a client with the specified PID to connect
    pub fn wait_client<P: AsRef<Path>>(&self, path: P, pid: u32) -> Result<(), TrapError> {
        // Create a one-shot server that will receive the initial connection
        let (server, server_name) =
            IpcOneShotServer::<(IpcSender<(u64, TrapAction)>, IpcReceiver<(u64, TrapFrame)>)>::new(
            )?;

        // Write the server_name to the file before accepting connections
        let filename = path.as_ref().join(format!("trap_server_{pid}.addr"));
        fs::write(&filename, server_name)?;
        let addr_guard = AddressFileGuard {
            ipc_path: path.as_ref().to_path_buf(),
            pid,
        };
        self.addr_guards.lock().expect("poisoning").push(addr_guard);

        // Wait for the client to connect and send its channels
        let (_, channel_pair) = server.accept()?;
        // extract the two channels
        let (action_sender, frame_receiver) = channel_pair;

        // Add the frame receiver to our receiver set
        let receiver_id = self
            .ipc_receiver_set
            .lock()
            .expect("poisoning")
            .add(frame_receiver)?;

        // Create a new client entry
        let client = Client {
            sender: action_sender,
            pid,
        };

        self.clients
            .lock()
            .expect("poisoned")
            .insert(receiver_id, client);

        Ok(())
    }

    /// Remove a client by PID
    /// This will remove the client from internal tracking and trigger the Drop implementation
    /// which will clean up associated resources like the addr file
    pub fn remove_client(&self, pid: u32) {
        self.addr_guards
            .lock()
            .expect("poisoning")
            .retain(|guard| guard.pid != pid);

        let receiver_id = {
            let clients_lock = self.clients.lock().expect("poisoned");
            clients_lock
                .iter()
                .find(|(_, client)| client.pid == pid)
                .map(|(id, _)| *id)
        };

        if let Some(id) = receiver_id {
            let removed = {
                let mut clients_lock = self.clients.lock().expect("poisoned");
                clients_lock.remove(&id).is_some()
            };

            // The receiver will be automatically removed from the set when the channel is closed
            // We don't need to explicitly remove it as IpcReceiverSet doesn't provide a removal method
            // It will detect closed channels in the next select() call
            if removed {
                tracing::debug!("Removed client with PID {}", pid);
            }
        }
    }

    pub async fn run(&self) -> Result<(), TrapError> {
        loop {
            let events = self.ipc_receiver_set.lock().expect("poisoning").select()?;
            for event in events {
                match event {
                    ipc::IpcSelectionResult::MessageReceived(id, msg) => {
                        // Extract the trap ID and frame from the message
                        if let Ok((trap_id, frame)) = msg.to::<(u64, TrapFrame)>() {
                            // Get the client associated with this receiver ID
                            let client_info = {
                                let clients = self.clients.lock().expect("poisoning");
                                clients
                                    .get(&id)
                                    .map(|client| (client.pid, client.sender.clone()))
                            };

                            if let Some((client_pid, sender)) = client_info {
                                // Handle the trap using the provided handler
                                // The handler will now need to include the trap_id when sending the response
                                let waker = IpcWaker {
                                    sender: Arc::new(Mutex::new(sender)),
                                };
                                self.handler
                                    .handle_trap(client_pid, trap_id, &frame, Box::new(waker))
                                    .await;
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
}
