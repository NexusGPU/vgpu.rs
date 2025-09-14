//! Core types for the weighted scheduler

use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use trap::{TrapFrame, Waker};

/// Represents a trap condition that a process is waiting for
pub struct Trap {
    pub frame: Arc<TrapFrame>,
    pub waker: Box<dyn Waker>,
    pub round: u32,
}

/// Wrapper for processes that includes associated traps
pub struct WithTraps<Proc> {
    pub process: Proc,
    pub traps: Vec<Trap>,
}

impl<Proc> Deref for WithTraps<Proc> {
    type Target = Proc;

    fn deref(&self) -> &Self::Target {
        &self.process
    }
}

impl<Proc> DerefMut for WithTraps<Proc> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.process
    }
}
