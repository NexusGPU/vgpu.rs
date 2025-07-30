//! Core types for the weighted scheduler

use std::ops::{Deref, DerefMut};

/// Represents a trap condition that a process is waiting for
pub struct Trap {
    pub frame: trap::TrapFrame,
    pub waker: Box<dyn trap::Waker>,
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
