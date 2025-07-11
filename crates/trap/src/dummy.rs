use crate::Trap;
use crate::TrapAction;
use crate::TrapError;
use crate::TrapFrame;

pub struct DummyTrap {}

impl Trap for DummyTrap {
    fn enter_trap_and_wait(&self, _frame: TrapFrame) -> Result<TrapAction, TrapError> {
        Ok(TrapAction::Fatal("dummy trap".to_string()))
    }
}
