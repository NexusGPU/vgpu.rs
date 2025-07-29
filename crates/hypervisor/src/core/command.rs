//! Command pattern implementation for CLI operations.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use anyhow::Result;

/// Trait for executable commands
pub trait Command: Send + Sync {
    /// Execute the command
    fn execute(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    /// Get command name for identification
    fn name(&self) -> &'static str;

    /// Get command description
    fn description(&self) -> &'static str {
        "No description available"
    }
}

/// Command registry for managing and executing commands
pub struct CommandRegistry {
    commands: HashMap<String, Box<dyn Command>>,
}

impl CommandRegistry {
    /// Create a new empty command registry
    pub fn new() -> Self {
        Self {
            commands: HashMap::new(),
        }
    }

    /// Register a command in the registry
    pub fn register<C: Command + 'static>(&mut self, command: C) {
        let name = command.name().to_string();
        self.commands.insert(name, Box::new(command));
    }

    /// Execute a command by name
    pub async fn execute(&self, command_name: &str) -> Result<()> {
        let command = self
            .commands
            .get(command_name)
            .ok_or_else(|| anyhow::anyhow!("Command '{}' not found", command_name))?;

        tracing::info!("Executing command: {}", command_name);
        command.execute().await
    }

    /// Get all registered command names
    pub fn list_commands(&self) -> Vec<&String> {
        self.commands.keys().collect()
    }

    /// Check if a command exists
    pub fn has_command(&self, command_name: &str) -> bool {
        self.commands.contains_key(command_name)
    }
}

impl Default for CommandRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Command registry builder for easier command registration
pub struct CommandRegistryBuilder {
    registry: CommandRegistry,
}

impl CommandRegistryBuilder {
    pub fn new() -> Self {
        Self {
            registry: CommandRegistry::new(),
        }
    }

    /// Register a command
    pub fn with_command<C: Command + 'static>(mut self, command: C) -> Self {
        self.registry.register(command);
        self
    }

    /// Build the final command registry
    pub fn build(self) -> CommandRegistry {
        self.registry
    }
}

impl Default for CommandRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}
