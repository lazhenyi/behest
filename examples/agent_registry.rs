//! Demonstrates defining named agents with permissions via `AgentRegistry`.
//!
//! Run with:
//! ```bash
//! cargo run --example agent_registry
//! ```

use behest::agent::{AgentDefinition, AgentRegistry, PermissionEffect, PermissionRule};

fn main() {
    let mut registry = AgentRegistry::new();

    let researcher = AgentDefinition::primary(
        "researcher",
        "Research agent that searches and summarizes information",
        "You are a research assistant. Search the web, read documents, and synthesize findings.",
    );

    let coder = AgentDefinition::primary(
        "coder",
        "Coding agent that writes and reviews code",
        "You are a coding assistant. Write clean, idiomatic Rust code.",
    )
    .with_permissions(vec![
        PermissionRule {
            tool: "write_file".into(),
            resource: "*".into(),
            effect: PermissionEffect::Allow,
        },
        PermissionRule {
            tool: "read_file".into(),
            resource: "*".into(),
            effect: PermissionEffect::Allow,
        },
        PermissionRule {
            tool: "shell_exec".into(),
            resource: "*".into(),
            effect: PermissionEffect::Ask,
        },
    ]);

    let reviewer = AgentDefinition::subagent(
        "reviewer",
        "Code reviewer agent spawned by coder for async review tasks",
        "You review code for correctness, style, and security.",
    );

    registry.register(researcher);
    registry.register(coder);
    registry.register(reviewer);

    println!("Registered {} agents", registry.len());
    println!();

    if let Some(default) = registry.default_agent() {
        println!("Default agent: {} ({})", default.name, default.description);
    }

    println!();

    println!("All agents:");
    for agent in registry.list() {
        let kind = match agent.mode {
            behest::agent::AgentMode::Primary => "primary",
            behest::agent::AgentMode::Subagent => "subagent",
        };
        println!(
            "  {:<12} {}  ({} permissions)",
            agent.name,
            kind,
            agent.permissions.len(),
        );
    }

    println!();

    println!("Selectable agents (non-hidden primary):");
    for agent in registry.list_selectable() {
        println!("  {} — {}", agent.name, agent.description);
    }

    if let Some(found) = registry.get("coder") {
        println!(
            "\nCoder agent permissions: {} rules",
            found.permissions.len()
        );
        for rule in &found.permissions {
            println!(
                "  tool={:?}, resource={:?} => {:?}",
                rule.tool, rule.resource, rule.effect
            );
        }
    }
}
