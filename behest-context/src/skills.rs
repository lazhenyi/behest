//! Skill registry — explicit loading and prompt injection.
//!
//! A "skill" is a markdown/text fragment injected into the system prompt to
//! give the model domain-specific guidance. Unlike aish's background-watcher
//! approach, this registry is purely user-driven: the caller decides when to
//! load, reload, or remove skills. There is no background thread, no file
//! watcher, no automatic behavior.
//!
//! # Stable knowledge injection
//!
//! [`StableKnowledge`] tracks whether a knowledge fragment has changed since
//! the last injection. Use it to avoid rewriting prompt sections that haven't
//! changed — this preserves provider-side prompt-cache prefixes.
//!
//! # Example
//!
//! ```ignore
//! use behest_context::skills::{Skill, SkillRegistry, StableKnowledge};
//! use std::collections::HashMap;
//!
//! let mut registry = SkillRegistry::new();
//! registry.register(Skill::new("git", "Git operations guide", "Always use ..."));
//!
//! // Render the catalog for the system prompt
//! let catalog = registry.render_catalog();
//!
//! // Stable injection — only updates if the catalog changed
//! let mut cache = StableKnowledge::new();
//! if cache.is_stale("skills", &catalog, registry.version()) {
//!     // update the prompt section
//!     cache.update("skills", catalog, registry.version());
//! }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A skill — a named text fragment for system-prompt injection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Skill {
    /// Stable skill name (alphanumeric + dash/underscore, ≤ 64 chars).
    pub name: String,
    /// Short human-readable description (≤ 200 chars).
    pub description: String,
    /// The skill body text injected into the prompt.
    pub content: String,
    /// Optional: file path the skill was loaded from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<PathBuf>,
    /// Optional: tools this skill is allowed to use.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_tools: Vec<String>,
}

impl Skill {
    /// Creates a new skill with the given name, description, and content.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            content: content.into(),
            source_path: None,
            allowed_tools: Vec::new(),
        }
    }

    /// Attaches a source file path.
    #[must_use]
    pub fn with_source_path(mut self, path: PathBuf) -> Self {
        self.source_path = Some(path);
        self
    }

    /// Sets the allowed-tools list.
    #[must_use]
    pub fn with_allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools = tools;
        self
    }
}

/// A registry of skills, keyed by name.
///
/// All mutation is explicit: the caller calls [`register`](Self::register),
/// [`load_from_directory`](Self::load_from_directory), or [`remove`](Self::remove).
/// There is no background watcher — reload happens when the caller says so.
///
/// A `version` counter increments on every mutation, letting consumers detect
/// changes via [`version`](Self::version).
pub struct SkillRegistry {
    skills: HashMap<String, Skill>,
    version: u64,
}

impl std::fmt::Debug for SkillRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillRegistry")
            .field("skill_count", &self.skills.len())
            .field("version", &self.version)
            .finish()
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
            version: 0,
        }
    }

    /// Registers a skill, replacing any with the same name.
    ///
    /// Returns the previous skill if one existed.
    pub fn register(&mut self, skill: Skill) -> Option<Skill> {
        self.version = self.version.saturating_add(1);
        self.skills.insert(skill.name.clone(), skill)
    }

    /// Removes a skill by name. Returns the removed skill if present.
    pub fn remove(&mut self, name: &str) -> Option<Skill> {
        let removed = self.skills.remove(name);
        if removed.is_some() {
            self.version = self.version.saturating_add(1);
        }
        removed
    }

    /// Returns a skill by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    /// Returns all registered skill names, sorted alphabetically.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.skills.keys().cloned().collect();
        names.sort();
        names
    }

    /// Returns the number of registered skills.
    #[must_use]
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Returns `true` when no skills are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Returns the current version counter.
    ///
    /// Increments on every mutation. Use this with [`StableKnowledge`] to
    /// detect when the registry has changed since the last prompt injection.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Renders the skill catalog as a text block for the system prompt.
    ///
    /// Each skill appears as:
    /// ```text
    /// ## <name>
    /// <description>
    /// Path: <source_path or "(inline)">
    /// ```
    ///
    /// Returns an empty string when no skills are registered.
    #[must_use]
    pub fn render_catalog(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }
        let mut entries: Vec<&Skill> = self.skills.values().collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));

        let mut out = String::new();
        for skill in entries {
            out.push_str(&format!(
                "## {}\n{}\nPath: {}\n\n",
                skill.name,
                skill.description,
                skill
                    .source_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(inline)".to_string())
            ));
        }
        out.trim_end().to_string()
    }

    /// Renders the full skill catalog including body content.
    ///
    /// Like [`render_catalog`](Self::render_catalog) but also includes the
    /// `content` body of each skill — use this when injecting the full skill
    /// text into the prompt.
    #[must_use]
    pub fn render_full(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }
        let mut entries: Vec<&Skill> = self.skills.values().collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));

        let mut out = String::new();
        for skill in entries {
            out.push_str(&format!(
                "## {}\n{}\n\n{}\n\n---\n",
                skill.name, skill.description, skill.content
            ));
        }
        out.trim_end().to_string()
    }

    /// Loads skills from a directory.
    ///
    /// Recursively scans for `SKILL.md` (case-insensitive) files. Each file
    /// must begin with a YAML frontmatter block delimited by `---`. Required
    /// frontmatter fields: `name`, `description`. The body after the
    /// frontmatter becomes the skill content.
    ///
    /// Returns the list of skill names loaded. Existing skills with the same
    /// name are replaced. Each successful parse calls [`register`](Self::register),
    /// which bumps the version counter.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read, or if a skill file
    /// is malformed (missing frontmatter or required `name` field).
    pub fn load_from_directory(&mut self, dir: &Path) -> std::io::Result<Vec<String>> {
        let mut loaded = Vec::new();
        self.walk_and_load(dir, &mut loaded)?;
        Ok(loaded)
    }

    fn walk_and_load(&mut self, dir: &Path, loaded: &mut Vec<String>) -> std::io::Result<()> {
        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let meta = entry.metadata()?;
            if meta.is_dir() {
                // Recurse (no symlink-loop detection for simplicity — caller's responsibility)
                self.walk_and_load(&path, loaded)?;
            } else if meta.is_file() {
                let fname = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                if fname.eq_ignore_ascii_case("SKILL.md") {
                    let skill = Self::parse_skill_file(&path)?;
                    let name = skill.name.clone();
                    self.register(skill);
                    loaded.push(name);
                }
            }
        }
        Ok(())
    }

    /// Parses a single SKILL.md file into a [`Skill`].
    ///
    /// Format:
    /// ```text
    /// ---
    /// name: my-skill
    /// description: A short description
    /// allowed_tools:
    ///   - read_file
    ///   - grep
    /// ---
    /// Skill body content here.
    /// ```
    fn parse_skill_file(path: &Path) -> std::io::Result<Skill> {
        let raw = std::fs::read_to_string(path)?;
        let (frontmatter, body) = split_frontmatter(&raw);

        let name = frontmatter
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("skill {} missing 'name' field", path.display()),
                )
            })?
            .to_string();

        let description = frontmatter
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let allowed_tools = frontmatter
            .get("allowed_tools")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Skill {
            name,
            description,
            content: body.trim().to_string(),
            source_path: Some(path.to_path_buf()),
            allowed_tools,
        })
    }
}

/// Splits a SKILL.md file into frontmatter (as a JSON object) and body text.
///
/// Frontmatter is delimited by `---` on its own line at the start of the file.
/// We parse it as YAML via serde_yaml... but to avoid adding serde_yaml as a
/// dependency, we do a minimal manual parse: `key: value` lines and
/// `key:` followed by `- item` lists. Values are wrapped into serde_json::Value.
fn split_frontmatter(raw: &str) -> (serde_json::Value, String) {
    let raw = raw.trim_start_matches('\u{feff}');
    if !raw.starts_with("---") {
        return (
            serde_json::Value::Object(serde_json::Map::new()),
            raw.to_string(),
        );
    }
    // Find the closing ---
    let after_first = &raw[3..];
    let close = match after_first.find("\n---") {
        Some(idx) => idx,
        None => {
            return (
                serde_json::Value::Object(serde_json::Map::new()),
                raw.to_string(),
            );
        }
    };
    let fm_raw = &after_first[..close];
    let body = &after_first[close + 4..];

    let map = parse_minimal_yaml(fm_raw);
    (
        serde_json::Value::Object(map),
        body.trim_start_matches('\n').to_string(),
    )
}

/// Minimal YAML-ish parser supporting `key: value` and `key:` + `- item` lists.
/// Sufficient for SKILL.md frontmatter. Returns a JSON object.
fn parse_minimal_yaml(text: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    let mut current_key: Option<String> = None;
    let mut current_list: Vec<serde_json::Value> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("- ") {
            if let Some(_key) = &current_key {
                let val = trimmed
                    .strip_prefix("- ")
                    .unwrap_or(trimmed)
                    .trim()
                    .trim_matches('"')
                    .to_string();
                current_list.push(serde_json::Value::String(val));
            }
            continue;
        }
        // Flush any pending list
        if let Some(key) = current_key.take()
            && !current_list.is_empty()
        {
            map.insert(
                key,
                serde_json::Value::Array(std::mem::take(&mut current_list)),
            );
        }
        // Parse `key: value`
        if let Some(colon) = trimmed.find(':') {
            let key = trimmed[..colon].trim().to_string();
            let val = trimmed[colon + 1..].trim();
            if val.is_empty() {
                // Start a list
                current_key = Some(key);
            } else {
                let val = val.trim_matches('"').to_string();
                map.insert(key, serde_json::Value::String(val));
            }
        }
    }
    // Flush trailing list
    if let Some(key) = current_key
        && !current_list.is_empty()
    {
        map.insert(key, serde_json::Value::Array(current_list));
    }
    map
}

// ── Stable knowledge injection ──

/// Cache-aware knowledge injection tracker.
///
/// Tracks the last-injected content and source version for each named
/// knowledge slot. Use [`is_stale`](Self::is_stale) to decide whether to
/// rewrite a prompt section — only rewriting when content actually changed
/// preserves provider-side prompt-cache prefixes.
///
/// # Example
///
/// ```ignore
/// use behest_context::skills::StableKnowledge;
///
/// let mut cache = StableKnowledge::new();
/// let new_content = "## skills\n...";
/// let version = registry.version();
///
/// if cache.is_stale("skills", new_content, version) {
///     prompt.update_section("skills", new_content);
///     cache.update("skills", new_content.to_string(), version);
/// }
/// ```
pub struct StableKnowledge {
    entries: HashMap<String, (String, u64)>,
}

impl std::fmt::Debug for StableKnowledge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StableKnowledge")
            .field("slots", &self.entries.len())
            .finish()
    }
}

impl Default for StableKnowledge {
    fn default() -> Self {
        Self::new()
    }
}

impl StableKnowledge {
    /// Creates an empty stable-knowledge cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Returns `true` when the slot's content or version has changed since
    /// the last [`update`](Self::update).
    ///
    /// A slot that has never been updated is always stale.
    #[must_use]
    pub fn is_stale(&self, key: &str, new_content: &str, new_version: u64) -> bool {
        match self.entries.get(key) {
            None => true,
            Some((content, version)) => *content != new_content || *version != new_version,
        }
    }

    /// Updates the cached content and version for a slot.
    pub fn update(&mut self, key: &str, content: String, version: u64) {
        self.entries.insert(key.to_string(), (content, version));
    }

    /// Returns the cached content for a slot, if present.
    #[must_use]
    pub fn cached_content(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(|(c, _)| c.as_str())
    }

    /// Returns the cached version for a slot, if present.
    #[must_use]
    pub fn cached_version(&self, key: &str) -> Option<u64> {
        self.entries.get(key).map(|(_, v)| *v)
    }

    /// Removes a slot from the cache.
    pub fn invalidate(&mut self, key: &str) {
        self.entries.remove(key);
    }

    /// Clears all cached slots.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the number of cached slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no slots are cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn skill_builder_methods() {
        let skill = Skill::new("git", "Git guide", "Always commit often")
            .with_source_path(PathBuf::from("/skills/git/SKILL.md"))
            .with_allowed_tools(vec!["bash".into()]);
        assert_eq!(skill.name, "git");
        assert!(skill.source_path.is_some());
        assert_eq!(skill.allowed_tools, vec!["bash".to_string()]);
    }

    #[test]
    fn registry_register_and_get() {
        let mut reg = SkillRegistry::new();
        assert!(reg.is_empty());
        reg.register(Skill::new("a", "desc a", "body a"));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("a").is_some());
        assert_eq!(reg.version(), 1);
    }

    #[test]
    fn registry_remove() {
        let mut reg = SkillRegistry::new();
        reg.register(Skill::new("a", "desc", "body"));
        let removed = reg.remove("a");
        assert!(removed.is_some());
        assert!(reg.is_empty());
        assert_eq!(reg.version(), 2); // register + remove = 2 mutations
    }

    #[test]
    fn registry_replace_bumps_version() {
        let mut reg = SkillRegistry::new();
        reg.register(Skill::new("a", "desc1", "body1"));
        let v1 = reg.version();
        reg.register(Skill::new("a", "desc2", "body2"));
        assert_eq!(reg.version(), v1 + 1);
        assert_eq!(reg.get("a").unwrap().description, "desc2");
    }

    #[test]
    fn render_catalog_empty() {
        let reg = SkillRegistry::new();
        assert_eq!(reg.render_catalog(), "");
    }

    #[test]
    fn render_catalog_sorted() {
        let mut reg = SkillRegistry::new();
        reg.register(Skill::new("zebra", "z desc", "z body"));
        reg.register(Skill::new("alpha", "a desc", "a body"));
        let catalog = reg.render_catalog();
        let zebra_pos = catalog.find("## zebra").unwrap();
        let alpha_pos = catalog.find("## alpha").unwrap();
        assert!(alpha_pos < zebra_pos);
    }

    #[test]
    fn render_full_includes_body() {
        let mut reg = SkillRegistry::new();
        reg.register(Skill::new("git", "Git guide", "Always commit often"));
        let full = reg.render_full();
        assert!(full.contains("Always commit often"));
        assert!(full.contains("## git"));
    }

    #[test]
    fn names_sorted() {
        let mut reg = SkillRegistry::new();
        reg.register(Skill::new("zebra", "", ""));
        reg.register(Skill::new("alpha", "", ""));
        assert_eq!(reg.names(), vec!["alpha".to_string(), "zebra".to_string()]);
    }

    #[test]
    fn stable_knowledge_new_slot_is_stale() {
        let cache = StableKnowledge::new();
        assert!(cache.is_stale("skills", "content", 1));
    }

    #[test]
    fn stable_knowledge_update_makes_not_stale() {
        let mut cache = StableKnowledge::new();
        cache.update("skills", "content".to_string(), 1);
        assert!(!cache.is_stale("skills", "content", 1));
    }

    #[test]
    fn stable_knowledge_content_change_makes_stale() {
        let mut cache = StableKnowledge::new();
        cache.update("skills", "old".to_string(), 1);
        assert!(cache.is_stale("skills", "new", 1));
    }

    #[test]
    fn stable_knowledge_version_change_makes_stale() {
        let mut cache = StableKnowledge::new();
        cache.update("skills", "same".to_string(), 1);
        assert!(cache.is_stale("skills", "same", 2));
    }

    #[test]
    fn stable_knowledge_invalidate() {
        let mut cache = StableKnowledge::new();
        cache.update("skills", "content".to_string(), 1);
        cache.invalidate("skills");
        assert!(cache.is_stale("skills", "content", 1));
    }

    #[test]
    fn stable_knowledge_cached_accessors() {
        let mut cache = StableKnowledge::new();
        cache.update("skills", "content".to_string(), 5);
        assert_eq!(cache.cached_content("skills"), Some("content"));
        assert_eq!(cache.cached_version("skills"), Some(5));
        assert_eq!(cache.cached_content("missing"), None);
    }

    #[test]
    fn split_frontmatter_parses_yaml() {
        let raw = "---\nname: my-skill\ndescription: A test skill\n---\nBody content here.";
        let (fm, body) = split_frontmatter(raw);
        assert_eq!(fm["name"], serde_json::json!("my-skill"));
        assert_eq!(fm["description"], serde_json::json!("A test skill"));
        assert_eq!(body, "Body content here.");
    }

    #[test]
    fn split_frontmatter_parses_list() {
        let raw = "---\nname: my-skill\nallowed_tools:\n  - read_file\n  - grep\n---\nBody";
        let (fm, body) = split_frontmatter(raw);
        assert_eq!(fm["name"], serde_json::json!("my-skill"));
        let tools = fm["allowed_tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0], serde_json::json!("read_file"));
        assert_eq!(body, "Body");
    }

    #[test]
    fn split_frontmatter_no_frontmatter() {
        let raw = "Just body, no frontmatter.";
        let (fm, body) = split_frontmatter(raw);
        assert!(fm.as_object().unwrap().is_empty());
        assert_eq!(body, "Just body, no frontmatter.");
    }

    #[test]
    fn split_frontmatter_unclosed_frontmatter() {
        let raw = "---\nname: broken\nNo closing delimiter";
        let (fm, _body) = split_frontmatter(raw);
        // Unclosed frontmatter → treated as no frontmatter
        assert!(fm.as_object().unwrap().is_empty());
    }

    #[test]
    fn load_from_directory_loads_skill_files() {
        let tmp = std::env::temp_dir().join("behest_skill_test_load");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let skill_path = tmp.join("SKILL.md");
        std::fs::write(
            &skill_path,
            "---\nname: test-skill\ndescription: A test\n---\nTest body content.",
        )
        .unwrap();

        let mut reg = SkillRegistry::new();
        let loaded = reg.load_from_directory(&tmp).unwrap();
        assert_eq!(loaded, vec!["test-skill".to_string()]);
        let skill = reg.get("test-skill").unwrap();
        assert_eq!(skill.description, "A test");
        assert_eq!(skill.content, "Test body content.");
        assert!(skill.source_path.is_some());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn load_from_directory_recurses() {
        let tmp = std::env::temp_dir().join("behest_skill_test_recurse");
        let _ = std::fs::remove_dir_all(&tmp);
        let subdir = tmp.join("subdir");
        std::fs::create_dir_all(&subdir).unwrap();
        std::fs::write(tmp.join("SKILL.md"), "---\nname: top\n---\nTop body").unwrap();
        std::fs::write(
            subdir.join("SKILL.md"),
            "---\nname: nested\n---\nNested body",
        )
        .unwrap();

        let mut reg = SkillRegistry::new();
        let mut loaded = reg.load_from_directory(&tmp).unwrap();
        loaded.sort();
        assert_eq!(loaded, vec!["nested".to_string(), "top".to_string()]);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn load_from_directory_missing_name_errors() {
        let tmp = std::env::temp_dir().join("behest_skill_test_missing_name");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("SKILL.md"), "---\ndescription: no name\n---\nBody").unwrap();

        let mut reg = SkillRegistry::new();
        let result = reg.load_from_directory(&tmp);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
