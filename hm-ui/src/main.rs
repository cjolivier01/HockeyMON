use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use eframe::egui;
use serde::{Deserialize, Serialize};

mod workspace;

#[derive(Debug, Parser)]
#[command(name = "hm-ui")]
#[command(about = "HockeyMON runtime operator UI")]
struct Args {
    #[arg(long, required_unless_present = "workspace")]
    spec: Option<PathBuf>,
    #[arg(long, required_unless_present = "workspace")]
    state: Option<PathBuf>,
    #[arg(long)]
    workspace: Option<PathBuf>,
    #[arg(long, default_value = "HM UI")]
    title: String,
}

/// Highest control-spec version this build understands. Reported back to the
/// producer so it can tell that a field it wrote was actually rendered; echoing
/// the spec's own number instead would pass at every future bump.
const SUPPORTED_SPEC_VERSION: u32 = 3;

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct UiSpec {
    #[serde(default)]
    session_id: String,
    /// Producer's schema version. Echoed into the state file so the producer can
    /// tell whether the sidecar it is talking to understands the spec it wrote.
    #[serde(default)]
    version: u32,
    #[serde(default)]
    title: String,
    #[serde(default)]
    subtitle: String,
    #[serde(default)]
    preview_path: Option<PathBuf>,
    #[serde(default)]
    action_ack_path: Option<PathBuf>,
    #[serde(default)]
    runtime_path: Option<PathBuf>,
    #[serde(default)]
    previews: Vec<PreviewSpec>,
    #[serde(default)]
    windows: Vec<WindowSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct PreviewSpec {
    name: String,
    path: PathBuf,
    #[serde(default)]
    metadata_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct PreviewDimensions {
    width: u32,
    height: u32,
}

fn preview_label(name: &str, dimensions: Option<&PreviewDimensions>) -> String {
    let name = view_label(name);
    match dimensions {
        Some(size) => format!("{name} ({} × {})", size.width, size.height),
        None => name.to_string(),
    }
}

fn parse_preview_dimensions(data: &str) -> Result<PreviewDimensions> {
    let size: PreviewDimensions = serde_json::from_str(data).context("parse preview dimensions")?;
    anyhow::ensure!(
        size.width > 0 && size.height > 0,
        "preview dimensions must be positive"
    );
    Ok(size)
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct WindowSpec {
    name: String,
    #[serde(default)]
    controls: Vec<ControlSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ControlSpec {
    name: String,
    max_value: i32,
    value: i32,
    #[serde(default)]
    min_value: i32,
    #[serde(default)]
    label: String,
    /// Displayed value = wire value / divisor + offset. Absent metadata retains
    /// the legacy name-based units for older producers.
    #[serde(default)]
    divisor: Option<f64>,
    #[serde(default)]
    offset: f64,
    #[serde(default)]
    unit: String,
    #[serde(default)]
    default_value: Option<i32>,
    #[serde(default)]
    system_default_value: Option<i32>,
    #[serde(default)]
    group: String,
    #[serde(default)]
    view: String,
    #[serde(default)]
    value_revision: u64,
    /// Labels for an enumerated control, indexed by value. Empty means a plain
    /// slider (or a checkbox when `max_value` is 1); the value stays an integer
    /// either way, so state, revisions and resets are unchanged.
    #[serde(default)]
    choices: Vec<String>,
    /// Optional one-line note shown under the widget, e.g. when a control only
    /// takes effect on the next run.
    #[serde(default)]
    description: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct UiState {
    version: u32,
    #[serde(default)]
    spec_version: u32,
    updated_ms: u128,
    #[serde(default)]
    windows: BTreeMap<String, BTreeMap<String, i32>>,
    #[serde(default)]
    control_revisions: BTreeMap<String, BTreeMap<String, u64>>,
    #[serde(default)]
    selected_preview: Option<String>,
    #[serde(default)]
    actions: Vec<UiAction>,
    #[serde(default)]
    last_action: Option<UiAction>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct UiAction {
    seq: u64,
    kind: String,
    #[serde(default)]
    windows: BTreeMap<String, BTreeMap<String, i32>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct UiActionAck {
    seq: u64,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RuntimeStatus {
    state: String,
    frame: Option<u64>,
    fps: Option<f64>,
    rate: Option<f64>,
    #[serde(default)]
    message: String,
    updated_ms: u128,
    #[serde(default)]
    capabilities: Vec<String>,
}

impl RuntimeStatus {
    fn permits(&self, action: &str) -> bool {
        self.capabilities.iter().any(|item| item == action)
            && match action {
                "pause" => self.state == "running",
                "resume" | "step" => self.state == "paused",
                "stop" => matches!(self.state.as_str(), "running" | "paused"),
                _ => false,
            }
    }
}

#[derive(Debug, Clone)]
struct PreviewTransform {
    zoom: f32,
    pan: egui::Vec2,
}

impl Default for PreviewTransform {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
        }
    }
}

impl PreviewTransform {
    fn zoom_at(&mut self, factor: f32, anchor: egui::Vec2) {
        let next = (self.zoom * factor).clamp(1.0, 16.0);
        self.pan = anchor + (self.pan - anchor) * (next / self.zoom);
        self.zoom = next;
        if self.zoom == 1.0 {
            self.pan = egui::Vec2::ZERO;
        }
    }
}

struct HmUiApp {
    workspace: Option<workspace::WorkspaceUi>,
    spec_path: PathBuf,
    state_path: PathBuf,
    spec: UiSpec,
    values: BTreeMap<String, BTreeMap<String, i32>>,
    control_revisions: BTreeMap<String, BTreeMap<String, u64>>,
    selected_page: usize,
    selected_preview: usize,
    preview_maximized: bool,
    preview_transforms: BTreeMap<String, PreviewTransform>,
    last_runtime_poll: SystemTime,
    runtime: Option<RuntimeStatus>,
    runtime_error: Option<String>,
    action_feedback: Option<String>,
    last_ack_seq: u64,
    last_spec_modified: Option<SystemTime>,
    last_spec_poll: SystemTime,
    last_action_ack_poll: SystemTime,
    last_preview_modified: BTreeMap<String, Option<SystemTime>>,
    last_preview_poll: SystemTime,
    preview_textures: BTreeMap<String, egui::TextureHandle>,
    preview_status: BTreeMap<String, String>,
    preview_dimensions: BTreeMap<String, PreviewDimensions>,
    action_seq: u64,
    actions: Vec<UiAction>,
    last_action: Option<UiAction>,
    status: String,
}

impl HmUiApp {
    fn new(spec_path: PathBuf, state_path: PathBuf, title: String) -> Self {
        let mut app = Self {
            workspace: None,
            spec_path,
            state_path,
            spec: UiSpec {
                session_id: String::new(),
                version: 0,
                title,
                subtitle: "Runtime camera controls".to_string(),
                preview_path: None,
                action_ack_path: None,
                runtime_path: None,
                previews: Vec::new(),
                windows: Vec::new(),
            },
            values: BTreeMap::new(),
            control_revisions: BTreeMap::new(),
            selected_page: 0,
            selected_preview: 0,
            preview_maximized: false,
            preview_transforms: BTreeMap::new(),
            last_runtime_poll: UNIX_EPOCH,
            runtime: None,
            runtime_error: None,
            action_feedback: None,
            last_ack_seq: 0,
            last_spec_modified: None,
            last_spec_poll: UNIX_EPOCH,
            last_action_ack_poll: UNIX_EPOCH,
            last_preview_modified: BTreeMap::new(),
            last_preview_poll: UNIX_EPOCH,
            preview_textures: BTreeMap::new(),
            preview_status: BTreeMap::new(),
            preview_dimensions: BTreeMap::new(),
            action_seq: 0,
            actions: Vec::new(),
            last_action: None,
            status: "Starting".to_string(),
        };
        if let Err(err) = app.reload_spec(true) {
            app.status = format!("Waiting for spec: {err}");
        }
        app
    }

    fn reload_spec(&mut self, force: bool) -> Result<()> {
        let meta = fs::metadata(&self.spec_path)
            .with_context(|| format!("metadata {}", self.spec_path.display()))?;
        let modified = meta.modified().ok();
        if !force && modified.is_some() && modified == self.last_spec_modified {
            return Ok(());
        }

        let data = fs::read_to_string(&self.spec_path)
            .with_context(|| format!("read {}", self.spec_path.display()))?;
        let mut spec: UiSpec = serde_json::from_str(&data).context("parse UI spec")?;
        validate_spec(&spec)?;
        clamp_spec_choices(&mut spec);
        if spec.title.is_empty() {
            spec.title = "HM UI".to_string();
        }
        if spec.previews.is_empty() {
            if let Some(path) = spec.preview_path.clone() {
                spec.previews.push(PreviewSpec {
                    name: "Preview".to_string(),
                    path,
                    metadata_path: None,
                });
            }
        }
        if !spec.session_id.is_empty() && spec.session_id != self.spec.session_id {
            self.values.clear();
            self.control_revisions.clear();
            self.actions.clear();
            self.last_action = None;
            self.action_seq = 0;
            self.last_ack_seq = 0;
            self.action_feedback = None;
            self.runtime = None;
            self.runtime_error = None;
            self.last_runtime_poll = UNIX_EPOCH;
            self.preview_textures.clear();
            self.last_preview_modified.clear();
            self.preview_status.clear();
        }
        for window in &spec.windows {
            let entry = self.values.entry(window.name.clone()).or_default();
            let revisions = self
                .control_revisions
                .entry(window.name.clone())
                .or_default();
            for control in &window.controls {
                let previous_revision = revisions.get(&control.name).copied();
                if !entry.contains_key(&control.name)
                    || previous_revision.is_none()
                    || control.value_revision > previous_revision.unwrap_or_default()
                {
                    entry.insert(control.name.clone(), control.value);
                    revisions.insert(control.name.clone(), control.value_revision);
                } else if !control.choices.is_empty() {
                    // A value carried over from an earlier spec can still be out
                    // of range for a control that has since gained choices.
                    let limit = control.choices.len() as i32 - 1;
                    if let Some(value) = entry.get_mut(&control.name) {
                        *value = (*value).clamp(0, limit);
                    }
                }
            }
            let valid_names: Vec<String> = window.controls.iter().map(|c| c.name.clone()).collect();
            entry.retain(|name, _| valid_names.contains(name));
            revisions.retain(|name, _| valid_names.contains(name));
        }
        let valid_windows: Vec<String> = spec.windows.iter().map(|w| w.name.clone()).collect();
        self.values.retain(|name, _| valid_windows.contains(name));
        self.control_revisions
            .retain(|name, _| valid_windows.contains(name));
        let previous_page = control_pages(&self.spec).get(self.selected_page).cloned();
        let previous_preview = self
            .spec
            .previews
            .get(self.selected_preview)
            .map(|p| p.name.clone());
        let pages = control_pages(&spec);
        if let Some(page) = previous_page {
            self.selected_page = pages.iter().position(|item| item == &page).unwrap_or(0);
        }
        if let Some(name) = previous_preview {
            self.selected_preview = spec
                .previews
                .iter()
                .position(|p| p.name == name)
                .unwrap_or(0);
        }
        if self.selected_page >= pages.len() + 2 {
            self.selected_page = 0;
        }
        if self.selected_preview >= spec.previews.len() {
            self.selected_preview = 0;
        }
        let preview_names: Vec<String> =
            spec.previews.iter().map(|item| item.name.clone()).collect();
        self.last_preview_modified
            .retain(|name, _| preview_names.contains(name));
        self.preview_textures
            .retain(|name, _| preview_names.contains(name));
        self.preview_status
            .retain(|name, _| preview_names.contains(name));
        self.preview_dimensions
            .retain(|name, _| preview_names.contains(name));
        self.preview_transforms
            .retain(|name, _| preview_names.contains(name));
        self.last_spec_modified = modified;
        if self.spec.runtime_path != spec.runtime_path {
            self.runtime = None;
            self.runtime_error = None;
            self.last_runtime_poll = UNIX_EPOCH;
        }
        self.spec = spec;
        if self.selected_page < pages.len() || self.selected_page == usize::MAX {
            self.align_page_to_preview();
        }
        self.status = "Connected".to_string();
        self.write_state()?;
        Ok(())
    }

    fn poll_spec(&mut self) {
        let now = SystemTime::now();
        if now
            .duration_since(self.last_spec_poll)
            .unwrap_or(Duration::from_secs(1))
            < Duration::from_millis(300)
        {
            return;
        }
        self.last_spec_poll = now;
        if let Err(err) = self.reload_spec(false) {
            self.status = if self.workspace.is_some() && missing_file(&err) {
                "Preparing pipeline…".into()
            } else {
                format!("Spec error: {err}")
            };
        }
    }

    fn poll_action_ack(&mut self) {
        let now = SystemTime::now();
        if now
            .duration_since(self.last_action_ack_poll)
            .unwrap_or(Duration::from_secs(1))
            < Duration::from_millis(300)
        {
            return;
        }
        self.last_action_ack_poll = now;
        let Some(path) = self.spec.action_ack_path.clone() else {
            return;
        };
        let data = match fs::read_to_string(&path) {
            Ok(data) => data,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return,
            Err(err) => {
                self.status = format!("Action acknowledgement read failed: {err}");
                return;
            }
        };
        let ack = match serde_json::from_str::<UiActionAck>(&data) {
            Ok(ack) => ack,
            Err(err) => {
                self.status = format!("Action acknowledgement parse failed: {err}");
                return;
            }
        };
        if ack.seq > self.last_ack_seq {
            self.last_ack_seq = ack.seq;
            self.action_feedback = Some(if let Some(error) = ack.error.filter(|s| !s.is_empty()) {
                format!("Action failed: {error}")
            } else if let Some(message) = ack.message.filter(|s| !s.is_empty()) {
                message
            } else {
                format!("Action {} completed", ack.seq)
            });
        }
        let previous_len = self.actions.len();
        self.actions.retain(|action| action.seq > ack.seq);
        if self
            .last_action
            .as_ref()
            .is_some_and(|action| action.seq <= ack.seq)
        {
            self.last_action = None;
        }
        if self.actions.len() != previous_len {
            if let Err(err) = self.write_state() {
                self.status = format!("State write failed: {err}");
            }
        }
    }

    fn poll_preview(&mut self, ctx: &egui::Context) {
        let now = SystemTime::now();
        if now
            .duration_since(self.last_preview_poll)
            .unwrap_or(Duration::from_secs(1))
            < Duration::from_millis(120)
        {
            return;
        }
        self.last_preview_poll = now;
        for preview in &self.spec.previews {
            let Some(path) = &preview.metadata_path else {
                self.preview_dimensions.remove(&preview.name);
                continue;
            };
            let data = match fs::read_to_string(path) {
                Ok(data) => data,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    self.preview_dimensions.remove(&preview.name);
                    continue;
                }
                Err(err) => {
                    self.preview_dimensions.remove(&preview.name);
                    self.status = format!("{} preview dimensions unavailable: {err}", preview.name);
                    continue;
                }
            };
            match parse_preview_dimensions(&data) {
                Ok(size) => {
                    self.preview_dimensions.insert(preview.name.clone(), size);
                }
                Err(err) => {
                    self.preview_dimensions.remove(&preview.name);
                    self.status = format!("{} preview dimensions invalid: {err}", preview.name);
                }
            }
        }
        let Some(preview) = self.spec.previews.get(self.selected_preview).cloned() else {
            return;
        };
        let Ok(meta) = fs::metadata(&preview.path) else {
            self.preview_status
                .insert(preview.name, "Waiting for preview frame".to_string());
            return;
        };
        let modified = meta.modified().ok();
        if modified.is_some()
            && self
                .last_preview_modified
                .get(&preview.name)
                .copied()
                .flatten()
                == modified
        {
            return;
        }
        match load_color_image(&preview.path) {
            Ok(image) => {
                let options = egui::TextureOptions::LINEAR;
                if let Some(texture) = self.preview_textures.get_mut(&preview.name) {
                    texture.set(image, options);
                } else {
                    let texture_name = format!("hm-ui-preview-{}", preview.name);
                    self.preview_textures.insert(
                        preview.name.clone(),
                        ctx.load_texture(texture_name, image, options),
                    );
                }
                self.last_preview_modified
                    .insert(preview.name.clone(), modified);
                self.preview_status
                    .insert(preview.name, "Live preview".to_string());
            }
            Err(err) => {
                self.preview_status
                    .insert(preview.name, format!("Preview load failed: {err}"));
            }
        }
    }

    fn set_action(&mut self, kind: &str) {
        self.action_seq += 1;
        let action = UiAction {
            seq: self.action_seq,
            kind: kind.to_string(),
            windows: self.values.clone(),
        };
        self.actions.push(action.clone());
        self.last_action = Some(action);
        self.action_feedback = Some(format!("{} requested…", action_label(kind)));
        if let Err(err) = self.write_state() {
            self.status = format!("State write failed: {err}");
        }
    }

    fn reset_values(&mut self, system_defaults: bool) {
        for window in &self.spec.windows {
            let entry = self.values.entry(window.name.clone()).or_default();
            let revisions = self
                .control_revisions
                .entry(window.name.clone())
                .or_default();
            for control in &window.controls {
                let value = if system_defaults {
                    control
                        .system_default_value
                        .or(control.default_value)
                        .unwrap_or(control.value)
                } else {
                    control.default_value.unwrap_or(control.value)
                };
                entry.insert(control.name.clone(), value);
                let revision = revisions.entry(control.name.clone()).or_default();
                *revision = revision.saturating_add(1);
            }
        }
        self.set_action(if system_defaults {
            "reset-system"
        } else {
            "reset-open"
        });
    }

    fn write_state(&mut self) -> Result<()> {
        let state = UiState {
            version: 1,
            spec_version: SUPPORTED_SPEC_VERSION,
            updated_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            windows: self.values.clone(),
            control_revisions: self.control_revisions.clone(),
            selected_preview: self
                .spec
                .previews
                .get(self.selected_preview)
                .map(|preview| preview.name.clone()),
            actions: self.actions.clone(),
            last_action: self.last_action.clone(),
        };
        write_json_atomic(&self.state_path, &state)
            .with_context(|| format!("write {}", self.state_path.display()))?;
        Ok(())
    }

    fn poll_runtime(&mut self) {
        let now = SystemTime::now();
        if now
            .duration_since(self.last_runtime_poll)
            .unwrap_or_default()
            < Duration::from_millis(250)
        {
            return;
        }
        self.last_runtime_poll = now;
        let Some(path) = &self.spec.runtime_path else {
            return;
        };
        let result = fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))
            .and_then(|data| parse_runtime_status(&data));
        match result {
            Ok(runtime) => {
                self.runtime = Some(runtime);
                self.runtime_error = None;
            }
            Err(error) => {
                // Disable actions while the producer's current capabilities are unknown.
                self.runtime = None;
                self.runtime_error = if self.workspace.is_some() && missing_file(&error) {
                    self.status = "Preparing pipeline…".into();
                    None
                } else {
                    Some(format!("Runtime status unavailable: {error:#}"))
                };
            }
        }
    }

    fn align_page_to_preview(&mut self) {
        let Some(preview) = self.spec.previews.get(self.selected_preview) else {
            return;
        };
        let pages = control_pages(&self.spec);
        if pages
            .get(self.selected_page)
            .is_some_and(|page| page.0 == preview.name)
        {
            return;
        }
        self.selected_page = pages
            .iter()
            .position(|page| page.0 == preview.name)
            .unwrap_or(usize::MAX);
    }

    fn select_preview(&mut self, index: usize) {
        if index >= self.spec.previews.len() {
            return;
        }
        self.selected_preview = index;
        self.align_page_to_preview();
        self.last_preview_poll = UNIX_EPOCH;
        if let Err(err) = self.write_state() {
            self.status = format!("State write failed: {err}");
        }
    }

    fn select_page(&mut self, index: usize) {
        let pages = control_pages(&self.spec);
        self.selected_page = index;
        if let Some(page) = pages.get(index) {
            if let Some(preview) = self.spec.previews.iter().position(|p| p.name == page.0) {
                self.selected_preview = preview;
                self.last_preview_poll = UNIX_EPOCH;
                if let Err(err) = self.write_state() {
                    self.status = format!("State write failed: {err}");
                }
            }
        }
    }

    fn draw_runtime(&mut self, ui: &mut egui::Ui) {
        let Some(runtime) = self.runtime.clone() else {
            if let Some(error) = &self.runtime_error {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            } else if self.workspace.is_some() {
                ui.label("Preparing pipeline…");
            } else if self.spec.runtime_path.is_some() {
                ui.label("Waiting for runtime status…");
            }
            return;
        };
        ui.horizontal_wrapped(|ui| {
            ui.strong(display_name(&runtime.state));
            for (kind, label) in [
                ("pause", "Pause"),
                ("resume", "Resume"),
                ("step", "Step batch"),
                ("stop", "Stop"),
            ] {
                if runtime.capabilities.iter().any(|item| item == kind)
                    && ui
                        .add_enabled(runtime.permits(kind), egui::Button::new(label))
                        .clicked()
                {
                    self.set_action(kind);
                }
            }
            if runtime.capabilities.iter().any(|item| item == "rate") {
                let mut rate = runtime.rate.unwrap_or(1.0);
                let before = rate;
                egui::ComboBox::from_id_salt("runtime_speed")
                    .selected_text(if rate == 0.0 {
                        "Max".into()
                    } else {
                        format!("{rate}×")
                    })
                    .show_ui(ui, |ui| {
                        for value in [0.25, 0.5, 1.0, 2.0, 4.0, 0.0] {
                            let label = if value == 0.0 {
                                "Max".into()
                            } else {
                                format!("{value}×")
                            };
                            ui.selectable_value(&mut rate, value, label);
                        }
                    });
                if rate != before {
                    self.set_action(&format!("rate:{rate}"));
                }
            }
            if let Some(frame) = runtime.frame {
                ui.label(format!("Frame {frame}"));
                if let Some(fps) = runtime.fps {
                    ui.label(format!(
                        "{} · {fps:.2} fps",
                        playback_time(frame as f64 / fps)
                    ));
                }
            }
            let age = now_ms().saturating_sub(runtime.updated_ms) / 1000;
            if age >= 10 && matches!(runtime.state.as_str(), "running" | "paused" | "stopping") {
                ui.weak(format!("Status updated {age}s ago"));
            }
            if !runtime.message.is_empty() {
                if runtime.state == "error" {
                    ui.colored_label(egui::Color32::LIGHT_RED, &runtime.message);
                } else {
                    ui.label(&runtime.message);
                }
            }
        });
    }

    fn draw_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(if self.spec.title.is_empty() {
                "HM UI"
            } else {
                &self.spec.title
            });
            ui.separator();
            ui.label(&self.spec.subtitle);
        });
        ui.horizontal_wrapped(|ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                if ui.button("Save").clicked() {
                    self.set_action("save");
                }
                if ui
                    .button("System defaults")
                    .on_hover_text("Reset every control to the system configuration")
                    .clicked()
                {
                    self.reset_values(true);
                }
                if ui
                    .button("Open-time defaults")
                    .on_hover_text("Reset every control to its value when this UI opened")
                    .clicked()
                {
                    self.reset_values(false);
                }
            });
        });
        self.draw_runtime(ui);
    }

    fn draw_sidebar(&mut self, ui: &mut egui::Ui) {
        let pages = control_pages(&self.spec);
        ui.horizontal_wrapped(|ui| {
            let mut views = Vec::new();
            for (index, (view, _)) in pages.iter().enumerate() {
                if views.contains(view) {
                    continue;
                }
                views.push(view.clone());
                let selected = pages
                    .get(self.selected_page)
                    .is_some_and(|page| &page.0 == view);
                if ui.selectable_label(selected, view_label(view)).clicked() {
                    self.select_page(index);
                }
            }
            if ui
                .selectable_label(self.selected_page == pages.len(), "Commands")
                .clicked()
            {
                self.selected_page = pages.len();
            }
            if ui
                .selectable_label(self.selected_page == pages.len() + 1, "Status")
                .clicked()
            {
                self.selected_page = pages.len() + 1;
            }
        });
        if let Some(current) = pages.get(self.selected_page).cloned() {
            ui.horizontal_wrapped(|ui| {
                for (index, (view, group)) in pages.iter().enumerate() {
                    if view == &current.0
                        && ui
                            .selectable_label(index == self.selected_page, group)
                            .clicked()
                    {
                        self.select_page(index);
                    }
                }
            });
        }
        ui.separator();
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui, page: &(String, String)) {
        let controls: Vec<(String, ControlSpec)> = self
            .spec
            .windows
            .iter()
            .flat_map(|window| {
                window.controls.iter().filter_map(|control| {
                    let view = control_view(control);
                    let group = control_group(control, window);
                    if view == page.0 && group == page.1 {
                        Some((window.name.clone(), control.clone()))
                    } else {
                        None
                    }
                })
            })
            .collect();
        ui.heading(format!("{} · {}", view_label(&page.0), page.1));
        ui.add_space(8.0);
        let mut any_changed = false;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (window_name, control) in &controls {
                    let mut control_changed = false;
                    ui.push_id((window_name, &control.name), |ui| {
                        let max_value = control.max_value;
                        let value = self
                            .values
                            .entry(window_name.clone())
                            .or_default()
                            .entry(control.name.clone())
                            .or_insert(control.value);
                        let open_default = control.default_value.unwrap_or(control.value);
                        let mut reset = false;

                        let choices = control.choices.as_slice();
                        ui.horizontal(|ui| {
                            let label_width = (ui.available_width() - 65.0).max(80.0);
                            ui.add_sized(
                                [label_width, ui.spacing().interact_size.y],
                                egui::Label::new(
                                    egui::RichText::new(control_label(control)).strong(),
                                )
                                .wrap(),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    reset = ui
                                        .add_enabled(
                                            *value != open_default,
                                            egui::Button::new("Reset"),
                                        )
                                        .on_hover_text(format!(
                                            "Open-time default: {}",
                                            format_control_value(control, open_default)
                                        ))
                                        .clicked();
                                },
                            );
                        });

                        let mut changed = if !choices.is_empty() {
                            // Already clamped by clamp_spec_choices, so the combo
                            // and the numeric readout above cannot disagree.
                            let mut selected = *value;
                            let changed = egui::ComboBox::from_id_salt((
                                window_name.as_str(),
                                control.name.as_str(),
                            ))
                            .width(ui.available_width())
                            .selected_text(choice_label(choices, selected))
                            .show_ui(ui, |ui| {
                                let mut picked = false;
                                for (index, label) in choices.iter().enumerate() {
                                    picked |= ui
                                        .selectable_value(&mut selected, index as i32, label)
                                        .changed();
                                }
                                picked
                            })
                            .inner
                            .unwrap_or(false);
                            *value = selected;
                            changed
                        } else if max_value == 1 && control.min_value == 0 {
                            let mut checked = *value > 0;
                            let changed = ui.checkbox(&mut checked, "Enabled").changed();
                            if changed {
                                *value = if checked { 1 } else { 0 };
                            }
                            changed
                        } else {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().slider_width =
                                    (ui.available_width() - 105.0).max(40.0);
                                let slider_changed = without_value_scroll(ui, |ui| {
                                    ui.add(
                                        egui::Slider::new(value, control.min_value..=max_value)
                                            .clamping(egui::SliderClamping::Always)
                                            .show_value(false),
                                    )
                                })
                                .changed();
                                let number_changed = numeric_editor(ui, control, value).changed();
                                slider_changed || number_changed
                            })
                            .inner
                        };
                        if reset {
                            *value = open_default;
                            changed = true;
                        }
                        if !control.description.is_empty() {
                            ui.label(egui::RichText::new(&control.description).weak().small());
                        }
                        control_changed |= changed;
                        any_changed |= control_changed;
                        ui.add_space(2.0);
                        ui.separator();
                        ui.add_space(2.0);
                    });
                    if control_changed {
                        let revision = self
                            .control_revisions
                            .entry(window_name.clone())
                            .or_default()
                            .entry(control.name.clone())
                            .or_default();
                        *revision = revision.saturating_add(1);
                    }
                }
            });
        if any_changed {
            if let Err(err) = self.write_state() {
                self.status = format!("State write failed: {err}");
            }
        }
    }

    fn draw_preview(&mut self, ui: &mut egui::Ui) {
        let previews = self.spec.previews.clone();
        ui.horizontal_wrapped(|ui| {
            for (index, preview) in previews.iter().enumerate() {
                if ui
                    .selectable_label(
                        self.selected_preview == index,
                        preview_label(&preview.name, self.preview_dimensions.get(&preview.name)),
                    )
                    .clicked()
                {
                    self.select_preview(index);
                }
            }
        });
        let name = previews
            .get(self.selected_preview)
            .map(|p| p.name.clone())
            .unwrap_or_default();
        let transform = self.preview_transforms.entry(name.clone()).or_default();
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(if self.preview_maximized {
                    "Restore preview"
                } else {
                    "Maximize preview"
                })
                .clicked()
            {
                self.preview_maximized = !self.preview_maximized;
            }
            if ui.button("−").on_hover_text("Zoom out").clicked() {
                transform.zoom_at(1.0 / 1.25, egui::Vec2::ZERO);
            }
            ui.label(format!("{:.0}%", transform.zoom * 100.0));
            if ui.button("+").on_hover_text("Zoom in").clicked() {
                transform.zoom_at(1.25, egui::Vec2::ZERO);
            }
            if ui
                .button("Fit")
                .on_hover_text("Reset zoom and pan")
                .clicked()
            {
                *transform = PreviewTransform::default();
            }
        });
        ui.weak("Wheel to zoom · Drag to pan · Double-click to fit");
        if let Some(status) = self.preview_status.get(&name) {
            if status != "Live preview" {
                ui.colored_label(egui::Color32::LIGHT_RED, status);
            }
        }
        let available = ui.available_size_before_wrap().max(egui::vec2(1.0, 1.0));
        let response = ui.allocate_response(available, egui::Sense::click_and_drag());
        let rect = response.rect;
        interact_preview(ui, &response, transform);
        let painter = ui.painter().with_clip_rect(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::BLACK);
        if let Some(texture) = self.preview_textures.get(&name) {
            let size = texture.size_vec2();
            let fit = (rect.width() / size.x).min(rect.height() / size.y);
            let image_rect = egui::Rect::from_center_size(
                rect.center() + transform.pan,
                size * fit * transform.zoom,
            );
            painter.image(
                texture.id(),
                image_rect,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        } else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Waiting for preview frame",
                egui::TextStyle::Body.resolve(ui.style()),
                egui::Color32::LIGHT_GRAY,
            );
        }
    }

    fn draw_commands(&mut self, ui: &mut egui::Ui) {
        ui.heading("Commands");
        ui.add_space(8.0);
        ui.label("Common local commands");
        ui.monospace("hmtrack --game-id <game> --camera-ui=1");
        ui.monospace("hmstitch --game-id <game> --camera-ui=1");
        ui.monospace("hmlevel --game-id <game>");
        ui.label("Level vertical posts and crop a calibrated NONA panorama in the browser.");
        ui.monospace("bazelisk build //hm-ui:hm-ui");
        ui.add_space(14.0);
        ui.label("Use Games & Sources to import recordings and Runs & Logs to launch or inspect processing. Setup stores options for the next run; Runtime controls the active preview.");
    }

    fn draw_status(&mut self, ui: &mut egui::Ui) {
        ui.heading("Status");
        ui.add_space(8.0);
        ui.label(format!("Connection: {}", self.status));
        ui.label(format!("Spec: {}", self.spec_path.display()));
        ui.label(format!("State: {}", self.state_path.display()));
        ui.label(format!("Windows: {}", self.spec.windows.len()));
        ui.label(format!("Preview views: {}", self.spec.previews.len()));
        let controls: usize = self.spec.windows.iter().map(|w| w.controls.len()).sum();
        ui.label(format!("Controls: {controls}"));
    }
}

fn missing_file(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
}

impl HmUiApp {
    fn draw(&mut self, ctx: &egui::Context) {
        if let Some(workspace) = &mut self.workspace {
            if !workspace.draw_navigation(ctx) {
                workspace.draw_content(ctx);
                return;
            }
        }
        if self.preview_maximized && ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.preview_maximized = false;
        }
        egui::TopBottomPanel::top("top").show(ctx, |ui| self.draw_top_bar(ui));
        egui::TopBottomPanel::bottom("feedback").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(&self.status);
                if let Some(feedback) = &self.action_feedback {
                    ui.separator();
                    ui.label(feedback);
                }
            });
        });
        if !self.preview_maximized {
            let max_width = (ctx.screen_rect().width() * 0.55).max(280.0);
            egui::SidePanel::right("controls")
                .default_width(350.0)
                .min_width(280.0)
                .max_width(max_width)
                .resizable(true)
                .show(ctx, |ui| {
                    self.draw_sidebar(ui);
                    let pages = control_pages(&self.spec);
                    if let Some(page) = pages.get(self.selected_page) {
                        self.draw_controls(ui, page);
                    } else if self.selected_page == pages.len() {
                        egui::ScrollArea::vertical().show(ui, |ui| self.draw_commands(ui));
                    } else if self.selected_page == pages.len() + 1 {
                        egui::ScrollArea::vertical().show(ui, |ui| self.draw_status(ui));
                    } else {
                        ui.label("No controls for this view");
                    }
                });
        }
        egui::CentralPanel::default().show(ctx, |ui| self.draw_preview(ui));
    }
}

impl eframe::App for HmUiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_spec();
        self.poll_action_ack();
        self.poll_runtime();
        self.poll_preview(ctx);
        self.draw(ctx);
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}

fn view_label(name: &str) -> &str {
    if name == "Final" {
        "Program"
    } else {
        name
    }
}

fn action_label(kind: &str) -> &str {
    match kind {
        "save" => "Save",
        "reset-open" => "Open-time defaults",
        "reset-system" => "System defaults",
        "pause" => "Pause",
        "resume" => "Resume",
        "step" => "Step batch",
        "stop" => "Stop",
        _ => kind,
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn playback_time(seconds: f64) -> String {
    let milliseconds = (seconds * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        milliseconds / 3_600_000,
        milliseconds / 60_000 % 60,
        milliseconds / 1000 % 60,
        milliseconds % 1000
    )
}

fn parse_runtime_status(data: &str) -> Result<RuntimeStatus> {
    let status: RuntimeStatus = serde_json::from_str(data).context("parse runtime status")?;
    anyhow::ensure!(
        matches!(
            status.state.as_str(),
            "running" | "paused" | "stopping" | "finished" | "error"
        ),
        "unknown runtime state: {}",
        status.state
    );
    anyhow::ensure!(
        status.fps.is_none_or(|fps| fps.is_finite() && fps > 0.0),
        "runtime fps must be positive"
    );
    anyhow::ensure!(
        status
            .rate
            .is_none_or(|rate| rate.is_finite() && rate >= 0.0),
        "runtime rate must be nonnegative"
    );
    Ok(status)
}

fn validate_spec(spec: &UiSpec) -> Result<()> {
    for window in &spec.windows {
        for control in &window.controls {
            anyhow::ensure!(
                control.min_value <= control.max_value,
                "{}: minimum exceeds maximum",
                control.name
            );
            anyhow::ensure!(
                control
                    .divisor
                    .is_none_or(|divisor| divisor.is_finite() && divisor != 0.0)
                    && control.offset.is_finite(),
                "{}: invalid numeric units",
                control.name
            );
        }
    }
    Ok(())
}

fn control_label(control: &ControlSpec) -> String {
    if control.label.is_empty() {
        display_name(&control.name)
    } else {
        control.label.clone()
    }
}

fn numeric_units(control: &ControlSpec) -> (f64, f64, String) {
    if let Some(divisor) = control.divisor {
        return (divisor, control.offset, control.unit.clone());
    }
    let (divisor, offset, unit) = if control.name == "Exposure_EV_x10" {
        (10.0, -4.0, "EV")
    } else if control.name.ends_with("_x100") {
        (100.0, 0.0, "")
    } else if control.name.ends_with("_x10") {
        (10.0, 0.0, "")
    } else if control.name.contains("Kelvin") || control.name.contains("Temperature") {
        (1.0, 0.0, "K")
    } else if control.name.ends_with("_Degrees") {
        (-1.0, 90.0, "deg")
    } else {
        (1.0, 0.0, "")
    };
    (divisor, offset, unit.to_string())
}

fn format_control_value(control: &ControlSpec, value: i32) -> String {
    if !control.choices.is_empty()
        || control.divisor.is_none()
        || control.max_value == 1 && control.min_value == 0
    {
        return format_choice_value(&control.name, value, control.max_value, &control.choices);
    }
    let (divisor, offset, unit) = numeric_units(control);
    let display = value as f64 / divisor + offset;
    let decimals = divisor.abs().log10().ceil().max(0.0) as usize;
    format!("{display:.decimals$} {unit}")
        .trim_end()
        .to_string()
}

/// Value widgets must leave wheel input for their enclosing ScrollArea, even
/// when a slider has keyboard focus or a DragValue is hovered.
fn without_value_scroll<R>(ui: &mut egui::Ui, draw: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let scroll = ui.input_mut(|input| {
        let scroll = (input.raw_scroll_delta, input.smooth_scroll_delta);
        input.raw_scroll_delta = egui::Vec2::ZERO;
        input.smooth_scroll_delta = egui::Vec2::ZERO;
        scroll
    });
    let result = draw(ui);
    ui.input_mut(|input| {
        input.raw_scroll_delta = scroll.0;
        input.smooth_scroll_delta = scroll.1;
    });
    result
}

fn numeric_editor(ui: &mut egui::Ui, control: &ControlSpec, value: &mut i32) -> egui::Response {
    let (divisor, offset, unit) = numeric_units(control);
    let mut display = *value as f64 / divisor + offset;
    let first = control.min_value as f64 / divisor + offset;
    let last = control.max_value as f64 / divisor + offset;
    let response = without_value_scroll(ui, |ui| {
        ui.add(
            egui::DragValue::new(&mut display)
                .range(first.min(last)..=first.max(last))
                .speed(1.0 / divisor.abs())
                .max_decimals(4)
                .update_while_editing(false)
                .suffix(if unit.is_empty() {
                    String::new()
                } else {
                    format!(" {unit}")
                }),
        )
    })
    .on_hover_text("Click to type a value; drag to adjust. Wheel scrolls the controls.");
    if response.changed() {
        *value = ((display - offset) * divisor)
            .round()
            .clamp(control.min_value as f64, control.max_value as f64) as i32;
    }
    response
}

fn interact_preview(
    ui: &mut egui::Ui,
    response: &egui::Response,
    transform: &mut PreviewTransform,
) {
    if response.double_clicked() {
        *transform = PreviewTransform::default();
        return;
    }
    if response.hovered() {
        let scroll = ui.input_mut(|input| {
            let scroll = input.smooth_scroll_delta.y;
            input.smooth_scroll_delta = egui::Vec2::ZERO;
            scroll
        });
        if scroll != 0.0 {
            let anchor =
                response.hover_pos().unwrap_or(response.rect.center()) - response.rect.center();
            transform.zoom_at((scroll * 0.004).exp(), anchor);
        }
    }
    if response.dragged() && transform.zoom > 1.0 {
        transform.pan += response.drag_delta();
    }
    let limit = response.rect.size() * transform.zoom;
    transform.pan = transform.pan.clamp(-limit, limit);
    if transform.zoom > 1.0 {
        response.clone().on_hover_cursor(egui::CursorIcon::Grab);
    }
}

fn display_name(name: &str) -> String {
    name.replace('_', " ")
}

fn control_view(control: &ControlSpec) -> String {
    if control.view.is_empty() {
        "Final".to_string()
    } else {
        control.view.clone()
    }
}

fn control_group(control: &ControlSpec, window: &WindowSpec) -> String {
    if control.group.is_empty() {
        window.name.clone()
    } else {
        control.group.clone()
    }
}

fn control_pages(spec: &UiSpec) -> Vec<(String, String)> {
    let mut pages: Vec<(String, String)> = Vec::new();
    for window in &spec.windows {
        for control in &window.controls {
            let page = (control_view(control), control_group(control, window));
            if !pages.contains(&page) {
                pages.push(page);
            }
        }
    }
    pages.sort_by_key(|(view, _)| match view.as_str() {
        "Stitched" => 0,
        "Final" => 1,
        _ => 2,
    });
    pages
}

/// Bring every enumerated control's value and defaults into label range once,
/// so nothing downstream has to decide between the raw value and a clamped one.
/// An out-of-range index is a producer bug; correcting it here keeps the widget
/// and its numeric readout from showing two different things all session.
fn clamp_spec_choices(spec: &mut UiSpec) {
    for window in &mut spec.windows {
        for control in &mut window.controls {
            if control.choices.is_empty() {
                continue;
            }
            let limit = control.choices.len() as i32 - 1;
            control.max_value = limit;
            control.value = control.value.clamp(0, limit);
            control.default_value = control.default_value.map(|value| value.clamp(0, limit));
            control.system_default_value = control
                .system_default_value
                .map(|value| value.clamp(0, limit));
        }
    }
}

fn choice_label(choices: &[String], value: i32) -> String {
    usize::try_from(value)
        .ok()
        .and_then(|index| choices.get(index))
        .cloned()
        .unwrap_or_else(|| value.to_string())
}

fn format_choice_value(name: &str, value: i32, max_value: i32, choices: &[String]) -> String {
    if choices.is_empty() {
        format_value(name, value, max_value)
    } else {
        choice_label(choices, value)
    }
}

fn format_value(name: &str, value: i32, max_value: i32) -> String {
    if max_value == 1 {
        if value > 0 {
            "on".to_string()
        } else {
            "off".to_string()
        }
    } else if name == "Exposure_EV_x10" {
        format!("{:+.1} EV", (value - 40) as f32 / 10.0)
    } else if name.ends_with("_x100") {
        format!("{:.2}", value as f32 / 100.0)
    } else if name.ends_with("_x10") {
        format!("{:.1}", value as f32 / 10.0)
    } else if name.contains("Kelvin") || name.contains("Temperature") {
        format!("{value} K")
    } else if name.ends_with("_Degrees") {
        format!("{} deg", 90 - value)
    } else {
        value.to_string()
    }
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    let data = serde_json::to_vec_pretty(value)?;
    fs::write(&tmp, data)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn load_color_image(path: &Path) -> Result<egui::ColorImage> {
    let image = image::open(path)
        .with_context(|| format!("decode {}", path.display()))?
        .to_rgba8();
    let size = [image.width() as usize, image.height() as usize];
    Ok(egui::ColorImage::from_rgba_unmultiplied(
        size,
        image.as_raw(),
    ))
}

fn configure_ui(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(22.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        egui::FontId::new(15.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::new(15.0, egui::FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::new(14.0, egui::FontFamily::Monospace),
    );
    style.text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::new(12.0, egui::FontFamily::Proportional),
    );
    style.spacing.interact_size.y = 25.0;
    ctx.set_style(style);
}

fn main() -> Result<()> {
    let args = Args::parse();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([760.0, 520.0]),
        ..Default::default()
    };
    let title = args.title.clone();
    eframe::run_native(
        &args.title,
        native_options,
        Box::new(move |cc| {
            configure_ui(&cc.egui_ctx);
            let spec = args.spec.clone().unwrap_or_else(|| {
                args.workspace
                    .as_ref()
                    .expect("workspace or spec required")
                    .join("live/spec.json")
            });
            let state = args.state.clone().unwrap_or_else(|| {
                args.workspace
                    .as_ref()
                    .expect("workspace or state required")
                    .join("live/state.json")
            });
            let mut app = HmUiApp::new(spec, state, title.clone());
            app.workspace = args.workspace.clone().map(workspace::WorkspaceUi::new);
            Ok(Box::new(app))
        }),
    )
    .map_err(|err| anyhow::anyhow!("{err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_labels_use_source_dimensions_and_keep_stream_names_stable() {
        let dimensions = parse_preview_dimensions(r#"{"width":8192,"height":3052}"#).unwrap();
        assert_eq!(
            preview_label("Stitched", Some(&dimensions)),
            "Stitched (8192 × 3052)"
        );
        assert_eq!(preview_label("Final", None), "Program");
    }

    #[test]
    fn preview_dimensions_reject_invalid_metadata() {
        for invalid in [
            r#"{"width":0,"height":1080}"#,
            r#"{"width":1920,"height":-1}"#,
            "{}",
            "bad",
        ] {
            assert!(parse_preview_dimensions(invalid).is_err());
        }
    }

    #[test]
    fn legacy_preview_specs_do_not_require_metadata() {
        let preview: PreviewSpec =
            serde_json::from_str(r#"{"name":"Stitched","path":"preview.jpg"}"#).unwrap();
        assert!(preview.metadata_path.is_none());
    }

    #[test]
    fn legacy_control_specs_have_no_choices_or_description() {
        let control: ControlSpec =
            serde_json::from_str(r#"{"name":"Shadow_Lift_Percent","max_value":100,"value":4}"#)
                .unwrap();
        assert!(control.choices.is_empty());
        assert!(control.description.is_empty());
    }

    #[test]
    fn enumerated_controls_read_back_their_labels() {
        let choices = vec!["Laplacian".to_string(), "Alpha".to_string()];
        // An enumerated control reports its label, not the raw index.
        assert_eq!(
            format_choice_value("Seam_Blend_Mode", 1, 1, &choices),
            "Alpha"
        );
        // A value with no label is still legible rather than blank.
        assert_eq!(format_choice_value("Seam_Blend_Mode", 7, 1, &choices), "7");
        assert_eq!(
            format_choice_value("Seam_Blend_Mode", -1, 1, &choices),
            "-1"
        );
        // Without choices the existing suffix-driven formatting is unchanged, so a
        // two-entry combo does not render as the "on"/"off" a max_value of 1 means.
        assert_eq!(format_choice_value("Enable_Thing", 1, 1, &[]), "on");
        assert_eq!(
            format_choice_value("Brightness_Multiplier_x100", 125, 300, &[]),
            "1.25"
        );
    }
}

#[cfg(test)]
mod input_tests;
