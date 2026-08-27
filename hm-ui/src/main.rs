use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use eframe::egui;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(name = "hm-ui")]
#[command(about = "HockeyMON runtime operator UI")]
struct Args {
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    state: PathBuf,
    #[arg(long, default_value = "HM UI")]
    title: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct UiSpec {
    #[serde(default)]
    title: String,
    #[serde(default)]
    subtitle: String,
    #[serde(default)]
    preview_path: Option<PathBuf>,
    #[serde(default)]
    action_ack_path: Option<PathBuf>,
    #[serde(default)]
    previews: Vec<PreviewSpec>,
    #[serde(default)]
    windows: Vec<WindowSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct PreviewSpec {
    name: String,
    path: PathBuf,
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
    default_value: Option<i32>,
    #[serde(default)]
    system_default_value: Option<i32>,
    #[serde(default)]
    group: String,
    #[serde(default)]
    view: String,
    #[serde(default)]
    value_revision: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct UiState {
    version: u32,
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
}

struct HmUiApp {
    spec_path: PathBuf,
    state_path: PathBuf,
    spec: UiSpec,
    values: BTreeMap<String, BTreeMap<String, i32>>,
    control_revisions: BTreeMap<String, BTreeMap<String, u64>>,
    selected_page: usize,
    selected_preview: usize,
    last_spec_modified: Option<SystemTime>,
    last_spec_poll: SystemTime,
    last_action_ack_poll: SystemTime,
    last_preview_modified: BTreeMap<String, Option<SystemTime>>,
    last_preview_poll: SystemTime,
    preview_textures: BTreeMap<String, egui::TextureHandle>,
    preview_status: BTreeMap<String, String>,
    action_seq: u64,
    actions: Vec<UiAction>,
    last_action: Option<UiAction>,
    status: String,
}

impl HmUiApp {
    fn new(spec_path: PathBuf, state_path: PathBuf, title: String) -> Self {
        let mut app = Self {
            spec_path,
            state_path,
            spec: UiSpec {
                title,
                subtitle: "Runtime camera controls".to_string(),
                preview_path: None,
                action_ack_path: None,
                previews: Vec::new(),
                windows: Vec::new(),
            },
            values: BTreeMap::new(),
            control_revisions: BTreeMap::new(),
            selected_page: 0,
            selected_preview: 0,
            last_spec_modified: None,
            last_spec_poll: UNIX_EPOCH,
            last_action_ack_poll: UNIX_EPOCH,
            last_preview_modified: BTreeMap::new(),
            last_preview_poll: UNIX_EPOCH,
            preview_textures: BTreeMap::new(),
            preview_status: BTreeMap::new(),
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
        if spec.title.is_empty() {
            spec.title = "HM UI".to_string();
        }
        if spec.previews.is_empty() {
            if let Some(path) = spec.preview_path.clone() {
                spec.previews.push(PreviewSpec {
                    name: "Preview".to_string(),
                    path,
                });
            }
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
        let pages = control_pages(&spec);
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
        self.last_spec_modified = modified;
        self.spec = spec;
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
            self.status = format!("Spec error: {err}");
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
        self.status = "Connected".to_string();
        Ok(())
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
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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
    }

    fn draw_sidebar(&mut self, ui: &mut egui::Ui) {
        let pages = control_pages(&self.spec);
        ui.vertical(|ui| {
            ui.label("Controls");
            let mut previous_view = String::new();
            for (idx, (view, group)) in pages.iter().enumerate() {
                if view != &previous_view {
                    ui.add_space(7.0);
                    ui.label(egui::RichText::new(format!("{view} view")).strong());
                    previous_view = view.clone();
                }
                let selected = idx == self.selected_page;
                if ui.selectable_label(selected, group).clicked() {
                    self.selected_page = idx;
                }
            }
            ui.separator();
            if ui
                .selectable_label(self.selected_page == pages.len(), "Commands")
                .clicked()
            {
                self.selected_page = pages.len();
            }
            if ui
                .selectable_label(self.selected_page == pages.len() + 1, "Status")
                .on_hover_text("Runtime connection and state file paths")
                .clicked()
            {
                self.selected_page = pages.len() + 1;
            }
        });
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
        ui.heading(format!("{} · {}", page.0, page.1));
        ui.add_space(8.0);
        let mut any_changed = false;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (window_name, control) in &controls {
                    let mut control_changed = false;
                    ui.push_id((window_name, &control.name), |ui| {
                        let max_value = control.max_value.max(1);
                        let value = self
                            .values
                            .entry(window_name.clone())
                            .or_default()
                            .entry(control.name.clone())
                            .or_insert(control.value);
                        let open_default = control.default_value.unwrap_or(control.value);
                        let mut reset = false;

                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(display_name(&control.name)).strong());
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
                                            format_value(&control.name, open_default, max_value,)
                                        ))
                                        .clicked();
                                    ui.monospace(format_value(&control.name, *value, max_value));
                                },
                            );
                        });

                        let mut changed = if max_value == 1 {
                            let mut checked = *value > 0;
                            let changed = ui.checkbox(&mut checked, "Enabled").changed();
                            if changed {
                                *value = if checked { 1 } else { 0 };
                            }
                            changed
                        } else {
                            let slider_width = ui.available_width();
                            ui.scope(|ui| {
                                ui.spacing_mut().slider_width = slider_width;
                                ui.add(
                                    egui::Slider::new(value, 0..=max_value)
                                        .clamping(egui::SliderClamping::Always)
                                        .show_value(false),
                                )
                            })
                            .inner
                            .changed()
                        };
                        if reset {
                            *value = open_default;
                            changed = true;
                        }
                        control_changed = changed;
                        any_changed |= changed;
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
        if previews.len() > 1 {
            ui.horizontal(|ui| {
                ui.label("View:");
                for (idx, preview) in previews.iter().enumerate() {
                    if ui
                        .selectable_label(self.selected_preview == idx, &preview.name)
                        .clicked()
                    {
                        self.selected_preview = idx;
                        self.last_preview_poll = UNIX_EPOCH;
                        if let Err(err) = self.write_state() {
                            self.status = format!("State write failed: {err}");
                        }
                    }
                }
            });
            ui.add_space(4.0);
        }
        let available = ui.available_size_before_wrap();
        let height = (available.y * 0.58)
            .max(160.0)
            .min((available.y - 100.0).max(160.0));
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(available.x.max(1.0), height),
            egui::Sense::hover(),
        );
        ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
        if let Some(preview) = previews.get(self.selected_preview) {
            if let Some(texture) = self.preview_textures.get(&preview.name) {
                let texture_size = texture.size_vec2();
                if texture_size.x > 0.0 && texture_size.y > 0.0 {
                    let scale = (rect.width() / texture_size.x)
                        .min(rect.height() / texture_size.y)
                        .max(0.01);
                    let image_rect =
                        egui::Rect::from_center_size(rect.center(), texture_size * scale);
                    ui.painter().image(
                        texture.id(),
                        image_rect,
                        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }
            } else {
                let status = self
                    .preview_status
                    .get(&preview.name)
                    .map(String::as_str)
                    .unwrap_or("Waiting for preview frame");
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    status,
                    egui::TextStyle::Body.resolve(ui.style()),
                    egui::Color32::LIGHT_GRAY,
                );
            }
        }
        ui.add_space(10.0);
    }

    fn draw_commands(&mut self, ui: &mut egui::Ui) {
        ui.heading("Commands");
        ui.add_space(8.0);
        ui.label("Common local commands");
        ui.monospace("hmtrack --game-id <game> --camera-ui=1");
        ui.monospace("hmstitch --game-id <game> --camera-ui=1");
        ui.monospace("bazelisk build //hm-ui:hm-ui");
        ui.add_space(14.0);
        ui.label("This panel is intentionally a launcher guide for now. The tracking process remains the owner of video, detector, and stitch runtime state.");
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

impl eframe::App for HmUiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_spec();
        self.poll_action_ack();
        self.poll_preview(ctx);
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            self.draw_top_bar(ui);
        });
        egui::SidePanel::left("sidebar")
            .min_width(160.0)
            .resizable(false)
            .show(ctx, |ui| {
                self.draw_sidebar(ui);
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_preview(ui);
            let pages = control_pages(&self.spec);
            if let Some(page) = pages.get(self.selected_page) {
                self.draw_controls(ui, page);
            } else if self.selected_page == pages.len() {
                self.draw_commands(ui);
            } else {
                self.draw_status(ui);
            }
        });
        ctx.request_repaint_after(Duration::from_millis(100));
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
            Ok(Box::new(HmUiApp::new(
                args.spec.clone(),
                args.state.clone(),
                title.clone(),
            )))
        }),
    )
    .map_err(|err| anyhow::anyhow!("{err}"))
}
