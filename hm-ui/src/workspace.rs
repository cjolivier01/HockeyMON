use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use eframe::egui;
use serde_json::{json, Value};

use crate::{load_color_image, now_ms, without_value_scroll, write_json_atomic};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Runtime,
    Games,
    Run,
    Pipeline,
    Setup,
    Geometry,
    Experiments,
    Highlights,
}

struct PendingCommand {
    operation: String,
    args: Value,
    game: String,
    requested: Instant,
}

pub(crate) struct WorkspaceUi {
    directory: PathBuf,
    page: Page,
    snapshot: Value,
    snapshot_modified: Option<SystemTime>,
    last_poll: Instant,
    sequence: u64,
    pending: BTreeMap<String, PendingCommand>,
    feedback: String,
    error: Option<String>,
    current_game: String,
    root_directory: String,
    new_game: String,
    source_left: String,
    source_right: String,
    source_extra: String,
    source_role: String,
    import_copy: bool,
    lens_file: String,
    run_mode: String,
    run_duration: f64,
    seek_seconds: f64,
    selected_job: String,
    job_log: String,
    export_path: String,
    output_changes: BTreeMap<String, Value>,
    output_request: Option<String>,
    output_status: String,
    calibration: Value,
    settings: BTreeMap<String, Value>,
    settings_dirty: bool,
    settings_section: String,
    geometry: GeometryDraft,
    textures: BTreeMap<String, CachedImage>,
    experiment_name: String,
    experiment_settings: String,
    experiment_kind: String,
    experiment_start: f64,
    experiment_duration: f64,
    experiment_replay: bool,
    replay_recording: String,
    replay_sources: Value,
    replay_run: String,
    replay_source: String,
    replay_start: f64,
    replay_end: f64,
    experiment_changes: BTreeMap<String, Value>,
    selected_experiment: String,
    highlight_source: String,
    highlight_document: Value,
    selected_highlight: usize,
    highlight_dirty: bool,
    media_path: String,
    media_frame: Value,
    media_time: f64,
    media_duration: f64,
    media_fps: f64,
    media_playing: bool,
    media_last_tick: Instant,
    media_needs_frame: bool,
    selected_annotation: usize,
    place_annotation: bool,
    track_choices: Value,
    selected_track: usize,
    track_route: String,
    track_run: String,
    track_capture_target: Option<(usize, usize, Value)>,
    telemetry: Value,
    pipeline: Value,
    models: Value,
    calibration_frames: u32,
    selected_pipeline_node: String,
    leveling: Value,
    leveling_points: [Vec<[f64; 2]>; 2],
    leveling_auto_crop: bool,
    leveling_preview: Value,
    leveling_preview_state: Value,
    log_offset: u64,
    last_log_poll: Instant,
}

struct CachedImage {
    modified: Option<SystemTime>,
    texture: egui::TextureHandle,
}

#[derive(Default)]
struct GeometryDraft {
    tab: usize,
    scoreboard: Vec<[f64; 2]>,
    left: Vec<[f64; 2]>,
    right: Vec<[f64; 2]>,
    rink_image: Vec<[f64; 2]>,
    rink_observations: Vec<Value>,
    rink_history: Vec<Vec<Value>>,
    rink_selected: usize,
    rink_camera: usize,
    rink_feature: String,
    rink_layout: Value,
    rink_fit: Value,
    rink_use_level: bool,
    rink_world: Vec<[f64; 2]>,
    goals: Vec<Vec<[f64; 2]>>,
    crop: [f64; 4],
    rotation: [f64; 3],
    dirty: bool,
}

impl WorkspaceUi {
    pub(crate) fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            page: Page::Games,
            snapshot: Value::Null,
            snapshot_modified: None,
            last_poll: Instant::now() - Duration::from_secs(1),
            sequence: 0,
            pending: BTreeMap::new(),
            feedback: "Connecting to workspace…".into(),
            error: None,
            current_game: String::new(),
            root_directory: String::new(),
            new_game: String::new(),
            source_left: String::new(),
            source_right: String::new(),
            source_extra: String::new(),
            source_role: "source".into(),
            import_copy: false,
            lens_file: String::new(),
            run_mode: "track".into(),
            run_duration: 0.0,
            seek_seconds: 0.0,
            selected_job: String::new(),
            job_log: String::new(),
            export_path: String::new(),
            output_changes: BTreeMap::new(),
            output_request: None,
            output_status: String::new(),
            calibration: Value::Null,
            settings: BTreeMap::new(),
            settings_dirty: false,
            settings_section: String::new(),
            geometry: GeometryDraft::default(),
            textures: BTreeMap::new(),
            experiment_name: String::new(),
            experiment_settings: String::new(),
            experiment_kind: "stitch".into(),
            experiment_start: 0.0,
            experiment_duration: 0.0,
            experiment_replay: false,
            replay_recording: String::new(),
            replay_sources: Value::Null,
            replay_run: String::new(),
            replay_source: String::new(),
            replay_start: 0.0,
            replay_end: 0.0,
            experiment_changes: BTreeMap::new(),
            selected_experiment: String::new(),
            highlight_source: String::new(),
            highlight_document: json!({"schema": 2, "base_name": "highlights", "archive_path": "", "archive_offset_ms": 0, "intervals": []}),
            selected_highlight: 0,
            highlight_dirty: false,
            media_path: String::new(),
            media_frame: Value::Null,
            media_time: 0.0,
            media_duration: 0.0,
            media_fps: 30.0,
            media_playing: false,
            media_last_tick: Instant::now(),
            media_needs_frame: false,
            selected_annotation: 0,
            place_annotation: false,
            track_choices: Value::Null,
            selected_track: 0,
            track_route: "program".into(),
            track_run: String::new(),
            track_capture_target: None,
            telemetry: Value::Null,
            pipeline: Value::Null,
            models: Value::Null,
            calibration_frames: 200,
            selected_pipeline_node: String::new(),
            leveling: Value::Null,
            leveling_points: [Vec::new(), Vec::new()],
            leveling_auto_crop: true,
            leveling_preview: Value::Null,
            leveling_preview_state: Value::Null,
            log_offset: 0,
            last_log_poll: Instant::now(),
        }
    }

    fn command(&mut self, operation: &str, mut args: Value) {
        if matches!(
            operation,
            "calibration.update"
                | "calibration.crop"
                | "calibration.leveling"
                | "calibration.scoreboard"
                | "calibration.control-points"
                | "calibration.rink-map"
                | "calibration.lens"
                | "calibration.reset_sync"
        ) {
            if let Some(revision) = self.calibration["revision"].as_str() {
                args["expected_revision"] = json!(revision);
            }
        }
        self.sequence += 1;
        let id = format!(
            "{:020}-{}-{:012}",
            now_ms(),
            std::process::id(),
            self.sequence
        );
        let path = self.directory.join("commands").join(format!("{id}.json"));
        match write_json_atomic(
            &path,
            &json!({"id": id, "operation": operation, "args": args}),
        ) {
            Ok(()) => {
                self.pending.insert(
                    id,
                    PendingCommand {
                        operation: operation.into(),
                        args: args.clone(),
                        game: self.current_game.clone(),
                        requested: Instant::now(),
                    },
                );
                if operation != "media.frame" {
                    self.feedback = format!("{} requested…", operation_label(operation));
                    self.error = None;
                }
            }
            Err(error) => {
                self.error = Some(format!(
                    "Could not send {}: {error:#}",
                    operation_label(operation)
                ))
            }
        }
    }

    fn poll(&mut self) {
        if self.last_poll.elapsed() < Duration::from_millis(200) {
            return;
        }
        self.last_poll = Instant::now();
        let path = self.directory.join("workspace.json");
        match fs::metadata(&path) {
            Ok(metadata) => {
                let modified = metadata.modified().ok();
                if self.snapshot.is_null() || modified != self.snapshot_modified {
                    match read_json(&path) {
                        Ok(snapshot) => {
                            if self.root_directory.is_empty() {
                                self.root_directory = text(&snapshot["root"]);
                            }
                            let game = text(&snapshot["current_game"]);
                            if game != self.current_game {
                                self.current_game = game;
                                self.calibration = Value::Null;
                                self.models = Value::Null;
                                self.settings.clear();
                                self.settings_dirty = false;
                                self.geometry = GeometryDraft::default();
                                self.pipeline = Value::Null;
                                self.telemetry = Value::Null;
                                self.leveling = Value::Null;
                                self.leveling_preview = Value::Null;
                                self.leveling_preview_state = Value::Null;
                                self.selected_experiment.clear();
                                self.experiment_changes.clear();
                                self.replay_recording.clear();
                                self.replay_sources = Value::Null;
                                self.replay_run.clear();
                                self.replay_source.clear();
                                self.highlight_source.clear();
                                self.highlight_document = json!({"schema": 2, "base_name": "highlights", "archive_path": "", "archive_offset_ms": 0, "intervals": []});
                                self.highlight_dirty = false;
                                self.selected_highlight = 0;
                                self.track_choices = Value::Null;
                                self.track_capture_target = None;
                                self.track_run.clear();
                                self.media_playing = false;
                                self.media_path.clear();
                                self.media_frame = Value::Null;
                                self.selected_job.clear();
                                self.job_log.clear();
                                self.output_changes.clear();
                                self.output_request = None;
                                self.output_status.clear();
                                self.log_offset = 0;
                                if matches!(self.page, Page::Setup | Page::Geometry) {
                                    self.command("calibration.snapshot", json!({}));
                                }
                            }
                            if self.snapshot.is_null() {
                                self.feedback = "Workspace ready".into();
                            }
                            if self.error.as_deref().is_some_and(|error| {
                                error.starts_with("Workspace status unavailable:")
                            }) {
                                self.error = None;
                            }
                            self.snapshot = snapshot;
                            self.snapshot_modified = modified;
                        }
                        Err(error) => self.error = Some(error),
                    }
                }
            }
            Err(error) => self.error = Some(format!("Workspace status unavailable: {error}")),
        }
        let pending: Vec<String> = self.pending.keys().cloned().collect();
        for id in pending {
            let path = self.directory.join("results").join(format!("{id}.json"));
            match fs::read_to_string(&path) {
                Ok(data) => match serde_json::from_str::<Value>(&data) {
                    Ok(response) => {
                        let command = self.pending.remove(&id).expect("pending command exists");
                        if response["ok"].as_bool() == Some(true) {
                            if command.game == self.current_game
                                || !(command.operation.starts_with("calibration.")
                                    || command.operation.starts_with("leveling.")
                                    || command.operation.starts_with("highlights.")
                                    || command.operation.starts_with("models.")
                                    || command.operation.starts_with("outputs.")
                                    || command.operation == "pipeline.inspect"
                                    || command.operation == "experiments.telemetry"
                                    || command.operation == "experiments.replay_sources")
                            {
                                self.accept_result(
                                    &command.operation,
                                    &command.args,
                                    response["result"].clone(),
                                );
                            }
                            if !matches!(
                                command.operation.as_str(),
                                "media.frame"
                                    | "outputs.update"
                                    | "calibration.scoreboard-proposal"
                            ) {
                                self.feedback =
                                    format!("{} completed", operation_label(&command.operation));
                            }
                        } else {
                            if command.operation == "media.frame" {
                                self.media_playing = false;
                                self.media_needs_frame = false;
                            }
                            self.error = Some(format!(
                                "{} failed: {}",
                                operation_label(&command.operation),
                                text(&response["error"])
                            ));
                        }
                    }
                    Err(error) => {
                        self.error = Some(format!("Invalid workspace response for {id}: {error}"));
                    }
                },
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    self.error = Some(format!("Could not read workspace response: {error}"))
                }
            }
        }
        self.observe_output_result();
    }

    fn observe_output_result(&mut self) {
        let Some(request) = self.output_request.as_deref() else {
            return;
        };
        let result = &self.snapshot["output_routes"];
        if result["request_id"].as_str() != Some(request) {
            return;
        }
        if let Some(ok) = result["ok"].as_bool() {
            self.output_request = None;
            if ok {
                self.output_changes.clear();
                self.output_status = "Live outputs updated".into();
                self.feedback = self.output_status.clone();
            } else {
                self.output_status =
                    format!("Live output update failed: {}", text(&result["error"]));
                self.error = Some(self.output_status.clone());
            }
        }
    }

    fn accept_result(&mut self, operation: &str, args: &Value, result: Value) {
        match operation {
            "experiments.replay_sources" => {
                if args["recording_path"].as_str() == Some(self.replay_recording.as_str()) {
                    self.replay_sources = result;
                    if let Some(run) = collection(&self.replay_sources["runs"]).first() {
                        self.replay_run = text(&run["run_id"]);
                        self.replay_start = run["start_ms"].as_f64().unwrap_or(0.0) / 1000.0;
                        self.replay_end = run["end_ms"].as_f64().unwrap_or(0.0) / 1000.0;
                    }
                }
            }
            "highlights.track_choices" => {
                if args["archive_path"].as_str() == Some(self.highlight_source.as_str())
                    && args["route"].as_str() == Some(self.track_route.as_str())
                    && args["run_id"].as_str().unwrap_or_default() == self.track_run
                {
                    self.track_run = text(&result["run_id"]);
                    self.selected_track = 0;
                    self.track_choices = result;
                }
            }
            "highlights.capture_track" => {
                if let Some((interval, annotation, before)) = self.track_capture_target.take() {
                    let target = self.highlight_document["intervals"]
                        .get_mut(interval)
                        .and_then(|item| item["annotations"].get_mut(annotation));
                    if args["archive_path"].as_str() == Some(self.highlight_source.as_str())
                        && target.as_deref() == Some(&before)
                    {
                        if let (Some(target), Some(captured)) =
                            (target.and_then(Value::as_object_mut), result.as_object())
                        {
                            target.extend(captured.clone());
                            self.highlight_dirty = true;
                        }
                    } else {
                        self.error = Some("The annotation changed while tracking was captured. Review it and capture again.".into());
                    }
                }
            }
            "calibration.scoreboard-proposal" => {
                if result["revision"] != self.calibration["revision"] {
                    self.error = Some(
                        "Scoreboard suggestion is stale. Reload calibration and request it again."
                            .into(),
                    );
                } else {
                    self.geometry.scoreboard = points(&result["points"]);
                    self.geometry.dirty = true;
                    self.feedback = format!(
                        "{} Review the corners, then save the polygon.",
                        text(&result["message"])
                    );
                }
            }
            "outputs.update" => {
                if let Some(request) = result["request_id"].as_str() {
                    self.output_request = Some(request.into());
                    self.output_status =
                        "Queued for the next output frame. Resume or step if paused.".into();
                    self.feedback = self.output_status.clone();
                } else {
                    self.error =
                        Some("Live output response did not contain a request identifier".into());
                }
            }
            "models.status" => self.models = result,
            "models.prepare" => self.command("models.status", json!({})),
            "media.info" | "media.frame" => {
                if args["path"].as_str() != Some(self.media_path.as_str()) {
                    return;
                }
                self.media_duration = result["duration_ms"].as_f64().unwrap_or(0.0) / 1000.0;
                self.media_fps = result["fps"]
                    .as_f64()
                    .filter(|fps| *fps > 0.0)
                    .unwrap_or(30.0);
                if operation == "media.frame" {
                    self.media_frame = result;
                } else {
                    self.media_needs_frame = true;
                }
            }
            "calibration.rink-assist" => {
                if args["observations"] != json!(self.geometry.rink_observations) {
                    self.error = Some("Rink marks changed while suggestions were prepared. Request assistance again to preserve your edits.".into());
                } else {
                    self.geometry
                        .rink_history
                        .push(self.geometry.rink_observations.clone());
                    self.geometry.rink_observations = collection(&result["observations"]);
                    self.geometry.rink_selected =
                        self.geometry.rink_observations.len().saturating_sub(1);
                    self.feedback = text(&result["message"]);
                    self.geometry.dirty = true;
                }
            }
            "calibration.rink-map" => {
                let use_level = self.geometry.rink_use_level;
                self.geometry.rink_use_level = false;
                self.calibration = result;
                self.load_calibration_draft();
                if use_level {
                    self.geometry.tab = 4;
                    self.command("leveling.open", json!({"from_rink_map":true}));
                }
            }
            "leveling.open" => {
                self.leveling = result.clone();
                self.leveling_auto_crop = result["auto_crop"].as_bool().unwrap_or(true);
                for index in 0..3 {
                    self.geometry.rotation[index] =
                        result["rotation_degrees"][index].as_f64().unwrap_or(0.0);
                }
                for index in 0..4 {
                    self.geometry.crop[index] = result["crop"][index]
                        .as_f64()
                        .unwrap_or(if index % 2 == 0 { 0.0 } else { 1.0 });
                }
                self.leveling_points = [Vec::new(), Vec::new()];
                self.leveling_preview = Value::Null;
            }
            "leveling.estimate" => {
                for index in 0..3 {
                    self.geometry.rotation[index] =
                        result["rotation_degrees"][index].as_f64().unwrap_or(0.0);
                }
                self.leveling_preview = Value::Null;
            }
            "leveling.preview" => {
                self.leveling_preview = result;
                self.leveling_preview_state = args["state"].clone();
            }
            "leveling.save" => {
                self.leveling = Value::Null;
                self.leveling_preview = Value::Null;
                self.calibration = result;
                self.load_calibration_draft();
            }
            "leveling.close" => {
                self.leveling = Value::Null;
                self.leveling_preview = Value::Null;
            }
            "pipeline.inspect" => self.pipeline = result,
            "experiments.telemetry" => self.telemetry = result,
            "calibration.snapshot"
            | "calibration.update"
            | "calibration.crop"
            | "calibration.leveling"
            | "calibration.scoreboard"
            | "calibration.control-points"
            | "calibration.lens"
            | "calibration.reset_sync" => {
                if result.get("settings").is_some() {
                    self.calibration = result;
                    self.load_calibration_draft();
                } else if operation != "calibration.snapshot" {
                    self.command("calibration.snapshot", json!({}));
                }
            }
            "jobs.log" => {
                if args["job_id"].as_str() == Some(self.selected_job.as_str()) {
                    if args["offset"].as_u64().unwrap_or(0) == 0 {
                        self.job_log.clear();
                    }
                    self.job_log.push_str(&text(&result["text"]));
                    self.log_offset = result["offset"].as_u64().unwrap_or(0);
                    if self.job_log.len() > 512_000 {
                        let start = self.job_log.len() - 512_000;
                        let start = self
                            .job_log
                            .char_indices()
                            .find(|(index, _)| *index >= start)
                            .map(|(index, _)| index)
                            .unwrap_or(0);
                        self.job_log.drain(..start);
                    }
                }
            }
            "highlights.load" => {
                self.highlight_document = result.get("document").cloned().unwrap_or(result);
                self.highlight_source = text(&self.highlight_document["archive_path"]);
                self.highlight_dirty = false;
                self.selected_highlight = 0;
                self.track_choices = Value::Null;
                self.track_capture_target = None;
                self.track_run.clear();
            }
            "highlights.save" => self.highlight_dirty = false,
            "jobs.export" => {
                self.feedback = format!("Exported {}", text(&result));
            }
            _ => {}
        }
    }

    pub(crate) fn draw_navigation(&mut self, ctx: &egui::Context) -> bool {
        self.poll();
        egui::TopBottomPanel::top("workspace_navigation").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong("HockeyMON");
                for (page, label) in [
                    (Page::Runtime, "Runtime"),
                    (Page::Games, "Games & Sources"),
                    (Page::Run, "Runs & Logs"),
                    (Page::Pipeline, "Pipeline"),
                    (Page::Setup, "Setup"),
                    (Page::Geometry, "Calibration"),
                    (Page::Experiments, "Experiments"),
                    (Page::Highlights, "Highlights"),
                ] {
                    if ui.selectable_label(self.page == page, label).clicked() {
                        self.page = page;
                        if page == Page::Pipeline {
                            self.command("pipeline.inspect", json!({}));
                        }
                        if matches!(page, Page::Setup | Page::Geometry)
                            && self.calibration.is_null()
                        {
                            self.command("calibration.snapshot", json!({}));
                        }
                    }
                }
            });
            if !self.current_game.is_empty() {
                ui.weak(format!("Game: {}", self.current_game));
            }
        });
        self.page == Page::Runtime
    }

    pub(crate) fn draw_content(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("workspace_feedback").show(ctx, |ui| {
            for error in collection(&self.snapshot["errors"]) {
                ui.colored_label(egui::Color32::LIGHT_RED, text(&error));
            }
            if let Some(error) = &self.error {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }
            ui.horizontal_wrapped(|ui| {
                ui.label(&self.feedback);
                if !self.pending.is_empty() {
                    let oldest = self
                        .pending
                        .values()
                        .map(|pending| pending.requested.elapsed().as_secs())
                        .max()
                        .unwrap_or(0);
                    ui.spinner();
                    ui.weak(format!("{} pending · {oldest}s", self.pending.len()));
                }
                if self.error.is_some() && ui.small_button("Dismiss").clicked() {
                    self.error = None;
                }
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| match self.page {
                    Page::Games => self.draw_games(ui),
                    Page::Run => self.draw_runs(ui),
                    Page::Pipeline => self.draw_pipeline(ui),
                    Page::Setup => self.draw_setup(ui),
                    Page::Geometry => self.draw_geometry(ui),
                    Page::Experiments => self.draw_experiments(ui),
                    Page::Highlights => self.draw_highlights(ui),
                    Page::Runtime => {}
                });
        });
    }

    fn draw_games(&mut self, ui: &mut egui::Ui) {
        ui.heading("Games & Sources");
        ui.collapsing("Game library location", |ui| {
            path_field(
                ui,
                "Library folder",
                &mut self.root_directory,
                "/path/to/games",
            );
            if ui
                .add_enabled(
                    !self.root_directory.trim().is_empty(),
                    egui::Button::new("Open game library"),
                )
                .clicked()
            {
                self.command(
                    "workspace.set_root",
                    json!({"path": self.root_directory.trim()}),
                );
            }
        });
        let games = collection(&self.snapshot["games"]);
        if games.is_empty() {
            ui.label("Create a game to begin, or refresh the game catalog.");
        }
        for game in games {
            let path = object_id(&game);
            let name = object_label(&game);
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(game_id(&self.current_game) == path, &name)
                    .clicked()
                {
                    self.command("games.select", json!({"game_id": game_id(&path)}));
                }
                if name != path {
                    ui.weak(&path);
                }
            });
        }
        if ui.button("Refresh games").clicked() {
            self.command("games.list", json!({}));
        }
        ui.separator();
        ui.heading("Create game");
        path_field(
            ui,
            "Game identifier",
            &mut self.new_game,
            "e.g. dublin-2026-10-22",
        );
        if ui
            .add_enabled(
                !self.new_game.trim().is_empty(),
                egui::Button::new("Create game"),
            )
            .clicked()
        {
            self.command("games.create", json!({"game_id": self.new_game.trim()}));
        }
        ui.separator();
        ui.heading("Import camera recordings");
        ui.label("Choose the original left and right camera files for the selected game.");
        source_field(
            ui,
            "Left camera",
            &mut self.source_left,
            "/path/to/left.mp4",
        );
        source_field(
            ui,
            "Right camera",
            &mut self.source_right,
            "/path/to/right.mp4",
        );
        ui.checkbox(
            &mut self.import_copy,
            "Copy recordings into the game directory",
        );
        if ui
            .add_enabled(
                !self.current_game.is_empty()
                    && (!self.source_left.trim().is_empty()
                        || !self.source_right.trim().is_empty()),
                egui::Button::new("Import recordings"),
            )
            .clicked()
        {
            let mode = if self.import_copy { "copy" } else { "link" };
            if !self.source_left.trim().is_empty() {
                self.command(
                    "games.import_videos",
                    json!({"role": "left", "paths": self.source_left.lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>(), "mode": mode}),
                );
            }
            if !self.source_right.trim().is_empty() {
                self.command(
                    "games.import_videos",
                    json!({"role": "right", "paths": self.source_right.lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>(), "mode": mode}),
                );
            }
        }
        ui.separator();
        ui.heading("Imported camera chapters");
        ui.collapsing("Additional camera or single source", |ui| {
            path_field(ui, "Source role", &mut self.source_role, "source or cam1, cam2, …");
            source_field(ui, "Recordings", &mut self.source_extra, "/path/to/source.mp4");
            if ui.add_enabled(!self.current_game.is_empty() && !self.source_extra.trim().is_empty(), egui::Button::new("Import additional source")).clicked() {
                let mode = if self.import_copy { "copy" } else { "link" };
                self.command("games.import_videos", json!({"role": self.source_role.trim(), "paths": self.source_extra.lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>(), "mode": mode}));
            }
        });
        let videos = collection(&self.snapshot["videos"]);
        let mut roles: BTreeMap<String, usize> = BTreeMap::new();
        for video in videos {
            let role = text(&video["role"]);
            *roles.entry(role.clone()).or_default() += 1;
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("{role}: {}", object_label(&video)));
                if ui
                    .small_button("Remove from game")
                    .on_hover_text("Remove this source reference; preserve the recording file")
                    .clicked()
                {
                    self.command(
                        "games.remove_video",
                        json!({"role":role, "path":video["path"]}),
                    );
                }
            });
        }
        ui.horizontal_wrapped(|ui| {
            for (role, count) in roles {
                if count > 1 && ui.button(format!("Join {count} {role} chapters")).clicked() {
                    self.command("games.concatenate", json!({"role": role}));
                }
            }
        });
        ui.separator();
        ui.heading("Recordings");
        for recording in collection(&self.snapshot["recordings"]) {
            ui.label(object_label(&recording));
        }
    }

    fn draw_runs(&mut self, ui: &mut egui::Ui) {
        if !self.selected_job.is_empty()
            && self.last_log_poll.elapsed() > Duration::from_secs(1)
            && !self
                .pending
                .values()
                .any(|pending| pending.operation == "jobs.log")
        {
            self.last_log_poll = Instant::now();
            self.command(
                "jobs.log",
                json!({"job_id": self.selected_job, "offset": self.log_offset}),
            );
        }
        ui.heading("Runs & Logs");
        ui.horizontal_wrapped(|ui| {
            ui.label("Run:");
            for (mode, label) in [("track", "Program"), ("stitch", "Stitching calibration")] {
                ui.selectable_value(&mut self.run_mode, mode.into(), label);
            }
        });
        ui.horizontal(|ui| {
            ui.label("Duration");
            without_value_scroll(ui, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.run_duration)
                        .range(0.0..=86_400.0)
                        .suffix(" s"),
                )
            });
            ui.weak("0 runs to the end of the recording");
        });
        if ui
            .add_enabled(
                !self.current_game.is_empty(),
                egui::Button::new("Start run"),
            )
            .clicked()
        {
            let arguments: Vec<String> = if self.run_duration > 0.0 {
                vec![format!("-t={}", self.run_duration)]
            } else {
                vec![]
            };
            self.command(
                "jobs.start",
                json!({"kind": self.run_mode, "arguments": arguments}),
            );
        }
        ui.label("Use Runtime for the live preview and camera controls. Stopping drains work already queued by the pipeline.");
        self.draw_live_outputs(ui);
        ui.separator();
        let jobs = collection(&self.snapshot["jobs"]);
        for job in &jobs {
            let id = object_id(job);
            ui.horizontal_wrapped(|ui| {
                if ui
                    .selectable_label(self.selected_job == id, object_label(job))
                    .clicked()
                {
                    self.selected_job = id.clone();
                    self.log_offset = 0;
                    self.job_log.clear();
                    self.command("jobs.log", json!({"job_id": id, "offset": 0}));
                }
                ui.label(text(&job["status"]));
                if let Some(code) = job.get("returncode").filter(|value| !value.is_null()) {
                    ui.weak(format!("Exit {code}"));
                }
                if job["status"] == "completed" {
                    if let Some(path) = job["output"].as_str() {
                        if ui.small_button("Inspect output").clicked() {
                            self.open_media(path);
                            self.page = Page::Highlights;
                        }
                    }
                }
                if matches!(job["kind"].as_str(), Some("track" | "stitch"))
                    && job["status"] != "stopping"
                    && ui.small_button("Restart").clicked()
                {
                    self.command("jobs.restart", json!({"job_id":id}));
                }
                if matches!(
                    job["status"].as_str(),
                    Some("running" | "paused" | "starting")
                ) {
                    if ui.small_button("Stop").clicked() {
                        self.command("jobs.stop", json!({"job_id": id}));
                    }
                    if job["kind"] == "track" {
                        without_value_scroll(ui, |ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.seek_seconds)
                                    .range(0.0..=604800.0)
                                    .suffix(" s"),
                            )
                        });
                        if ui
                            .small_button("Seek source")
                            .on_hover_text(
                                "Finish the current recording, then restart at this source time",
                            )
                            .clicked()
                        {
                            self.command(
                                "jobs.seek",
                                json!({"job_id": id, "seconds": self.seek_seconds}),
                            );
                        }
                    }
                }
            });
        }
        if jobs.is_empty() {
            ui.label("No runs yet.");
        }
        if !self.selected_job.is_empty() {
            ui.separator();
            ui.horizontal(|ui| {
                ui.heading("Run log");
                if ui.button("Refresh log").clicked() {
                    self.command("jobs.log", json!({"job_id": self.selected_job}));
                }
            });
            egui::ScrollArea::both()
                .id_salt("job_log")
                .max_height(320.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.job_log)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY)
                            .interactive(false),
                    );
                });
            path_field(
                ui,
                "Export command",
                &mut self.export_path,
                "/path/to/run.sh",
            );
            if ui
                .add_enabled(
                    !self.export_path.is_empty(),
                    egui::Button::new("Export run command"),
                )
                .clicked()
            {
                self.command(
                    "jobs.export",
                    json!({"job_id": self.selected_job, "path": self.export_path}),
                );
            }
        }
        ui.separator();
        ui.heading("Archives");
        for archive in collection(&self.snapshot["archives"]) {
            ui.horizontal_wrapped(|ui| {
                ui.label(object_label(&archive));
                if ui.small_button("Edit highlights").clicked() {
                    self.select_highlight_archive(object_id(&archive));
                    self.page = Page::Highlights;
                }
            });
        }
    }

    fn draw_live_outputs(&mut self, ui: &mut egui::Ui) {
        let routes = self.snapshot["output_routes"].clone();
        let recordings = self.snapshot["recording_outputs"].clone();
        let active = collection(&self.snapshot["jobs"]).iter().any(|job| {
            matches!(
                job["status"].as_str(),
                Some("running" | "paused" | "starting")
            )
        });
        let network = routes["capabilities"]
            .as_array()
            .is_some_and(|items| items.iter().any(|item| item == "network-routes"))
            && routes["state"] != "finalized"
            && active;
        let archives = recordings["capabilities"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let recording = recordings["state"] == "running" && !archives.is_empty();
        let waiting = self.output_request.is_some()
            || self
                .pending
                .values()
                .any(|command| command.operation == "outputs.update");
        ui.collapsing("Live publishing & recording", |ui| {
            if !network && !recording {
                ui.weak("Start a run to change its supported outputs here.");
                return;
            }
            ui.add_enabled_ui(!waiting, |ui| {
                if recording {
                    for (key, label) in [("archive_program", "Record Program"), ("archive_stitched", "Record Stitched"), ("archive_program_4k", "Record Program 4K")] {
                        if archives.iter().any(|item| item == key) {
                            let mut enabled = self.output_changes.get(key).unwrap_or(&recordings["active"][key]).as_bool().unwrap_or(false);
                            if ui.checkbox(&mut enabled, label).changed() { self.output_changes.insert(key.into(), json!(enabled)); }
                        }
                    }
                    ui.weak("Stopping a recording finalizes its file. Starting again creates a new segment.");
                }
                if network {
                    for (key, label) in [("show_youtube", "Publish Program over RTMP"), ("spare_rtmp_enabled", "Publish spare RTMP stream"), ("rtsp_enabled", "Serve local RTSP")] {
                        let mut enabled = self.output_changes.get(key).unwrap_or(&routes["config"][key]).as_bool().unwrap_or(false);
                        if ui.checkbox(&mut enabled, label).changed() { self.output_changes.insert(key.into(), json!(enabled)); }
                    }
                    ui.weak("Saved publish addresses and stream keys remain unchanged until edited.");
                    for (key, label, secret) in [("youtube_stream_url", "RTMP ingest URL", true), ("youtube_stream_key", "RTMP stream key", true), ("spare_rtmp_url", "Spare RTMP URL", true), ("rtsp_host", "RTSP listen address", false)] {
                        let mut value = self.output_changes.get(key).or_else(|| routes["config"].get(key)).and_then(Value::as_str).unwrap_or("").to_owned();
                        ui.horizontal(|ui| {
                            ui.label(label);
                            if ui.add(egui::TextEdit::singleline(&mut value).password(secret).hint_text(if secret { "Unchanged" } else { "127.0.0.1" })).changed() {
                                self.output_changes.insert(key.into(), json!(value));
                            }
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.label("RTSP port");
                        let mut port = self.output_changes.get("rtsp_port").unwrap_or(&routes["config"]["rtsp_port"]).as_u64().unwrap_or(8554);
                        if without_value_scroll(ui, |ui| ui.add(egui::DragValue::new(&mut port).range(1..=65535))).changed() { self.output_changes.insert("rtsp_port".into(), json!(port)); }
                    });
                    let mut mounts = self.output_changes.get("rtsp_mounts").or_else(|| routes["config"].get("rtsp_mounts")).and_then(Value::as_array).cloned().unwrap_or_else(|| vec![
                        json!({"mount": routes["config"]["rtsp_program_mount"].as_str().unwrap_or("/program"), "source":"program"}),
                        json!({"mount": routes["config"]["rtsp_stitched_mount"].as_str().unwrap_or("/stitched"), "source":"stitched"})]);
                    let mut changed = false;
                    let mut remove = None;
                    ui.label("RTSP mounts");
                    let mount_count = mounts.len();
                    for (index, mount) in mounts.iter_mut().enumerate() {
                        ui.push_id(("rtsp-mount", index), |ui| ui.horizontal(|ui| {
                            let mut path = text(&mount["mount"]);
                            if ui.text_edit_singleline(&mut path).changed() { mount["mount"] = json!(path); changed = true; }
                            let mut source = mount["source"].as_str().unwrap_or("program").to_owned();
                            egui::ComboBox::from_id_salt("source").selected_text(human_label(&source)).show_ui(ui, |ui| {
                                for option in ["program", "stitched"] { if ui.selectable_value(&mut source, option.into(), human_label(option)).changed() { mount["source"] = json!(source); changed = true; } }
                            });
                            if ui.add_enabled(mount_count > 1, egui::Button::new("Remove")).clicked() { remove = Some(index); }
                        }));
                    }
                    if let Some(index) = remove { mounts.remove(index); changed = true; }
                    if ui.add_enabled(mounts.len() < 16, egui::Button::new("Add RTSP mount")).clicked() { mounts.push(json!({"mount": format!("/stream{}", mounts.len() + 1), "source":"program"})); changed = true; }
                    if changed { self.output_changes.insert("rtsp_mounts".into(), json!(mounts)); }
                }
                ui.horizontal(|ui| {
                    if ui.add_enabled(!self.output_changes.is_empty(), egui::Button::new("Apply live outputs")).clicked() {
                        self.command("outputs.update", json!({"changes": self.output_changes}));
                    }
                    if ui.button("Discard output edits").clicked() { self.output_changes.clear(); }
                });
            });
            if !self.output_status.is_empty() { ui.label(&self.output_status); }
            for segment in collection(&recordings["segments"]) {
                ui.weak(format!("{} · {} · {}", text(&segment["role"]), text(&segment["state"]), text(&segment["path"])));
            }
        });
    }

    fn draw_pipeline(&mut self, ui: &mut egui::Ui) {
        ui.heading("Pipeline inspector · next run");
        ui.weak("This graph reflects saved launch settings for the next run. The active process keeps the graph it started with.");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Refresh pipeline").clicked() {
                self.command("pipeline.inspect", json!({}));
            }
            for key in ["mode", "queue_size", "graph", "device"] {
                if !self.pipeline[key].is_null() {
                    ui.label(format!(
                        "{}: {}",
                        human_label(key),
                        text(&self.pipeline[key])
                    ));
                }
            }
        });
        let nodes = collection(&self.pipeline["nodes"]);
        if nodes.is_empty() {
            ui.label("Load a game to inspect its configured processing graph.");
            return;
        }
        ui.weak("Select a stage to inspect its inputs, implementation and configured parameters. Disabled stages preserve graph connections.");
        let mut levels: BTreeMap<String, usize> =
            nodes.iter().map(|node| (text(&node["name"]), 0)).collect();
        for _ in 0..nodes.len() {
            let previous = levels.clone();
            for node in &nodes {
                let level = collection(&node["depends"])
                    .iter()
                    .map(|name| previous.get(&text(name)).copied().unwrap_or(0) + 1)
                    .max()
                    .unwrap_or(0);
                levels.insert(text(&node["name"]), level.min(nodes.len()));
            }
            if previous == levels {
                break;
            }
        }
        let mut counts = BTreeMap::<usize, usize>::new();
        let mut layout = BTreeMap::new();
        for node in &nodes {
            let name = text(&node["name"]);
            let level = levels[&name];
            let row = counts.entry(level).or_default();
            layout.insert(name, egui::vec2(level as f32 * 245.0, *row as f32 * 95.0));
            *row += 1;
        }
        let width = (levels.values().max().copied().unwrap_or(0) + 1) as f32 * 245.0;
        let height = counts.values().max().copied().unwrap_or(1) as f32 * 95.0;
        egui::ScrollArea::both()
            .id_salt("pipeline-graph")
            .max_height(420.0)
            .show(ui, |ui| {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
                let painter = ui.painter().with_clip_rect(ui.clip_rect());
                for node in &nodes {
                    let name = text(&node["name"]);
                    let position = rect.min + layout[&name];
                    for dependency in collection(&node["depends"]) {
                        if let Some(source) = layout.get(&text(&dependency)) {
                            let source = rect.min + *source + egui::vec2(210.0, 32.0);
                            painter.arrow(
                                source,
                                position + egui::vec2(0.0, 32.0) - source,
                                egui::Stroke::new(1.5_f32, egui::Color32::GRAY),
                            );
                        }
                    }
                    let node_rect = egui::Rect::from_min_size(position, egui::vec2(210.0, 65.0));
                    let enabled = node["enabled"].as_bool().unwrap_or(true);
                    let color = if name == self.selected_pipeline_node {
                        egui::Color32::from_rgb(40, 90, 145)
                    } else if enabled {
                        egui::Color32::from_rgb(35, 65, 55)
                    } else {
                        egui::Color32::from_gray(45)
                    };
                    painter.rect_filled(node_rect, 5.0, color);
                    painter.text(
                        node_rect.center_top() + egui::vec2(0.0, 10.0),
                        egui::Align2::CENTER_TOP,
                        &name,
                        egui::FontId::proportional(14.0),
                        egui::Color32::WHITE,
                    );
                    let class = node["class"]
                        .as_str()
                        .unwrap_or("stage")
                        .rsplit('.')
                        .next()
                        .unwrap_or("stage");
                    painter.text(
                        node_rect.center_bottom() - egui::vec2(0.0, 10.0),
                        egui::Align2::CENTER_BOTTOM,
                        if enabled { class } else { "Disabled" },
                        egui::FontId::proportional(11.0),
                        egui::Color32::GRAY,
                    );
                    if ui
                        .interact(
                            node_rect,
                            ui.make_persistent_id(("pipeline-stage", &name)),
                            egui::Sense::click(),
                        )
                        .clicked()
                    {
                        self.selected_pipeline_node = name;
                    }
                }
            });
        if let Some(node) = nodes
            .iter()
            .find(|node| node["name"].as_str() == Some(self.selected_pipeline_node.as_str()))
        {
            ui.separator();
            ui.heading(&self.selected_pipeline_node);
            for key in ["class", "enabled", "depends", "input_keys", "output_keys"] {
                if !node[key].is_null() {
                    ui.label(format!("{}: {}", human_label(key), text(&node[key])));
                }
            }
            if let Some(params) = node["params"].as_object() {
                egui::Grid::new("pipeline-parameters")
                    .striped(true)
                    .show(ui, |ui| {
                        for (name, value) in params {
                            ui.monospace(name);
                            ui.label(text(value));
                            ui.end_row();
                        }
                    });
            }
        }
    }

    fn load_calibration_draft(&mut self) {
        if let Some(selection) = self.calibration["rink_mask_frame_time"]
            .get("selection")
            .cloned()
        {
            self.calibration["settings"]["stitching.rink_mask_frame_time"] = selection;
        }
        self.settings = self.calibration["settings"]
            .as_object()
            .map(|items| {
                items
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            })
            .unwrap_or_default();
        self.settings_dirty = false;
        self.geometry.scoreboard = points(&self.calibration["scoreboard"]["points"]);
        self.geometry.left.clear();
        self.geometry.right.clear();
        for pair in collection(&self.calibration["control_points"]["pairs"]) {
            if let Some(values) = pair.as_array().filter(|values| values.len() == 4) {
                self.geometry.left.push([
                    values[0].as_f64().unwrap_or(0.0),
                    values[1].as_f64().unwrap_or(0.0),
                ]);
                self.geometry.right.push([
                    values[2].as_f64().unwrap_or(0.0),
                    values[3].as_f64().unwrap_or(0.0),
                ]);
            }
        }
        self.geometry.rink_observations = collection(&self.calibration["rink_map"]["observations"]);
        self.geometry.rink_history.clear();
        self.geometry.rink_selected = 0;
        self.geometry.rink_layout = self.calibration["rink_map"]["layout"].clone();
        self.geometry.rink_fit = self.calibration["rink_map"]["fit"].clone();
        let landmarks = collection(&self.calibration["rink_map"]["landmarks"]);
        self.geometry.rink_image = landmarks
            .iter()
            .map(|value| point(&value["image"]))
            .collect();
        self.geometry.rink_world = landmarks
            .iter()
            .map(|value| point(&value["rink"]))
            .collect();
        self.geometry.goals = collection(&self.calibration["rink_map"]["goals"])
            .iter()
            .map(points)
            .collect();
        let crop = &self.calibration["settings"]["stitching.projection_framing.crop"];
        self.geometry.crop = [
            crop[0].as_f64().unwrap_or(0.0),
            crop[1].as_f64().unwrap_or(1.0),
            crop[2].as_f64().unwrap_or(0.0),
            crop[3].as_f64().unwrap_or(1.0),
        ];
        let rotation =
            &self.calibration["settings"]["stitching.projection_framing.rotation_degrees"];
        self.geometry.rotation = [
            rotation[0].as_f64().unwrap_or(0.0),
            rotation[1].as_f64().unwrap_or(0.0),
            rotation[2].as_f64().unwrap_or(0.0),
        ];
        self.geometry.dirty = false;
    }

    fn draw_setup(&mut self, ui: &mut egui::Ui) {
        ui.heading("Setup");
        if let Some(effective) = self.calibration["rink_mask_frame_time"]["effective"].as_str() {
            ui.weak(format!(
                "Mask frame: {effective} ({})",
                text(&self.calibration["rink_mask_frame_time"]["source"])
            ));
        }
        ui.collapsing("Model readiness and preparation", |ui| {
            if ui.button("Check model readiness").clicked() {
                self.command("models.status", json!({}));
            }
            ui.horizontal(|ui| {
                ui.label("Calibration source frames");
                without_value_scroll(ui, |ui| ui.add(egui::DragValue::new(&mut self.calibration_frames).range(1..=10000)));
            });
            ui.weak("Detector and pose INT8 calibration use this game's recordings. ReID preparation downloads the published NVIDIA model. Progress appears in Runs & Logs.");
            for (target, label) in [("detector", "Detector INT8"), ("pose", "Pose INT8"), ("reid", "Player ReID")] {
                ui.horizontal_wrapped(|ui| {
                    ui.label(label);
                    let status = collection(&self.models["models"]).into_iter().find(|status| status["target"] == target);
                    if let Some(status) = status {
                        ui.label(format!("{}: {}", text(&status["state"]), text(&status["message"])));
                        if status["runtime_available"] == false { ui.colored_label(egui::Color32::LIGHT_RED, "ONNX Runtime is unavailable"); }
                    }
                    if ui.add_enabled(!self.current_game.is_empty() && !self.settings_dirty, egui::Button::new(format!("Prepare {label}"))).clicked() {
                        self.command("models.prepare", json!({"target":target, "frames":self.calibration_frames}));
                    }
                });
            }
            if self.settings_dirty { ui.weak("Apply model settings before preparing."); }
        });
        if self.calibration.is_null() {
            ui.label("Select a game and load its configuration.");
            if ui.button("Load configuration").clicked() {
                self.command("calibration.snapshot", json!({}));
            }
            return;
        }
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(self.settings_dirty, egui::Button::new("Apply settings"))
                .clicked()
            {
                let changes: BTreeMap<_, _> = self
                    .settings
                    .iter()
                    .filter(|(key, value)| self.calibration["settings"][key.as_str()] != **value)
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                self.command("calibration.update", json!({"changes": changes}));
            }
            if ui.button("Reload saved settings").clicked() {
                self.command("calibration.snapshot", json!({}));
            }
            if self.settings_dirty {
                ui.weak("Unsaved changes");
            }
        });
        ui.collapsing("Synchronization and measured lens calibration", |ui| {
            ui.label("Manual frame offsets are in the stitching settings below. Reset them to request automatic audio synchronization on the next run.");
            if ui.button("Reset synchronization").clicked() { self.command("calibration.reset_sync", json!({})); }
            path_field(ui, "Lens calibration file", &mut self.lens_file, "/path/to/calibration.json");
            ui.weak("Import a measured KB4 calibration containing left_uniforms and right_uniforms. Requires AKAZE and an OpenCV mapping backend.");
            if ui.add_enabled(!self.lens_file.trim().is_empty(), egui::Button::new("Import measured lens calibration")).clicked() {
                match read_json(Path::new(self.lens_file.trim())) {
                    Ok(calibration) => self.command("calibration.lens", json!({"calibration": calibration})),
                    Err(error) => self.error = Some(error),
                }
            }
        });
        ui.collapsing("Rebuild stitching calibration", |ui| {
            ui.label("Remove generated calibration products so the next run rebuilds them. Original recordings and output archives are preserved.");
            if ui.button("Rebuild calibration on next run").clicked() {
                self.command("games.clean_stitching", json!({}));
            }
        });
        let catalog = collection(&self.calibration["catalog"]);
        let mut sections: Vec<String> = catalog
            .iter()
            .filter_map(|item| item["key"].as_str())
            .map(|key| key.split('.').next().unwrap_or(key).into())
            .collect();
        sections.sort();
        sections.dedup();
        if self.settings_section.is_empty() {
            self.settings_section = sections.first().cloned().unwrap_or_default();
        }
        ui.horizontal_wrapped(|ui| {
            for section in &sections {
                ui.selectable_value(
                    &mut self.settings_section,
                    section.clone(),
                    match section.as_str() {
                        "aspen" => "Pipeline".into(),
                        "detector" => "Detection".into(),
                        "game" => "Game".into(),
                        "hm_ui" => "Run".into(),
                        "pipeline" => "Player analysis".into(),
                        "rink" => "Camera & rink".into(),
                        "stitching" => "Stitching".into(),
                        "team" => "Teams".into(),
                        "video_out" => "Outputs".into(),
                        _ => human_label(section),
                    },
                );
            }
        });
        ui.separator();
        for item in catalog {
            let Some(key) = item["key"].as_str() else {
                continue;
            };
            if key.split('.').next() != Some(self.settings_section.as_str()) {
                continue;
            }
            let value = self.settings.entry(key.into()).or_insert(Value::Null);
            ui.push_id(key, |ui| {
                ui.strong(item["label"].as_str().unwrap_or(key));
                self.settings_dirty |= setting_editor(ui, &item, value);
                if let Some(description) =
                    item["description"].as_str().filter(|text| !text.is_empty())
                {
                    ui.weak(description);
                }
                ui.add_space(4.0);
                ui.separator();
            });
        }
    }

    fn geometry_image(
        &mut self,
        ui: &mut egui::Ui,
        descriptor: &Value,
        points: &mut Vec<[f64; 2]>,
        maximum: Option<usize>,
        id: &str,
    ) -> bool {
        let path = descriptor["image"]
            .as_str()
            .or_else(|| descriptor["path"].as_str())
            .unwrap_or_default();
        if path.is_empty() {
            ui.colored_label(
                egui::Color32::YELLOW,
                descriptor["unavailable"]
                    .as_str()
                    .unwrap_or("Image unavailable. Run stitching calibration to create the image."),
            );
            return false;
        }
        let image_path = Path::new(path);
        let modified = fs::metadata(image_path)
            .and_then(|meta| meta.modified())
            .ok();
        if !self.textures.contains_key(path)
            || self
                .textures
                .get(path)
                .is_some_and(|cached| cached.modified != modified)
        {
            match load_color_image(image_path) {
                Ok(image) => {
                    let texture = ui.ctx().load_texture(
                        format!("workspace-{path}"),
                        image,
                        egui::TextureOptions::LINEAR,
                    );
                    self.textures
                        .insert(path.into(), CachedImage { modified, texture });
                }
                Err(error) => {
                    ui.colored_label(
                        egui::Color32::LIGHT_RED,
                        format!("Image unavailable: {error:#}"),
                    );
                    return false;
                }
            }
        }
        let image = &self.textures[path].texture;
        let width = descriptor["width"]
            .as_f64()
            .filter(|width| *width > 0.0)
            .unwrap_or(image.size()[0] as f64);
        let height = descriptor["height"]
            .as_f64()
            .filter(|height| *height > 0.0)
            .unwrap_or(image.size()[1] as f64);
        let overlays = collection(&descriptor["overlays"]);
        ui.data_mut(|data| data.insert_temp(ui.id().with((id, "overlays")), overlays));
        point_canvas(ui, image, [width, height], points, maximum, id)
    }

    fn draw_geometry(&mut self, ui: &mut egui::Ui) {
        ui.heading("Calibration");
        ui.horizontal_wrapped(|ui| {
            for (index, label) in [
                "Scoreboard",
                "Control points",
                "Rink map",
                "Framing",
                "Leveling",
            ]
            .iter()
            .enumerate()
            {
                ui.selectable_value(&mut self.geometry.tab, index, *label);
            }
            if ui.button("Reload saved calibration").clicked() {
                self.command("calibration.snapshot", json!({}));
            }
        });
        if self.calibration.is_null() {
            ui.label("Loading calibration for the selected game…");
            return;
        }
        ui.separator();
        match self.geometry.tab {
            0 => {
                ui.label("Place four scoreboard corners in order. Drag a corner to refine it; wheel zooms the image.");
                let descriptor = self.calibration["scoreboard"].clone();
                let mut draft = std::mem::take(&mut self.geometry.scoreboard);
                self.geometry.dirty |=
                    self.geometry_image(ui, &descriptor, &mut draft, Some(4), "scoreboard");
                self.geometry.scoreboard = draft;
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Suggest scoreboard").clicked() {
                        self.command("calibration.scoreboard-proposal", json!({}));
                    }
                    if ui.button("Remove last corner").clicked() {
                        self.geometry.scoreboard.pop();
                        self.geometry.dirty = true;
                    }
                    if ui.button("Clear corners").clicked() {
                        self.geometry.scoreboard.clear();
                        self.geometry.dirty = true;
                    }
                    if ui
                        .add_enabled(
                            self.geometry.scoreboard.len() == 4,
                            egui::Button::new("Save scoreboard polygon"),
                        )
                        .clicked()
                    {
                        self.command(
                            "calibration.scoreboard",
                            json!({"points": self.geometry.scoreboard}),
                        );
                    }
                    if ui.button("Disable scoreboard").clicked() {
                        self.command("calibration.scoreboard", json!({"points": []}));
                    }
                });
                point_table(ui, "scoreboard-points", &mut self.geometry.scoreboard);
            }
            1 => {
                ui.label(
                    "Match the same landmark in both cameras. Point numbers identify each pair.",
                );
                let descriptors = collection(&self.calibration["control_points"]["images"]);
                let mut left = std::mem::take(&mut self.geometry.left);
                let mut right = std::mem::take(&mut self.geometry.right);
                ui.columns(2, |columns| {
                    columns[0].strong("Left camera");
                    self.geometry_image(
                        &mut columns[0],
                        descriptors.first().unwrap_or(&Value::Null),
                        &mut left,
                        None,
                        "left-control-points",
                    );
                    columns[1].strong("Right camera");
                    self.geometry_image(
                        &mut columns[1],
                        descriptors.get(1).unwrap_or(&Value::Null),
                        &mut right,
                        None,
                        "right-control-points",
                    );
                });
                self.geometry.left = left;
                self.geometry.right = right;
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Remove last pair").clicked() {
                        self.geometry.left.pop();
                        self.geometry.right.pop();
                    }
                    let paired = self.geometry.left.len() == self.geometry.right.len();
                    ui.label(format!(
                        "{} left / {} right points",
                        self.geometry.left.len(),
                        self.geometry.right.len()
                    ));
                    if ui
                        .add_enabled(
                            paired && self.geometry.left.len() >= 4,
                            egui::Button::new("Save control points"),
                        )
                        .clicked()
                    {
                        let pairs: Vec<[f64; 4]> = self
                            .geometry
                            .left
                            .iter()
                            .zip(&self.geometry.right)
                            .map(|(left, right)| [left[0], left[1], right[0], right[1]])
                            .collect();
                        self.command("calibration.control-points", json!({"pairs": pairs}));
                    }
                });
                ui.columns(2, |columns| {
                    point_table(&mut columns[0], "left-point-table", &mut self.geometry.left);
                    point_table(
                        &mut columns[1],
                        "right-point-table",
                        &mut self.geometry.right,
                    );
                });
            }
            2 => self.draw_rink_map(ui),
            3 => {
                ui.heading("Panorama framing");
                ui.label(
                    "Set the calibrated panorama crop and leveling rotation for the next run.",
                );
                ui.horizontal_wrapped(|ui| {
                    for (index, label) in ["Left", "Right", "Top", "Bottom"].iter().enumerate() {
                        ui.label(*label);
                        without_value_scroll(ui, |ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.geometry.crop[index])
                                    .range(0.0..=1.0)
                                    .speed(0.001)
                                    .max_decimals(4),
                            )
                        });
                    }
                });
                if ui.button("Apply crop").clicked() {
                    self.command(
                        "calibration.crop",
                        json!({"crop": self.geometry.crop, "auto_crop": false}),
                    );
                }
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    for (index, label) in ["Yaw", "Pitch", "Roll"].iter().enumerate() {
                        ui.label(*label);
                        without_value_scroll(ui, |ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.geometry.rotation[index])
                                    .range(-180.0..=180.0)
                                    .speed(0.1)
                                    .suffix("°"),
                            )
                        });
                    }
                });
                if ui.button("Apply leveling").clicked() {
                    self.command(
                        "calibration.leveling",
                        json!({"rotation_degrees": self.geometry.rotation}),
                    );
                }
            }
            _ => self.draw_leveling(ui),
        }
    }

    fn draw_rink_map(&mut self, ui: &mut egui::Ui) {
        ui.label("Identify painted lines and board intersections at ice level in the calibrated source cameras. Near boards are closest to the camera. Lines need two or more samples; their endpoints are not assumed to touch boards.");
        ui.collapsing("Rink dimensions (metres)", |ui| {
            for (key, label) in [
                ("length", "Length"),
                ("width", "Width"),
                ("blue_line_distance", "Centre to blue line"),
                ("goal_inset", "Goal inset"),
                ("corner_radius", "Corner radius"),
            ] {
                ui.horizontal(|ui| {
                    ui.label(label);
                    let mut value = self.geometry.rink_layout[key].as_f64().unwrap_or(0.0);
                    if ui
                        .add(egui::DragValue::new(&mut value).speed(0.01))
                        .changed()
                    {
                        self.geometry.rink_layout[key] = json!(value);
                        self.geometry.dirty = true;
                    }
                });
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui.button("Find landmarks").clicked() {
                self.command(
                    "calibration.rink-assist",
                    json!({"observations": self.geometry.rink_observations, "detect":true}),
                );
            }
            if ui.button("Use rink mask").clicked() {
                self.command(
                    "calibration.rink-assist",
                    json!({"observations": self.geometry.rink_observations, "use_mask":true}),
                );
            }
            if ui
                .add_enabled(
                    !self.geometry.rink_history.is_empty(),
                    egui::Button::new("Undo"),
                )
                .clicked()
            {
                self.geometry.rink_observations =
                    self.geometry.rink_history.pop().unwrap_or_default();
                self.geometry.rink_selected =
                    self.geometry.rink_observations.len().saturating_sub(1);
                self.geometry.dirty = true;
            }
        });
        let features = collection(&self.calibration["rink_map"]["features"]);
        ui.horizontal_wrapped(|ui| {
            ui.label("New mark source:");
            ui.selectable_value(&mut self.geometry.rink_camera, 0, "Camera 1");
            ui.selectable_value(&mut self.geometry.rink_camera, 1, "Camera 2");
            egui::ComboBox::from_id_salt("new-rink-feature")
                .selected_text(if self.geometry.rink_feature.is_empty() { "Choose landmark".into() } else { human_label(&self.geometry.rink_feature) })
                .show_ui(ui, |ui| {
                    for feature in &features {
                        let id = text(&feature["id"]);
                        ui.selectable_value(&mut self.geometry.rink_feature, id.clone(), human_label(&id));
                    }
                });
            if ui.add_enabled(!self.geometry.rink_feature.is_empty(), egui::Button::new("Add mark")).clicked() {
                self.geometry.rink_history.push(self.geometry.rink_observations.clone());
                let line = features.iter().find(|f| f["id"] == self.geometry.rink_feature).is_some_and(|f| f["line"] == true);
                self.geometry.rink_observations.push(json!({"feature":self.geometry.rink_feature,"camera":self.geometry.rink_camera,"line":line,"enabled":true,"pixels":[]}));
                self.geometry.rink_selected = self.geometry.rink_observations.len() - 1;
                self.geometry.dirty = true;
            }
        });
        let previous = self.geometry.rink_observations.clone();
        egui::ScrollArea::vertical()
            .max_height(220.0)
            .id_salt("rink-observation-list")
            .show(ui, |ui| {
                for (index, mark) in self.geometry.rink_observations.iter_mut().enumerate() {
                    ui.horizontal_wrapped(|ui| {
                        ui.selectable_value(
                            &mut self.geometry.rink_selected,
                            index,
                            format!(
                                "{}: Camera {}",
                                index + 1,
                                mark["camera"].as_u64().unwrap_or(0) + 1
                            ),
                        );
                        let mut enabled = mark["enabled"].as_bool().unwrap_or(true);
                        if ui.checkbox(&mut enabled, "Use").changed() {
                            mark["enabled"] = json!(enabled);
                        }
                        let mut identity = text(&mark["feature"]);
                        egui::ComboBox::from_id_salt(("rink-identity", index))
                            .selected_text(if identity.is_empty() {
                                "Unassigned".into()
                            } else {
                                human_label(&identity)
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut identity, String::new(), "Unassigned");
                                for feature in &features {
                                    if feature["line"].as_bool().unwrap_or(false)
                                        == mark["line"].as_bool().unwrap_or(false)
                                    {
                                        let id = text(&feature["id"]);
                                        ui.selectable_value(
                                            &mut identity,
                                            id.clone(),
                                            human_label(&id),
                                        );
                                    }
                                }
                            });
                        mark["feature"] = json!(identity);
                        ui.weak(text(&mark["label"]));
                    });
                }
            });
        if previous != self.geometry.rink_observations {
            self.geometry.rink_history.push(previous);
            self.geometry.dirty = true;
        }
        if let Some(mark) = self
            .geometry
            .rink_observations
            .get(self.geometry.rink_selected)
            .cloned()
        {
            let camera = mark["camera"].as_u64().unwrap_or(0) as usize;
            ui.label(format!("Editing Camera {} landmark", camera + 1));
            let mut descriptor = self.calibration["rink_map"]["sources"][camera].clone();
            let mut overlays =
                collection(&self.calibration["rink_map"]["mapping"]["source_overlays"][camera]);
            overlays.extend(
                self.geometry
                    .rink_observations
                    .iter()
                    .enumerate()
                    .filter(|(index, mark)| {
                        *index != self.geometry.rink_selected
                            && mark["camera"].as_u64() == Some(camera as u64)
                    })
                    .map(|(_, mark)| mark.clone()),
            );
            descriptor["overlays"] = json!(overlays);
            let mut pixels = points(&mark["pixels"]);
            let maximum = if mark["arc"] == true {
                Some(3)
            } else if mark["line"] == true {
                Some(64)
            } else {
                Some(1)
            };
            if mark["arc"] == true {
                ui.label("Drag the three handles along the near-board ice edge. The midpoint controls the smooth arc. Enable Use after review.");
            }
            if self.geometry_image(ui, &descriptor, &mut pixels, maximum, "rink-source-marks") {
                self.geometry
                    .rink_history
                    .push(self.geometry.rink_observations.clone());
                self.geometry.rink_observations[self.geometry.rink_selected]["pixels"] =
                    json!(pixels);
                self.geometry.dirty = true;
            }
            ui.horizontal_wrapped(|ui| {
                if ui.button("Remove last point").clicked() {
                    self.geometry
                        .rink_history
                        .push(self.geometry.rink_observations.clone());
                    pixels.pop();
                    self.geometry.rink_observations[self.geometry.rink_selected]["pixels"] =
                        json!(pixels);
                    self.geometry.dirty = true;
                }
                if ui.button("Delete mark").clicked() {
                    self.geometry
                        .rink_history
                        .push(self.geometry.rink_observations.clone());
                    self.geometry
                        .rink_observations
                        .remove(self.geometry.rink_selected);
                    self.geometry.rink_selected = self.geometry.rink_selected.saturating_sub(1);
                    self.geometry.dirty = true;
                }
            });
            let identity = text(&mark["feature"]);
            if mark["line"] == true
                && ["goal-left-line", "goal-right-line"].contains(&identity.as_str())
                && pixels.len() >= 2
            {
                ui.label("Use goal-line ends at boards only if both numbered endpoints meet the boards at ice level. Choose which end is nearest the camera:");
                ui.horizontal_wrapped(|ui| {
                    for (near,label) in [(0,"First endpoint is near"),(1,"Last endpoint is near")] {
                        if ui.button(label).clicked() {
                            self.geometry.rink_history.push(self.geometry.rink_observations.clone());
                            let prefix = identity.trim_end_matches("-line");
                            let ends = [pixels[0], pixels[pixels.len()-1]];
                            for (side, endpoint) in [("near",near),("far",1-near)] {
                                self.geometry.rink_observations.push(json!({"feature":format!("{prefix}-{side}"),"camera":mark["camera"],"line":false,"enabled":mark["enabled"],"pixels":[ends[endpoint]]}));
                            }
                            self.geometry.rink_selected = self.geometry.rink_observations.len()-2;
                            self.geometry.dirty = true;
                        }
                    }
                });
            }
        } else {
            ui.label("Choose an identified landmark and Add mark, or request suggestions.");
        }
        let dimensions = [
            self.geometry.rink_layout["length"]
                .as_f64()
                .unwrap_or(60.96),
            self.geometry.rink_layout["width"]
                .as_f64()
                .unwrap_or(25.908),
        ];
        ui.collapsing("Rink layout (near boards below)", |ui| {
            let mut reference = vec![[dimensions[0] / 2.0, dimensions[1] / 2.0]];
            rink_canvas(ui, dimensions, &mut reference, 0);
        });
        ui.label(format!(
            "Saved fit: {} — {}",
            text(&self.geometry.rink_fit["state"]),
            text(&self.geometry.rink_fit["message"])
        ));
        ui.horizontal_wrapped(|ui| {
            for (label, use_level) in [("Save partial / fitted map",false),("Save and use for leveling",true)] {
                if ui.button(label).clicked() {
                    self.geometry.rink_use_level = use_level;
                    self.command("calibration.rink-map",json!({"observations":self.geometry.rink_observations,"layout":self.geometry.rink_layout,"landmarks":[],"goals":self.geometry.goals,"allow_partial":true}));
                }
            }
        });
        ui.weak("Partial maps preserve evidence without enabling camera guidance. Saving a complete map enables its use by rink camera controls. Undo restores goal-line conversions in one step.");
    }

    fn leveling_state(&self) -> Value {
        let posts: Vec<Value> = self.leveling_points.iter().enumerate().flat_map(|(image_index, points)| points.chunks_exact(2).map(move |pair| json!({"image_index": image_index, "first": pair[0], "second": pair[1]}))).collect();
        json!({"rotation_degrees": self.geometry.rotation, "crop": self.geometry.crop, "auto_crop": self.leveling_auto_crop, "posts": posts})
    }

    fn draw_leveling(&mut self, ui: &mut egui::Ui) {
        ui.heading("Level panorama from vertical posts");
        ui.label("Mark the two ends of visible vertical posts on each original camera image. Estimate the level, inspect the rendered preview, then save it.");
        if self.leveling.is_null() {
            if ui.button("Open leveling editor").clicked() {
                self.command("leveling.open", json!({}));
            }
            return;
        }
        let sources = collection(&self.leveling["sources"]);
        let mut points = std::mem::take(&mut self.leveling_points);
        ui.columns(2, |columns| {
            for (index, column) in columns.iter_mut().enumerate() {
                column.strong(if index == 0 {
                    "Left camera posts"
                } else {
                    "Right camera posts"
                });
                self.geometry_image(
                    column,
                    sources.get(index).unwrap_or(&Value::Null),
                    &mut points[index],
                    None,
                    &format!("leveling-{index}"),
                );
                if column.button("Remove last post").clicked() {
                    points[index].pop();
                    points[index].pop();
                }
                column.weak(format!("{} complete posts", points[index].len() / 2));
            }
        });
        self.leveling_points = points;
        let complete = self
            .leveling_points
            .iter()
            .all(|points| points.len() % 2 == 0);
        let count: usize = self
            .leveling_points
            .iter()
            .map(|points| points.len() / 2)
            .sum();
        ui.horizontal_wrapped(|ui| {
            for (index, label) in ["Yaw", "Pitch", "Roll"].iter().enumerate() {
                ui.label(*label);
                without_value_scroll(ui, |ui| ui.add(egui::DragValue::new(&mut self.geometry.rotation[index]).range(-180.0..=180.0).speed(0.1).suffix("°")));
            }
            if ui.add_enabled(complete && count >= 2, egui::Button::new("Estimate level")).clicked() {
                self.command("leveling.estimate", json!({"posts": self.leveling_state()["posts"], "yaw": self.geometry.rotation[0]}));
            }
        });
        ui.checkbox(
            &mut self.leveling_auto_crop,
            "Automatically crop valid pixels",
        );
        if !self.leveling_auto_crop {
            ui.horizontal_wrapped(|ui| {
                for (index, label) in ["Left", "Right", "Top", "Bottom"].iter().enumerate() {
                    ui.label(*label);
                    without_value_scroll(ui, |ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.geometry.crop[index])
                                .range(0.0..=1.0)
                                .speed(0.001)
                                .max_decimals(4),
                        )
                    });
                }
            });
        }
        let state = self.leveling_state();
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(complete, egui::Button::new("Render leveling preview"))
                .clicked()
            {
                self.command("leveling.preview", json!({"state": state}));
            }
            let current_preview =
                self.leveling_preview["token"].is_string() && state == self.leveling_preview_state;
            if ui
                .add_enabled(current_preview, egui::Button::new("Save reviewed leveling"))
                .clicked()
            {
                self.command(
                    "leveling.save",
                    json!({"state": state, "token": self.leveling_preview["token"]}),
                );
            }
            if ui.button("Close leveling editor").clicked() {
                self.command("leveling.close", json!({}));
            }
        });
        if !complete {
            ui.colored_label(
                egui::Color32::YELLOW,
                "Complete both endpoints of the last post before estimating or rendering.",
            );
        }
        if !self.leveling_preview.is_null() {
            let descriptor = self.leveling_preview.clone();
            self.geometry_image(
                ui,
                &descriptor,
                &mut Vec::new(),
                Some(0),
                "leveling-preview",
            );
        }
    }

    fn draw_experiments(&mut self, ui: &mut egui::Ui) {
        ui.heading("Stitching & Camera Experiments");
        ui.label("Create a candidate configuration, run it, inspect the result, and apply the selected candidate to the game.");
        ui.horizontal_wrapped(|ui| {
            ui.label("Candidate source start");
            without_value_scroll(ui, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.experiment_start)
                        .range(0.0..=604800.0)
                        .suffix(" s"),
                )
            });
            ui.label("Duration");
            without_value_scroll(ui, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.experiment_duration)
                        .range(0.0..=86400.0)
                        .suffix(" s"),
                )
            });
            ui.weak("0 runs to the end");
        });
        for experiment in collection(&self.snapshot["experiments"]) {
            let id = object_id(&experiment);
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(
                    &mut self.selected_experiment,
                    id.clone(),
                    object_label(&experiment),
                );
                ui.label(text(&experiment["status"]));
                if ui.small_button("Run candidate").clicked() {
                    let arguments = if experiment["replay"].is_object() {
                        Vec::new()
                    } else {
                        bounded_run_arguments(self.experiment_start, self.experiment_duration)
                    };
                    self.command(
                        "experiments.run",
                        json!({"experiment_id": id, "arguments": arguments}),
                    );
                }
                if ui.small_button("Use candidate").clicked() {
                    self.command("experiments.promote", json!({"experiment_id": id}));
                }
            });
            if self.selected_experiment == id {
                if experiment["replay"].is_object() {
                    ui.weak("Replay uses the source and time range saved with this candidate.");
                }
                if let Some(changes) = experiment["changes"].as_object() {
                    for (key, value) in changes {
                        ui.label(format!("{}: {}", human_label(key), text(value)));
                    }
                }
            }
        }
        if ui
            .add_enabled(
                !self.selected_experiment.is_empty(),
                egui::Button::new("Compare camera telemetry"),
            )
            .clicked()
        {
            self.command(
                "experiments.telemetry",
                json!({"experiment_id": self.selected_experiment}),
            );
        }
        if !self.telemetry.is_null() {
            draw_telemetry(ui, &self.telemetry);
        }
        ui.separator();
        ui.heading("New candidate");
        path_field(
            ui,
            "Candidate name",
            &mut self.experiment_name,
            "e.g. wider field of view",
        );
        ui.horizontal(|ui| {
            let changed = ui
                .selectable_value(&mut self.experiment_kind, "stitch".into(), "Stitching")
                .changed()
                | ui.selectable_value(&mut self.experiment_kind, "camera".into(), "Camera")
                    .changed();
            if changed {
                self.experiment_settings.clear();
                self.experiment_changes.clear();
            }
        });
        if self.experiment_kind == "camera" {
            ui.checkbox(
                &mut self.experiment_replay,
                "Replay an existing tracking recording",
            );
            if self.experiment_replay {
                self.draw_replay_sources(ui);
            }
        }
        let prefix = if self.experiment_kind == "camera" {
            "rink.camera."
        } else {
            "stitching."
        };
        let catalog: Vec<Value> = collection(&self.calibration["catalog"])
            .into_iter()
            .filter(|item| {
                item["key"]
                    .as_str()
                    .is_some_and(|key| key.starts_with(prefix))
            })
            .collect();
        if catalog.is_empty() {
            if ui.button("Load available settings").clicked() {
                self.command("calibration.snapshot", json!({}));
            }
        } else {
            egui::ComboBox::from_id_salt("experiment-setting")
                .selected_text(if self.experiment_settings.is_empty() {
                    "Choose a setting"
                } else {
                    &self.experiment_settings
                })
                .show_ui(ui, |ui| {
                    for item in &catalog {
                        if let Some(key) = item["key"].as_str() {
                            ui.selectable_value(
                                &mut self.experiment_settings,
                                key.into(),
                                item["label"].as_str().unwrap_or(key),
                            );
                        }
                    }
                });
            if ui
                .add_enabled(
                    !self.experiment_settings.is_empty(),
                    egui::Button::new("Add setting"),
                )
                .clicked()
            {
                let value = self
                    .settings
                    .get(&self.experiment_settings)
                    .cloned()
                    .unwrap_or(Value::Null);
                self.experiment_changes
                    .insert(self.experiment_settings.clone(), value);
            }
        }
        let mut remove = None;
        for (key, value) in &mut self.experiment_changes {
            ui.push_id(key, |ui| {
                if let Some(item) = catalog
                    .iter()
                    .find(|item| item["key"].as_str() == Some(key.as_str()))
                {
                    ui.strong(item["label"].as_str().unwrap_or(key));
                    setting_editor(ui, item, value);
                }
                if ui.small_button("Remove setting").clicked() {
                    remove = Some(key.clone());
                }
            });
        }
        if let Some(key) = remove {
            self.experiment_changes.remove(&key);
        }
        if ui
            .add_enabled(
                !self.experiment_name.trim().is_empty()
                    && !self.experiment_changes.is_empty()
                    && (!self.experiment_replay
                        || self.experiment_kind != "camera"
                        || (!self.replay_run.is_empty() && self.replay_end > self.replay_start)),
                egui::Button::new("Create candidate"),
            )
            .clicked()
        {
            let mut args = json!({"kind": self.experiment_kind, "name": self.experiment_name, "changes": self.experiment_changes});
            if self.experiment_replay && self.experiment_kind == "camera" {
                let mut replay = json!({"recording_path":self.replay_recording,"run_id":self.replay_run,"start_ms":(self.replay_start * 1000.0).round() as i64,"end_ms":(self.replay_end * 1000.0).round() as i64});
                if !self.replay_source.is_empty() {
                    replay["source_path"] = json!(self.replay_source);
                }
                args["replay"] = replay;
            }
            self.command("experiments.create", args);
        }
    }

    fn draw_replay_sources(&mut self, ui: &mut egui::Ui) {
        ui.weak("Compare camera settings using the same recorded player tracks. Each candidate keeps its own recording snapshot.");
        let mut changed = false;
        egui::ComboBox::from_id_salt("replay-recording")
            .selected_text(if self.replay_recording.is_empty() {
                "Choose a telemetry recording"
            } else {
                &self.replay_recording
            })
            .show_ui(ui, |ui| {
                for recording in collection(&self.snapshot["recordings"]) {
                    if let Some(path) = recording["path"].as_str() {
                        changed |= ui
                            .selectable_value(
                                &mut self.replay_recording,
                                path.into(),
                                object_label(&recording),
                            )
                            .changed();
                    }
                }
            });
        if changed
            || (!self.replay_recording.is_empty()
                && ui.button("Reload recording details").clicked())
        {
            self.replay_sources = Value::Null;
            self.replay_run.clear();
            self.replay_source.clear();
            self.command(
                "experiments.replay_sources",
                json!({"recording_path":self.replay_recording}),
            );
        }
        let runs = collection(&self.replay_sources["runs"]);
        if runs.is_empty() {
            return;
        }
        egui::ComboBox::from_id_salt("replay-run")
            .selected_text(&self.replay_run)
            .show_ui(ui, |ui| {
                for run in &runs {
                    if ui
                        .selectable_value(
                            &mut self.replay_run,
                            text(&run["run_id"]),
                            format!(
                                "{} · {} samples",
                                text(&run["producer"]),
                                text(&run["sample_count"])
                            ),
                        )
                        .changed()
                    {
                        self.replay_start = run["start_ms"].as_f64().unwrap_or(0.0) / 1000.0;
                        self.replay_end = run["end_ms"].as_f64().unwrap_or(0.0) / 1000.0;
                    }
                }
            });
        egui::ComboBox::from_id_salt("replay-source")
            .selected_text(if self.replay_source.is_empty() {
                "Matching full view"
            } else {
                &self.replay_source
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.replay_source, String::new(), "Matching full view");
                for source in collection(&self.replay_sources["sources"]) {
                    if let Some(path) = source["path"].as_str() {
                        ui.selectable_value(
                            &mut self.replay_source,
                            path.into(),
                            object_label(&source),
                        );
                    }
                }
            });
        ui.horizontal_wrapped(|ui| {
            ui.label("Recording range");
            without_value_scroll(ui, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.replay_start)
                        .range(0.0..=604800.0)
                        .suffix(" s"),
                )
            });
            ui.label("to");
            without_value_scroll(ui, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.replay_end)
                        .range(0.0..=604800.0)
                        .suffix(" s"),
                )
            });
        });
        if let Some(run) = runs
            .iter()
            .find(|run| run["run_id"].as_str() == Some(self.replay_run.as_str()))
        {
            let segments = collection(&run["segments"]);
            if segments.len() > 1 {
                ui.collapsing("Choose a continuous recording segment", |ui| {
                    for segment in segments {
                        let start = segment["start_ms"].as_f64().unwrap_or(0.0) / 1000.0;
                        let end = segment["end_ms"].as_f64().unwrap_or(0.0) / 1000.0;
                        if ui.button(format!("{start:.3} s – {end:.3} s")).clicked() {
                            self.replay_start = start;
                            self.replay_end = end;
                        }
                    }
                });
            }
        }
    }

    fn open_media(&mut self, path: &str) {
        self.media_path = path.to_string();
        self.media_frame = Value::Null;
        self.media_time = 0.0;
        self.media_duration = 0.0;
        self.media_playing = false;
        self.media_needs_frame = false;
        self.command("media.info", json!({"path": path}));
    }

    fn draw_media(&mut self, ui: &mut egui::Ui) {
        let elapsed = self.media_last_tick.elapsed().as_secs_f64();
        self.media_last_tick = Instant::now();
        if self.media_playing {
            self.media_time = (self.media_time + elapsed.min(0.5)).min(self.media_duration);
            self.media_needs_frame = true;
            if self.media_time >= self.media_duration {
                self.media_playing = false;
            }
        }
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    self.media_duration > 0.0,
                    egui::Button::new(if self.media_playing {
                        "Pause preview"
                    } else {
                        "Play preview"
                    }),
                )
                .clicked()
            {
                self.media_playing = !self.media_playing;
            }
            for (label, direction) in [("Previous frame", -1.0), ("Next frame", 1.0)] {
                if ui
                    .add_enabled(self.media_duration > 0.0, egui::Button::new(label))
                    .clicked()
                {
                    self.media_playing = false;
                    self.media_time = (self.media_time + direction / self.media_fps)
                        .clamp(0.0, self.media_duration);
                    self.media_needs_frame = true;
                }
            }
            if without_value_scroll(ui, |ui| {
                ui.add(
                    egui::DragValue::new(&mut self.media_time)
                        .range(0.0..=self.media_duration)
                        .speed(1.0 / self.media_fps)
                        .suffix(" s")
                        .max_decimals(3),
                )
            })
            .changed()
            {
                self.media_needs_frame = true;
            }
            ui.weak(format!("of {}", crate::playback_time(self.media_duration)));
        });
        if without_value_scroll(ui, |ui| {
            ui.add(
                egui::Slider::new(&mut self.media_time, 0.0..=self.media_duration)
                    .show_value(false),
            )
        })
        .changed()
        {
            self.media_needs_frame = true;
        }
        if self.media_needs_frame
            && !self
                .pending
                .values()
                .any(|pending| pending.operation == "media.frame")
        {
            self.media_needs_frame = false;
            self.command("media.frame", json!({"path": self.media_path, "time_ms": (self.media_time.min((self.media_duration - 1.0 / self.media_fps).max(0.0)) * 1000.0).round() as u64, "width": 1280}));
        }
        let path = self.media_frame["path"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        if path.is_empty() {
            ui.label("Loading video frame…");
            return;
        }
        if !self.textures.contains_key(&path) {
            match load_color_image(Path::new(&path)) {
                Ok(image) => {
                    let texture = ui.ctx().load_texture(
                        format!("media-{path}"),
                        image,
                        egui::TextureOptions::LINEAR,
                    );
                    // Keep a bounded number of decoded inspection frames.
                    self.textures.retain(|name, _| {
                        !name.starts_with(&self.directory.to_string_lossy().to_string())
                    });
                    self.textures.insert(
                        path.clone(),
                        CachedImage {
                            modified: None,
                            texture,
                        },
                    );
                }
                Err(error) => {
                    ui.colored_label(
                        egui::Color32::LIGHT_RED,
                        format!("Could not display video frame: {error:#}"),
                    );
                    return;
                }
            }
        }
        let texture = self.textures[&path].texture.clone();
        let size = texture.size_vec2();
        let width = ui.available_width().min(1000.0);
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(width, (width * size.y / size.x).min(460.0)),
            egui::Sense::click_and_drag(),
        );
        let scale = (rect.width() / size.x).min(rect.height() / size.y);
        let image_rect = egui::Rect::from_center_size(rect.center(), size * scale);
        let painter = ui.painter().with_clip_rect(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::BLACK);
        painter.image(
            texture.id(),
            image_rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        if self.media_path != self.highlight_source {
            return;
        }
        let source_time = self.media_time * 1000.0
            + self.highlight_document["archive_offset_ms"]
                .as_f64()
                .unwrap_or(0.0);
        let Some(annotations) = self.highlight_document["intervals"]
            .get_mut(self.selected_highlight)
            .and_then(|interval| interval["annotations"].as_array_mut())
        else {
            return;
        };
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.place_annotation, "Place annotations on image");
            for (index, _) in annotations.iter().enumerate() {
                ui.selectable_value(
                    &mut self.selected_annotation,
                    index,
                    format!("Annotation {}", index + 1),
                );
            }
        });
        if self.place_annotation && (response.clicked() || response.dragged()) {
            if let (Some(annotation), Some(position)) = (
                annotations.get_mut(self.selected_annotation),
                response
                    .interact_pointer_pos()
                    .filter(|position| image_rect.contains(*position)),
            ) {
                let x = ((position.x - image_rect.left()) / image_rect.width()) as f64;
                let y = ((position.y - image_rect.top()) / image_rect.height()) as f64;
                annotation["x"] = json!(x);
                annotation["y"] = json!(y);
                if annotation["motion"]
                    .as_str()
                    .is_some_and(|mode| mode != "fixed")
                {
                    if !annotation["positions"].is_array() {
                        annotation["positions"] = json!([]);
                    }
                    let positions = annotation["positions"]
                        .as_array_mut()
                        .expect("position array");
                    let timestamp = source_time.round() as u64;
                    positions.retain(|position| position["time_ms"].as_u64() != Some(timestamp));
                    positions.push(json!({"time_ms": timestamp, "x": x, "y": y}));
                    positions.sort_by_key(|position| position["time_ms"].as_u64().unwrap_or(0));
                }
                self.highlight_dirty = true;
            }
        }
        for (index, annotation) in annotations.iter().enumerate() {
            paint_annotation(
                &painter,
                image_rect,
                annotation,
                source_time,
                self.place_annotation && index == self.selected_annotation,
            );
        }
    }

    fn select_highlight_archive(&mut self, path: String) {
        self.highlight_source = path;
        self.highlight_document["archive_path"] = json!(self.highlight_source);
        self.track_choices = Value::Null;
        self.track_run.clear();
        self.highlight_dirty = true;
    }

    fn draw_highlights(&mut self, ui: &mut egui::Ui) {
        ui.heading("Highlights");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Load saved highlights").clicked() {
                self.command("highlights.load", json!({}));
            }
            if ui.button("Save highlights").clicked() {
                self.command("highlights.save", json!({"plan": self.highlight_document}));
            }
            if ui.button("Preview highlights").clicked() {
                self.command(
                    "highlights.preview",
                    json!({"plan": self.highlight_document}),
                );
            }
            if ui.button("Render highlights").clicked() {
                self.command(
                    "highlights.render",
                    json!({"plan": self.highlight_document}),
                );
            }
            if self.highlight_dirty {
                ui.weak("Unsaved changes");
            }
        });
        let archives = collection(&self.snapshot["archives"]);
        egui::ComboBox::from_id_salt("highlight-archive")
            .width(ui.available_width().min(700.0))
            .selected_text(if self.highlight_source.is_empty() {
                "Choose an archive"
            } else {
                &self.highlight_source
            })
            .show_ui(ui, |ui| {
                for archive in archives {
                    let path = archive["path"].as_str().unwrap_or_default();
                    if ui
                        .selectable_value(
                            &mut self.highlight_source,
                            path.into(),
                            object_label(&archive),
                        )
                        .changed()
                    {
                        self.select_highlight_archive(self.highlight_source.clone());
                    }
                }
            });
        if path_field(
            ui,
            "Archive file",
            &mut self.highlight_source,
            "/path/to/game-1.mp4",
        ) {
            self.select_highlight_archive(self.highlight_source.clone());
        }
        let mut base_name = self.highlight_document["base_name"]
            .as_str()
            .unwrap_or("highlights")
            .to_string();
        if path_field(ui, "Output name", &mut base_name, "highlights") {
            self.highlight_document["base_name"] = json!(base_name);
            self.highlight_dirty = true;
        }
        let mut offset = self.highlight_document["archive_offset_ms"]
            .as_f64()
            .unwrap_or(0.0)
            / 1000.0;
        ui.horizontal(|ui| {
            ui.label("Archive time offset");
            if without_value_scroll(ui, |ui| {
                ui.add(egui::DragValue::new(&mut offset).speed(0.1).suffix(" s"))
            })
            .changed()
            {
                self.highlight_document["archive_offset_ms"] =
                    json!((offset * 1000.0).round() as i64);
                self.track_choices = Value::Null;
                self.highlight_dirty = true;
            }
        });
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !self.highlight_source.is_empty(),
                    egui::Button::new("Inspect archive"),
                )
                .clicked()
            {
                self.open_media(&self.highlight_source.clone());
            }
            if !self.media_path.is_empty() {
                ui.weak(format!("Inspecting {}", self.media_path));
            }
        });
        if !self.media_path.is_empty() {
            self.draw_media(ui);
        }
        ui.separator();
        if !self.highlight_document["intervals"].is_array() {
            self.highlight_document["intervals"] = json!([]);
        }
        let intervals = self.highlight_document["intervals"]
            .as_array_mut()
            .expect("initialized interval array");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Add clip").clicked() {
                intervals.push(json!({"label": format!("Clip {}", intervals.len() + 1), "event_mode": false, "start_ms": 0, "end_ms": 5000, "is_card": false, "annotations": []}));
                self.selected_highlight = intervals.len() - 1;
                self.highlight_dirty = true;
            }
            if ui.button("Add title card").clicked() {
                intervals.push(json!({"label": "Title card", "event_mode": false, "start_ms": 0, "end_ms": 3000, "is_card": true, "annotations": [], "card": {"heading": "Game highlights", "team_a": "", "team_b": "", "date": "", "duration_ms": 3000, "background": "#101b2c", "color": "#ffffff", "logo_a": "", "logo_b": ""}}));
                self.selected_highlight = intervals.len() - 1;
                self.highlight_dirty = true;
            }
            if ui.add_enabled(!intervals.is_empty(), egui::Button::new("Remove selected")).clicked() {
                intervals.remove(self.selected_highlight.min(intervals.len() - 1));
                self.selected_highlight = self.selected_highlight.saturating_sub(1);
                self.highlight_dirty = true;
            }
            if ui.add_enabled(self.selected_highlight > 0, egui::Button::new("Move earlier")).clicked() {
                intervals.swap(self.selected_highlight, self.selected_highlight - 1);
                self.selected_highlight -= 1;
                self.highlight_dirty = true;
            }
            if ui.add_enabled(self.selected_highlight + 1 < intervals.len(), egui::Button::new("Move later")).clicked() {
                intervals.swap(self.selected_highlight, self.selected_highlight + 1);
                self.selected_highlight += 1;
                self.highlight_dirty = true;
            }
        });
        ui.horizontal_wrapped(|ui| {
            for (index, interval) in intervals.iter().enumerate() {
                ui.selectable_value(
                    &mut self.selected_highlight,
                    index,
                    format!(
                        "{} · {}",
                        index + 1,
                        interval["label"].as_str().unwrap_or("Clip")
                    ),
                );
            }
        });
        if let Some(interval) = intervals.get_mut(self.selected_highlight) {
            self.highlight_dirty |= string_property(ui, interval, "label", "Label");
            if interval["is_card"].as_bool() == Some(true) {
                for (key, label) in [
                    ("heading", "Heading"),
                    ("team_a", "Home team"),
                    ("team_b", "Away team"),
                    ("date", "Date"),
                    ("background", "Background color"),
                    ("color", "Text color"),
                    ("logo_a", "Home team logo"),
                    ("logo_b", "Away team logo"),
                ] {
                    self.highlight_dirty |= string_property(ui, &mut interval["card"], key, label);
                }
                self.highlight_dirty |=
                    time_property(ui, &mut interval["card"], "duration_ms", "Duration");
            } else {
                self.highlight_dirty |= time_property(ui, interval, "start_ms", "Clip start");
                self.highlight_dirty |= time_property(ui, interval, "end_ms", "Clip end");
                let mut event_mode = interval["event_mode"].as_bool().unwrap_or(false);
                if ui.checkbox(&mut event_mode, "Event time mode").changed() {
                    interval["event_mode"] = json!(event_mode);
                    if event_mode && interval["event_ms"].is_null() {
                        let start = interval["start_ms"].as_i64().unwrap_or(0);
                        let duration = interval["end_ms"].as_i64().unwrap_or(start + 5000) - start;
                        interval["event_ms"] = json!(start + duration * 3 / 4);
                        interval["duration_ms"] = json!(duration);
                    }
                    self.highlight_dirty = true;
                }
                if event_mode {
                    self.highlight_dirty |= time_property(ui, interval, "event_ms", "Event time");
                    self.highlight_dirty |=
                        time_property(ui, interval, "duration_ms", "Clip duration");
                }
                ui.weak("Clip and annotation times use the source timeline; the archive offset aligns timestamps with the recording.");
                self.highlight_dirty |= annotations_editor(ui, interval);
            }
        }
        self.draw_track_capture(ui);
    }

    fn draw_track_capture(&mut self, ui: &mut egui::Ui) {
        let annotations = collection(
            &self.highlight_document["intervals"]
                .get(self.selected_highlight)
                .unwrap_or(&Value::Null)["annotations"],
        );
        if annotations.is_empty() {
            return;
        }
        ui.collapsing("Follow a recorded player", |ui| {
            ui.weak("Use tracking from this archive's matching telemetry. Inspect the archive at the player selection time, then capture up to 120 seconds of motion.");
            ui.horizontal_wrapped(|ui| {
                ui.label("Annotation");
                egui::ComboBox::from_id_salt("tracked-annotation").selected_text(format!("{}", self.selected_annotation + 1)).show_ui(ui, |ui| {
                    for index in 0..annotations.len() { ui.selectable_value(&mut self.selected_annotation, index, format!("Annotation {}", index + 1)); }
                });
                ui.label("Recorded view");
                egui::ComboBox::from_id_salt("tracked-route").selected_text(human_label(&self.track_route)).show_ui(ui, |ui| {
                    for route in ["program", "program-full", "stitched"] {
                        if ui.selectable_value(&mut self.track_route, route.into(), human_label(route)).changed() { self.track_choices = Value::Null; }
                    }
                });
            });
            let runs = collection(&self.track_choices["runs"]);
            if runs.len() > 1 {
                egui::ComboBox::from_id_salt("tracked-run").selected_text(&self.track_run).show_ui(ui, |ui| {
                    for run in runs { if ui.selectable_value(&mut self.track_run, text(&run["run_id"]), format!("{} · {}", text(&run["producer"]), text(&run["started_utc"]))).changed() { self.track_choices["choices"] = json!([]); } }
                });
            }
            let mut common = json!({"archive_path":self.highlight_source,"archive_offset_ms":self.highlight_document["archive_offset_ms"].as_i64().unwrap_or(0),"route":self.track_route});
            if !self.track_run.is_empty() { common["run_id"] = json!(self.track_run); }
            let waiting = self.pending.values().any(|command| matches!(command.operation.as_str(), "highlights.track_choices" | "highlights.capture_track"));
            if ui.add_enabled(!self.highlight_source.is_empty() && !waiting, egui::Button::new("Find players at preview time")).clicked() {
                let mut args = common.clone();
                args["time_ms"] = json!((self.media_time * 1000.0).round() as i64 + self.highlight_document["archive_offset_ms"].as_i64().unwrap_or(0));
                self.command("highlights.track_choices", args);
            }
            let choices = collection(&self.track_choices["choices"]);
            for (index, choice) in choices.iter().enumerate() {
                ui.selectable_value(&mut self.selected_track, index, format!("Player {} · x {:.0}% y {:.0}%", text(&choice["tracking_id"]), choice["box"]["x"].as_f64().unwrap_or(0.0) * 100.0, choice["box"]["y"].as_f64().unwrap_or(0.0) * 100.0));
            }
            if !self.track_choices.is_null() && choices.is_empty() { ui.weak("No tracked player is visible at this time in the selected recording."); }
            let target = annotations.get(self.selected_annotation);
            if ui.add_enabled(!waiting && target.is_some() && choices.get(self.selected_track).is_some(), egui::Button::new("Capture player motion")).clicked() {
                let annotation = target.expect("selected annotation");
                let mut args = common;
                args["start_ms"] = annotation["start_ms"].clone();
                args["end_ms"] = annotation["end_ms"].clone();
                args["choice"] = choices[self.selected_track].clone();
                self.track_capture_target = Some((self.selected_highlight, self.selected_annotation, annotation.clone()));
                self.command("highlights.capture_track", args);
            }
        });
    }
}

fn bounded_run_arguments(start: f64, duration: f64) -> Vec<String> {
    let mut arguments = Vec::new();
    if start > 0.0 {
        let milliseconds = (start * 1000.0).round() as u64;
        arguments.extend([
            "--start-time".into(),
            format!(
                "{:02}:{:02}:{:02}.{:03}",
                milliseconds / 3_600_000,
                milliseconds / 60_000 % 60,
                milliseconds / 1000 % 60,
                milliseconds % 1000
            ),
        ]);
    }
    if duration > 0.0 {
        arguments.push(format!("-t={duration}"));
    }
    arguments
}

fn read_json(path: &Path) -> Result<Value, String> {
    let data =
        fs::read_to_string(path).map_err(|error| format!("Read {}: {error}", path.display()))?;
    serde_json::from_str(&data).map_err(|error| format!("Parse {}: {error}", path.display()))
}

fn text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn collection(value: &Value) -> Vec<Value> {
    match value {
        Value::Array(items) => items.clone(),
        Value::Object(items) => items.values().cloned().collect(),
        _ => Vec::new(),
    }
}

fn object_id(value: &Value) -> String {
    for key in ["id", "path", "game_id", "name"] {
        if let Some(id) = value[key].as_str() {
            return id.into();
        }
    }
    text(value)
}

fn object_label(value: &Value) -> String {
    for key in ["name", "label", "id", "path"] {
        if let Some(label) = value[key].as_str() {
            return label.into();
        }
    }
    text(value)
}

fn game_id(path: &str) -> &str {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
}

fn operation_label(operation: &str) -> String {
    human_label(operation)
}

fn human_label(value: &str) -> String {
    value.replace(['.', '_', '-'], " ")
}

fn path_field(ui: &mut egui::Ui, label: &str, value: &mut String, hint: &str) -> bool {
    ui.horizontal(|ui| {
        ui.add_sized([150.0, 20.0], egui::Label::new(label));
        ui.add(
            egui::TextEdit::singleline(value)
                .hint_text(hint)
                .desired_width(f32::INFINITY),
        )
        .changed()
    })
    .inner
}

fn source_field(ui: &mut egui::Ui, label: &str, value: &mut String, hint: &str) -> bool {
    ui.horizontal(|ui| {
        ui.add_sized(
            [150.0, 38.0],
            egui::Label::new(format!("{label}\nOne chapter per line")),
        );
        ui.add(
            egui::TextEdit::multiline(value)
                .hint_text(hint)
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        )
        .changed()
    })
    .inner
}

fn point(value: &Value) -> [f64; 2] {
    [
        value[0].as_f64().unwrap_or(0.0),
        value[1].as_f64().unwrap_or(0.0),
    ]
}
fn points(value: &Value) -> Vec<[f64; 2]> {
    collection(value).iter().map(point).collect()
}

fn point_table(ui: &mut egui::Ui, id: &str, points: &mut [[f64; 2]]) {
    egui::Grid::new(id).striped(true).show(ui, |ui| {
        for (index, point) in points.iter_mut().enumerate() {
            ui.label((index + 1).to_string());
            without_value_scroll(ui, |ui| {
                ui.add(egui::DragValue::new(&mut point[0]).speed(0.1).prefix("x "))
            });
            without_value_scroll(ui, |ui| {
                ui.add(egui::DragValue::new(&mut point[1]).speed(0.1).prefix("y "))
            });
            ui.end_row();
        }
    });
}

fn setting_editor(ui: &mut egui::Ui, item: &Value, value: &mut Value) -> bool {
    if item["key"] == "stitching.rink_mask_frame_time" {
        let before = value.clone();
        let mut mode = if value.is_null() {
            0
        } else if value == "auto" {
            1
        } else {
            2
        };
        let initial_mode = mode;
        egui::ComboBox::from_id_salt("mask-frame-mode")
            .selected_text(["Rink default", "Automatic (first frame)", "Custom time"][mode])
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut mode, 0, "Rink default");
                ui.selectable_value(&mut mode, 1, "Automatic (first frame)");
                ui.selectable_value(&mut mode, 2, "Custom time");
            });
        if mode != initial_mode {
            *value = match mode {
                0 => Value::Null,
                1 => json!("auto"),
                _ => json!("-00:00:01"),
            };
        }
        if mode == 2 {
            let mut timestamp = text(value);
            if ui.text_edit_singleline(&mut timestamp).changed() {
                *value = json!(timestamp);
            }
            ui.weak("HH:MM:SS.mmm; negative times count back from the recording end");
        }
        return *value != before;
    }
    match item["kind"].as_str().unwrap_or("text") {
        "bool" => {
            let mut checked = value.as_bool().unwrap_or(false);
            let changed = ui.checkbox(&mut checked, "Enabled").changed();
            if changed {
                *value = json!(checked);
            }
            changed
        }
        "int" | "number" => {
            if value.is_null() {
                ui.weak("Inherited from the game or system configuration");
                if ui.button("Set explicit value").clicked() {
                    *value = json!(item["min"].as_f64().unwrap_or(0.0));
                    return true;
                }
                return false;
            }
            let mut number = value.as_f64().unwrap_or(0.0);
            let min = item["min"].as_f64().unwrap_or(-1e12);
            let max = item["max"].as_f64().unwrap_or(1e12);
            let integer = item["kind"] == "int";
            let changed = without_value_scroll(ui, |ui| {
                ui.add(
                    egui::DragValue::new(&mut number)
                        .range(min..=max)
                        .speed(if integer { 1.0 } else { 0.01 })
                        .max_decimals(if integer { 0 } else { 6 })
                        .update_while_editing(false),
                )
            })
            .changed();
            if changed {
                *value = if integer {
                    json!(number.round() as i64)
                } else {
                    json!(number)
                };
            }
            changed
        }
        "choice" => {
            let mut selected = text(value);
            let original = selected.clone();
            egui::ComboBox::from_id_salt("value-choice")
                .selected_text(&selected)
                .show_ui(ui, |ui| {
                    for choice in collection(&item["choices"]) {
                        let label = text(&choice);
                        ui.selectable_value(&mut selected, label.clone(), label);
                    }
                });
            if selected != original {
                *value = json!(selected);
                true
            } else {
                false
            }
        }
        "array" => {
            if let Some(items) = value.as_array_mut() {
                let mut changed = false;
                ui.horizontal_wrapped(|ui| {
                    for (index, item) in items.iter_mut().enumerate() {
                        ui.push_id(index, |ui| {
                            if let Some(mut number) = item.as_f64() {
                                if without_value_scroll(ui, |ui| {
                                    ui.add(egui::DragValue::new(&mut number).speed(0.1))
                                })
                                .changed()
                                {
                                    *item = json!(number);
                                    changed = true;
                                }
                            } else {
                                ui.label(text(item));
                            }
                        });
                    }
                });
                changed
            } else {
                ui.weak("Inherited; configure this value with its calibration editor.");
                false
            }
        }
        _ => {
            let mut string = text(value);
            let changed = ui
                .add(egui::TextEdit::singleline(&mut string).desired_width(f32::INFINITY))
                .changed();
            if changed {
                *value = json!(string);
            }
            changed
        }
    }
}

fn string_property(ui: &mut egui::Ui, object: &mut Value, key: &str, label: &str) -> bool {
    let mut value = text(&object[key]);
    if path_field(ui, label, &mut value, "") {
        object[key] = json!(value);
        true
    } else {
        false
    }
}

fn time_property(ui: &mut egui::Ui, object: &mut Value, key: &str, label: &str) -> bool {
    let mut value = object[key].as_f64().unwrap_or(0.0) / 1000.0;
    let changed = ui
        .horizontal(|ui| {
            ui.add_sized([150.0, 20.0], egui::Label::new(label));
            without_value_scroll(ui, |ui| {
                ui.add(
                    egui::DragValue::new(&mut value)
                        .range(0.0..=604800.0)
                        .speed(0.1)
                        .suffix(" s")
                        .max_decimals(3),
                )
            })
            .changed()
        })
        .inner;
    if changed {
        object[key] = json!((value * 1000.0).round() as i64);
    }
    changed
}

fn annotations_editor(ui: &mut egui::Ui, interval: &mut Value) -> bool {
    if !interval["annotations"].is_array() {
        interval["annotations"] = json!([]);
    }
    let annotations = interval["annotations"]
        .as_array_mut()
        .expect("initialized annotations");
    let mut changed = false;
    ui.collapsing("Annotations", |ui| {
        let mut remove = None;
        for (index, annotation) in annotations.iter_mut().enumerate() {
            ui.push_id(index, |ui| {
                let mut kind = annotation["kind"].as_str().unwrap_or("text").to_string();
                ui.horizontal(|ui| {
                    for option in ["text", "arrow", "box"] {
                        if ui.selectable_value(&mut kind, option.into(), human_label(option)).changed() { annotation["kind"] = json!(kind); changed = true; }
                    }
                    if ui.small_button("Remove").clicked() { remove = Some(index); }
                });
                changed |= string_property(ui, annotation, "text", "Text");
                changed |= string_property(ui, annotation, "color", "Color");
                changed |= time_property(ui, annotation, "start_ms", "Visible from");
                changed |= time_property(ui, annotation, "end_ms", "Visible until");
                for (key, label) in [("x", "X"), ("y", "Y"), ("dx", "Width / direction X"), ("dy", "Height / direction Y"), ("size", "Text size"), ("thickness", "Line thickness")] {
                    let mut value = annotation[key].as_f64().unwrap_or(0.5);
                    ui.horizontal(|ui| {
                        ui.label(label);
                        if without_value_scroll(ui, |ui| ui.add(egui::DragValue::new(&mut value).range(if matches!(key, "dx" | "dy") { -1.0..=1.0 } else { 0.0..=1.0 }).speed(0.01))).changed() { annotation[key] = json!(value); changed = true; }
                    });
                }
                let mut blink = annotation["blink"].as_bool().unwrap_or(false);
                if ui.checkbox(&mut blink, "Blink annotation").changed() { annotation["blink"] = json!(blink); changed = true; }
                if blink { changed |= time_property(ui, annotation, "blink_ms", "Blink period"); }
                let mut motion = annotation["motion"].as_str().unwrap_or("fixed").to_string();
                ui.horizontal(|ui| {
                    ui.label("Motion");
                    for (mode, label) in [("fixed", "Fixed"), ("keyframes", "Keyframes"), ("track", "Tracked positions")] {
                        if ui.selectable_value(&mut motion, mode.into(), label).changed() { annotation["motion"] = json!(motion); changed = true; }
                    }
                });
                if motion != "fixed" {
                    ui.weak("Positions use source timestamps in seconds and normalized image coordinates. Add positions in time order.");
                    if !annotation["positions"].is_array() { annotation["positions"] = json!([]); }
                    let positions = annotation["positions"].as_array_mut().expect("positions array");
                    for (position_index, position) in positions.iter_mut().enumerate() {
                        ui.push_id(position_index, |ui| {
                            changed |= time_property(ui, position, "time_ms", "Source time");
                            ui.horizontal(|ui| {
                                for key in ["x", "y"] {
                                    let mut coordinate = position[key].as_f64().unwrap_or(0.5);
                                    ui.label(key);
                                    if without_value_scroll(ui, |ui| ui.add(egui::DragValue::new(&mut coordinate).range(if motion == "track" { -2.0..=2.0 } else { 0.0..=1.0 }).speed(0.01))).changed() { position[key] = json!(coordinate); changed = true; }
                                }
                            });
                        });
                    }
                    if ui.button("Add position").clicked() {
                        let time = positions.last().and_then(|p| p["time_ms"].as_i64()).map(|time| time + 1000).unwrap_or(0);
                        positions.push(json!({"time_ms": time, "x": 0.5, "y": 0.5})); changed = true;
                    }
                    if ui.button("Remove last position").clicked() { positions.pop(); changed = true; }
                }
                ui.separator();
            });
        }
        if let Some(index) = remove { annotations.remove(index); changed = true; }
        if ui.button("Add annotation").clicked() {
            annotations.push(json!({"kind": "text", "text": "", "color": "#ffffff", "start_ms": 0, "end_ms": 5000, "x": 0.5, "y": 0.5, "dx": 0.0, "dy": 0.12, "size": 0.055, "thickness": 0.008, "weight": 900, "blink": false, "blink_ms": 250, "motion": "fixed", "positions": []}));
            changed = true;
        }
    });
    changed
}

fn points_from_value(value: &Value) -> Vec<[f64; 2]> {
    points(value)
}

fn draw_rink_polyline(
    painter: &egui::Painter,
    points: &[[f64; 2]],
    arc: bool,
    to_screen: &impl Fn([f64; 2]) -> egui::Pos2,
    color: egui::Color32,
) {
    let line: Vec<[f64; 2]> = if arc && points.len() == 3 {
        let control = [
            2.0 * points[1][0] - (points[0][0] + points[2][0]) / 2.0,
            2.0 * points[1][1] - (points[0][1] + points[2][1]) / 2.0,
        ];
        (0..=64)
            .map(|i| {
                let t = i as f64 / 64.0;
                [0, 1].map(|axis| {
                    (1.0 - t).powi(2) * points[0][axis]
                        + 2.0 * t * (1.0 - t) * control[axis]
                        + t * t * points[2][axis]
                })
            })
            .collect()
    } else {
        points.to_vec()
    };
    for segment in line.windows(2) {
        painter.line_segment(
            [to_screen(segment[0]), to_screen(segment[1])],
            egui::Stroke::new(2.0_f32, color),
        );
    }
}

fn point_canvas(
    ui: &mut egui::Ui,
    image: &egui::TextureHandle,
    dimensions: [f64; 2],
    points: &mut Vec<[f64; 2]>,
    maximum: Option<usize>,
    id: &str,
) -> bool {
    let transform_id = ui.make_persistent_id((id, "transform"));
    let selected_id = ui.make_persistent_id((id, "selected-point"));
    let mut transform = ui
        .data_mut(|data| data.get_temp::<crate::PreviewTransform>(transform_id))
        .unwrap_or_default();
    if ui.button("Fit image").clicked() {
        transform = crate::PreviewTransform::default();
    }
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width().max(1.0), 340.0),
        egui::Sense::click_and_drag(),
    );
    let fit = (rect.width() / dimensions[0] as f32).min(rect.height() / dimensions[1] as f32);
    let size = egui::vec2(dimensions[0] as f32, dimensions[1] as f32) * fit * transform.zoom;
    let image_rect = egui::Rect::from_center_size(rect.center() + transform.pan, size);
    let to_screen = |point: [f64; 2]| {
        image_rect.min
            + egui::vec2(
                point[0] as f32 / dimensions[0] as f32 * image_rect.width(),
                point[1] as f32 / dimensions[1] as f32 * image_rect.height(),
            )
    };
    let pointer = response.interact_pointer_pos();
    let near = pointer.and_then(|pointer| {
        points
            .iter()
            .enumerate()
            .find(|(_, point)| to_screen(**point).distance(pointer) < 12.0)
            .map(|(index, _)| index)
    });
    if response.drag_started() {
        let origin = ui.input(|input| input.pointer.press_origin());
        let grabbed = origin.and_then(|origin| {
            points
                .iter()
                .position(|point| to_screen(*point).distance(origin) < 12.0)
        });
        ui.data_mut(|data| data.insert_temp(selected_id, grabbed));
    }
    let selected = ui
        .data_mut(|data| data.get_temp::<Option<usize>>(selected_id))
        .flatten();
    let from_screen = |position: egui::Pos2| {
        [
            ((position.x - image_rect.left()) / image_rect.width() * dimensions[0] as f32)
                .clamp(0.0, dimensions[0] as f32 - 1.0) as f64,
            ((position.y - image_rect.top()) / image_rect.height() * dimensions[1] as f32)
                .clamp(0.0, dimensions[1] as f32 - 1.0) as f64,
        ]
    };
    let mut changed = false;
    if response.dragged() && selected.is_some() {
        if let (Some(index), Some(position)) = (selected, pointer) {
            if let Some(point) = points.get_mut(index) {
                *point = from_screen(position);
                changed = true;
            }
        }
    } else {
        crate::interact_preview(ui, &response, &mut transform);
    }
    if response.clicked()
        && !response.double_clicked()
        && near.is_none()
        && maximum.is_none_or(|maximum| points.len() < maximum)
    {
        if let Some(position) = pointer.filter(|position| image_rect.contains(*position)) {
            points.push(from_screen(position));
            changed = true;
        }
    }
    if response.drag_stopped() {
        ui.data_mut(|data| data.remove::<Option<usize>>(selected_id));
    }
    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 0.0, egui::Color32::BLACK);
    painter.image(
        image.id(),
        image_rect,
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
    let overlays = ui
        .data_mut(|data| data.get_temp::<Vec<Value>>(ui.id().with((id, "overlays"))))
        .unwrap_or_default();
    for mark in &overlays {
        let vertices = points_from_value(&mark["pixels"]);
        let color = if mark["enabled"] == false {
            egui::Color32::LIGHT_RED
        } else {
            egui::Color32::LIGHT_GREEN
        };
        if let Some(first) = vertices.first() {
            let position = to_screen(*first);
            if vertices.len() == 1 {
                painter.circle_filled(position, 4.0, color);
            }
            painter.text(
                position,
                egui::Align2::LEFT_BOTTOM,
                human_label(&text(&mark["feature"])),
                egui::FontId::proportional(12.0),
                color,
            );
        }
        draw_rink_polyline(&painter, &vertices, mark["arc"] == true, &to_screen, color);
    }
    if id == "rink-source-marks" && maximum != Some(1) {
        draw_rink_polyline(
            &painter,
            points,
            maximum == Some(3),
            &to_screen,
            egui::Color32::YELLOW,
        );
    }
    for (index, point) in points.iter().enumerate() {
        let position = to_screen(*point);
        painter.circle_filled(position, 5.0, egui::Color32::YELLOW);
        painter.text(
            position + egui::vec2(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            (index + 1).to_string(),
            egui::FontId::proportional(16.0),
            egui::Color32::YELLOW,
        );
        if index > 0 && maximum == Some(4) {
            painter.line_segment(
                [to_screen(points[index - 1]), position],
                egui::Stroke::new(2.0_f32, egui::Color32::YELLOW),
            );
        }
    }
    if points.len() == 4 && maximum == Some(4) {
        painter.line_segment(
            [to_screen(points[3]), to_screen(points[0])],
            egui::Stroke::new(2.0_f32, egui::Color32::YELLOW),
        );
    }
    ui.data_mut(|data| data.insert_temp(transform_id, transform));
    changed
}

fn rink_canvas(ui: &mut egui::Ui, dimensions: [f64; 2], points: &mut [[f64; 2]], active: usize) {
    let width = ui.available_width().min(700.0);
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(width, width * dimensions[1] as f32 / dimensions[0] as f32),
        egui::Sense::click_and_drag(),
    );
    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 24.0, egui::Color32::from_rgb(225, 238, 244));
    for (fraction, color) in [
        (0.33, egui::Color32::BLUE),
        (0.5, egui::Color32::RED),
        (0.67, egui::Color32::BLUE),
    ] {
        let x = rect.left() + rect.width() * fraction;
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(2.0_f32, color),
        );
    }
    painter.circle_stroke(
        rect.center(),
        rect.height() * 0.18,
        egui::Stroke::new(2.0_f32, egui::Color32::BLUE),
    );
    if (response.clicked() || response.dragged()) && active < points.len() {
        if let Some(position) = response.interact_pointer_pos() {
            points[active] = [
                ((position.x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f64 * dimensions[0],
                ((position.y - rect.top()) / rect.height()).clamp(0.0, 1.0) as f64 * dimensions[1],
            ];
        }
    }
    for (index, point) in points.iter().enumerate() {
        let position = rect.min
            + egui::vec2(
                (point[0] / dimensions[0]) as f32 * rect.width(),
                (point[1] / dimensions[1]) as f32 * rect.height(),
            );
        let color = if index == active {
            egui::Color32::RED
        } else {
            egui::Color32::BLACK
        };
        painter.circle_filled(position, 5.0, color);
        painter.text(
            position + egui::vec2(8.0, 8.0),
            egui::Align2::LEFT_TOP,
            (index + 1).to_string(),
            egui::FontId::proportional(15.0),
            color,
        );
    }
}

fn annotation_box_rect(rect: egui::Rect, center: egui::Pos2, annotation: &Value) -> egui::Rect {
    let size = egui::vec2(
        annotation["dx"].as_f64().unwrap_or(0.0).abs().max(0.02) as f32 * rect.width(),
        annotation["dy"].as_f64().unwrap_or(0.12).abs().max(0.02) as f32 * rect.height(),
    );
    egui::Rect::from_center_size(center, size)
}

fn paint_annotation(
    painter: &egui::Painter,
    rect: egui::Rect,
    annotation: &Value,
    timestamp: f64,
    selected: bool,
) {
    let start = annotation["start_ms"].as_f64().unwrap_or(0.0);
    let end = annotation["end_ms"].as_f64().unwrap_or(f64::MAX);
    if !selected && !(start..=end).contains(&timestamp) {
        return;
    }
    let mut x = annotation["x"].as_f64().unwrap_or(0.5);
    let mut y = annotation["y"].as_f64().unwrap_or(0.5);
    if annotation["motion"]
        .as_str()
        .is_some_and(|mode| mode != "fixed")
    {
        let positions = collection(&annotation["positions"]);
        if let Some(first) = positions.first() {
            x = first["x"].as_f64().unwrap_or(x);
            y = first["y"].as_f64().unwrap_or(y);
        }
        for pair in positions.windows(2) {
            let before = pair[0]["time_ms"].as_f64().unwrap_or(0.0);
            let after = pair[1]["time_ms"].as_f64().unwrap_or(before + 1.0);
            if timestamp >= before {
                let fraction = ((timestamp - before) / (after - before)).clamp(0.0, 1.0);
                x = pair[0]["x"].as_f64().unwrap_or(x) * (1.0 - fraction)
                    + pair[1]["x"].as_f64().unwrap_or(x) * fraction;
                y = pair[0]["y"].as_f64().unwrap_or(y) * (1.0 - fraction)
                    + pair[1]["y"].as_f64().unwrap_or(y) * fraction;
            }
        }
    }
    let origin = rect.min + egui::vec2(x as f32 * rect.width(), y as f32 * rect.height());
    let delta = egui::vec2(
        annotation["dx"].as_f64().unwrap_or(0.0) as f32 * rect.width(),
        annotation["dy"].as_f64().unwrap_or(0.12) as f32 * rect.height(),
    );
    let color = annotation["color"]
        .as_str()
        .and_then(|value| value.strip_prefix('#'))
        .and_then(|hex| u32::from_str_radix(hex, 16).ok())
        .map(|value| egui::Color32::from_rgb((value >> 16) as u8, (value >> 8) as u8, value as u8))
        .unwrap_or(egui::Color32::YELLOW);
    let stroke = egui::Stroke::new(
        (annotation["thickness"].as_f64().unwrap_or(0.008) as f32 * rect.height()).max(1.0),
        color,
    );
    match annotation["kind"].as_str().unwrap_or("text") {
        "arrow" => painter.arrow(origin, delta, stroke),
        "box" => {
            painter.rect_stroke(annotation_box_rect(rect, origin, annotation), 0.0, stroke);
        }
        _ => {
            painter.text(
                origin,
                egui::Align2::CENTER_CENTER,
                annotation["text"].as_str().unwrap_or(""),
                egui::FontId::proportional(
                    annotation["size"].as_f64().unwrap_or(0.055) as f32 * rect.height(),
                ),
                color,
            );
        }
    }
    if selected {
        painter.circle_stroke(
            origin,
            8.0,
            egui::Stroke::new(2.0_f32, egui::Color32::WHITE),
        );
    }
}

fn draw_telemetry(ui: &mut egui::Ui, telemetry: &Value) {
    if let Some(message) = telemetry["message"].as_str() {
        ui.weak(message);
    }
    let mut roles: Vec<String> = ["baseline", "trial"]
        .iter()
        .flat_map(|side| {
            telemetry[*side]["series"]
                .as_object()
                .into_iter()
                .flat_map(|series| series.keys().cloned())
        })
        .collect();
    roles.sort();
    roles.dedup();
    for role in roles {
        ui.strong(format!(
            "{role} camera centre · baseline blue / candidate orange"
        ));
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width().min(900.0), 220.0),
            egui::Sense::hover(),
        );
        let plot = rect.shrink2(egui::vec2(24.0, 12.0));
        let painter = ui.painter().with_clip_rect(rect);
        painter.rect_filled(plot, 0.0, egui::Color32::from_gray(20));
        let maximum = ["baseline", "trial"]
            .iter()
            .flat_map(|side| collection(&telemetry[*side]["series"][&role]))
            .filter_map(|point| point["time_seconds"].as_f64())
            .fold(1.0_f64, f64::max);
        for (side, color) in [
            ("baseline", egui::Color32::LIGHT_BLUE),
            ("trial", egui::Color32::from_rgb(250, 170, 70)),
        ] {
            let points = collection(&telemetry[side]["series"][&role]);
            for axis in ["x", "y"] {
                let line: Vec<_> = points
                    .iter()
                    .filter_map(|point| {
                        Some(egui::pos2(
                            plot.left()
                                + (point["time_seconds"].as_f64()? / maximum) as f32 * plot.width(),
                            plot.bottom() - point[axis].as_f64()? as f32 * plot.height(),
                        ))
                    })
                    .collect();
                if line.len() >= 2 {
                    painter.add(egui::Shape::line(
                        line,
                        egui::Stroke::new(if axis == "x" { 2.0_f32 } else { 1.0_f32 }, color),
                    ));
                }
            }
        }
        painter.text(
            plot.left_bottom(),
            egui::Align2::LEFT_BOTTOM,
            "0",
            egui::FontId::proportional(11.0),
            egui::Color32::GRAY,
        );
        painter.text(
            plot.right_bottom(),
            egui::Align2::RIGHT_BOTTOM,
            format!("{maximum:.1}s · x thick / y thin"),
            egui::FontId::proportional(11.0),
            egui::Color32::GRAY,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input_tests::{click_events, run_frame, text_positions};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture {
        directory: PathBuf,
        workspace: WorkspaceUi,
    }
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let directory = std::env::temp_dir().join(format!(
                "hm-workspace-input-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&directory).unwrap();
            write_json_atomic(&directory.join("workspace.json"), &json!({"current_game": "/games/game-a", "games": [{"id": "game-a", "path": "/games/game-a"}, {"id": "game-b", "path": "/games/game-b"}]})).unwrap();
            let mut workspace = WorkspaceUi::new(directory.clone());
            workspace.poll();
            Self {
                directory,
                workspace,
            }
        }
        fn commands(&self) -> Vec<Value> {
            let mut paths: Vec<_> = fs::read_dir(self.directory.join("commands"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect();
            paths.sort();
            paths.iter().map(|path| read_json(path).unwrap()).collect()
        }
        fn respond(&mut self, result: Value, ok: bool) {
            let id = self.workspace.pending.keys().next().unwrap().clone();
            write_json_atomic(&self.directory.join("results").join(format!("{id}.json")), &json!({"id": id, "ok": ok, "result": result, "error": if ok { "" } else { "Invalid requested edit" }})).unwrap();
            self.workspace.last_poll = Instant::now() - Duration::from_secs(1);
            self.workspace.poll();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.directory).unwrap();
        }
    }

    #[test]
    fn runs_archive_selection_updates_the_render_document() {
        let mut fixture = Fixture::new();
        fixture.workspace.page = Page::Run;
        fixture.workspace.highlight_document["archive_path"] = json!("/old.mp4");
        fixture.workspace.track_run = "old-run".into();
        fixture.workspace.track_choices = json!({"choices":[1]});
        fixture.workspace.snapshot["archives"] = json!([{"path":"/games/game-a/archive-2.mp4"}]);
        let context = egui::Context::default();
        run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        run_frame(
            &context,
            click_events(text_positions(&output, "Edit highlights")[0]),
            |ctx| fixture.workspace.draw_content(ctx),
        );
        assert_eq!(
            fixture.workspace.highlight_source,
            "/games/game-a/archive-2.mp4"
        );
        assert_eq!(
            fixture.workspace.highlight_document["archive_path"],
            "/games/game-a/archive-2.mp4"
        );
        assert!(fixture.workspace.track_run.is_empty());
        assert!(fixture.workspace.track_choices.is_null());
        assert!(fixture.workspace.highlight_dirty);
    }

    #[test]
    fn boxes_use_centered_absolute_dimensions_and_minimum_size() {
        let canvas = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(1000.0, 500.0));
        let center = egui::pos2(300.0, 200.0);
        let rectangle = annotation_box_rect(canvas, center, &json!({"dx":-0.2,"dy":0.4}));
        assert_eq!(rectangle.min, egui::pos2(200.0, 100.0));
        assert_eq!(rectangle.max, egui::pos2(400.0, 300.0));
        let minimum = annotation_box_rect(canvas, center, &json!({"dx":0,"dy":0}));
        assert_eq!(minimum.size(), egui::vec2(20.0, 10.0));
    }

    #[test]
    fn lens_and_sync_mutations_refresh_revision_for_the_next_edit() {
        for operation in ["calibration.lens", "calibration.reset_sync"] {
            let mut fixture = Fixture::new();
            fixture.workspace.calibration = json!({"revision":"old", "settings":{}});
            fixture.workspace.accept_result(operation, &json!({}), json!({"revision":"new", "settings":{"stitching.horizontal_fov":120}, "catalog":[]}));
            assert_eq!(fixture.workspace.calibration["revision"], "new");
            assert_eq!(fixture.workspace.settings["stitching.horizontal_fov"], 120);
            assert!(fixture.workspace.pending.is_empty());
        }
    }

    #[test]
    fn game_selection_and_import_buttons_send_supported_operations() {
        let mut fixture = Fixture::new();
        fixture.workspace.page = Page::Games;
        fixture.workspace.source_left = "/recordings/left.mp4".into();
        fixture.workspace.source_right = "/recordings/right.mp4".into();
        let context = egui::Context::default();
        run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        let game = text_positions(&output, "game-b")[0];
        run_frame(&context, click_events(game), |ctx| {
            fixture.workspace.draw_content(ctx)
        });
        let commands = fixture.commands();
        assert_eq!(commands[0]["operation"], "games.select");
        assert_eq!(commands[0]["args"], json!({"game_id": "game-b"}));
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        let import = text_positions(&output, "Import recordings")[0];
        run_frame(&context, click_events(import), |ctx| {
            fixture.workspace.draw_content(ctx)
        });
        let commands = fixture.commands();
        assert_eq!(commands.len(), 3);
        assert_eq!(
            commands[1]["args"],
            json!({"role": "left", "paths": ["/recordings/left.mp4"], "mode": "link"})
        );
        assert_eq!(
            commands[2]["args"],
            json!({"role": "right", "paths": ["/recordings/right.mp4"], "mode": "link"})
        );
    }

    #[test]
    fn setup_applies_only_changed_keys_and_preserves_inherited_values() {
        let mut fixture = Fixture::new();
        fixture.workspace.calibration = json!({"revision": "version-1", "settings": {"stitching.flag": false, "stitching.inherited": null}, "catalog": [{"key": "stitching.flag", "label": "Flag", "kind": "bool"}, {"key": "stitching.inherited", "label": "Inherited", "kind": "number", "min": 0.01}]});
        fixture.workspace.load_calibration_draft();
        fixture.workspace.page = Page::Setup;
        let context = egui::Context::default();
        run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        assert!(!fixture.workspace.settings_dirty);
        assert_eq!(
            fixture.workspace.settings["stitching.inherited"],
            Value::Null
        );
        run_frame(
            &context,
            click_events(text_positions(&output, "Enabled")[0]),
            |ctx| fixture.workspace.draw_content(ctx),
        );
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        run_frame(
            &context,
            click_events(text_positions(&output, "Apply settings")[0]),
            |ctx| fixture.workspace.draw_content(ctx),
        );
        let command = &fixture.commands()[0];
        assert_eq!(command["operation"], "calibration.update");
        assert_eq!(
            command["args"],
            json!({"changes": {"stitching.flag": true}, "expected_revision": "version-1"})
        );
    }

    #[test]
    fn highlight_title_card_and_event_mode_emit_complete_backend_documents() {
        let mut fixture = Fixture::new();
        fixture.workspace.page = Page::Highlights;
        let context = egui::Context::default();
        run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        run_frame(
            &context,
            click_events(text_positions(&output, "Add title card")[0]),
            |ctx| fixture.workspace.draw_content(ctx),
        );
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        run_frame(
            &context,
            click_events(text_positions(&output, "Save highlights")[0]),
            |ctx| fixture.workspace.draw_content(ctx),
        );
        let commands = fixture.commands();
        assert_eq!(commands[0]["operation"], "highlights.save");
        assert_eq!(commands[0]["args"]["plan"]["schema"], 2);
        assert_eq!(
            commands[0]["args"]["plan"]["intervals"][0]["card"]["duration_ms"],
            3000
        );
        assert_eq!(commands[0]["args"]["plan"]["intervals"][0]["is_card"], true);
    }

    #[test]
    fn point_canvas_clicks_and_drags_use_source_image_coordinates() {
        let context = egui::Context::default();
        let texture = context.load_texture(
            "test-points",
            egui::ColorImage::new([20, 10], egui::Color32::BLACK),
            egui::TextureOptions::LINEAR,
        );
        let mut points = Vec::new();
        let output = run_frame(&context, vec![], |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                point_canvas(
                    ui,
                    &texture,
                    [2000.0, 1000.0],
                    &mut points,
                    Some(4),
                    "test-points",
                );
            });
        });
        let image_rect = output
            .shapes
            .iter()
            .find_map(|shape| match &shape.shape {
                egui::epaint::Shape::Mesh(mesh) if mesh.texture_id == texture.id() => {
                    Some(mesh.calc_bounds())
                }
                _ => None,
            })
            .unwrap();
        run_frame(&context, click_events(image_rect.center()), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                point_canvas(
                    ui,
                    &texture,
                    [2000.0, 1000.0],
                    &mut points,
                    Some(4),
                    "test-points",
                );
            });
        });
        assert_eq!(points.len(), 1);
        assert!((points[0][0] - 1000.0).abs() < 0.01);
        assert!((points[0][1] - 500.0).abs() < 0.01);
        run_frame(
            &context,
            vec![egui::Event::PointerButton {
                pos: image_rect.center(),
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            }],
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    point_canvas(
                        ui,
                        &texture,
                        [2000.0, 1000.0],
                        &mut points,
                        Some(4),
                        "test-points",
                    );
                });
            },
        );
        run_frame(
            &context,
            vec![egui::Event::PointerMoved(
                image_rect.center() + egui::vec2(68.0, 34.0),
            )],
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    point_canvas(
                        ui,
                        &texture,
                        [2000.0, 1000.0],
                        &mut points,
                        Some(4),
                        "test-points",
                    );
                });
            },
        );
        assert!((points[0][0] - 1200.0).abs() < 0.01, "{points:?}");
        assert!((points[0][1] - 600.0).abs() < 0.01, "{points:?}");
    }

    #[test]
    fn command_failures_remain_visible_and_media_replies_match_the_selected_file() {
        let mut fixture = Fixture::new();
        fixture
            .workspace
            .command("calibration.update", json!({"changes": {}}));
        fixture.respond(Value::Null, false);
        assert!(fixture
            .workspace
            .error
            .as_deref()
            .unwrap()
            .contains("Invalid requested edit"));
        fixture.workspace.media_path = "/old.mp4".into();
        fixture
            .workspace
            .command("media.frame", json!({"path": "/old.mp4", "time_ms": 0}));
        fixture.workspace.media_path = "/new.mp4".into();
        fixture.respond(
            json!({"path": "old.jpg", "duration_ms": 1000, "fps": 30}),
            true,
        );
        assert!(fixture.workspace.media_frame.is_null());
        assert!(fixture.workspace.error.is_some());
    }

    #[test]
    fn live_recording_edits_send_only_touched_keys_and_wait_for_runtime_ack() {
        let mut fixture = Fixture::new();
        fixture.workspace.page = Page::Run;
        fixture.workspace.snapshot["recording_outputs"] = json!({"state":"running", "capabilities":["archive_program"], "active":{"archive_program":false}});
        fixture.workspace.snapshot["output_routes"] =
            json!({"state":"running", "capabilities":[], "config":{}});
        let context = egui::Context::default();
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        run_frame(
            &context,
            click_events(text_positions(&output, "Live publishing & recording")[0]),
            |ctx| fixture.workspace.draw_content(ctx),
        );
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        run_frame(
            &context,
            click_events(text_positions(&output, "Record Program")[0]),
            |ctx| fixture.workspace.draw_content(ctx),
        );
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        run_frame(
            &context,
            click_events(text_positions(&output, "Apply live outputs")[0]),
            |ctx| fixture.workspace.draw_content(ctx),
        );
        let commands = fixture.commands();
        assert_eq!(commands[0]["operation"], "outputs.update");
        assert_eq!(
            commands[0]["args"],
            json!({"changes":{"archive_program":true}})
        );
        fixture.respond(
            json!({"request_id":"route-1", "state":"queued for the next output frame"}),
            true,
        );
        assert_eq!(fixture.workspace.output_request.as_deref(), Some("route-1"));
        assert!(fixture.workspace.feedback.starts_with("Queued"));
        assert!(!fixture.workspace.output_changes.is_empty());
        fixture.workspace.snapshot["output_routes"] = json!({"request_id":"unrelated", "ok":true});
        fixture.workspace.observe_output_result();
        assert!(fixture.workspace.output_request.is_some());
        fixture.workspace.snapshot["output_routes"] = json!({"request_id":"route-1", "ok":true});
        fixture.workspace.observe_output_result();
        assert!(fixture.workspace.output_changes.is_empty());
        assert_eq!(fixture.workspace.output_status, "Live outputs updated");
        fixture.workspace.output_request = Some("route-2".into());
        fixture
            .workspace
            .output_changes
            .insert("rtsp_enabled".into(), json!(true));
        fixture.workspace.snapshot["output_routes"] =
            json!({"request_id":"route-2", "ok":false, "error":"Port already in use"});
        fixture.workspace.observe_output_result();
        assert!(fixture
            .workspace
            .error
            .as_deref()
            .unwrap()
            .contains("Port already in use"));
        assert!(!fixture.workspace.output_changes.is_empty());
    }

    #[test]
    fn candidate_runs_send_bounded_source_times() {
        for replay in [Value::Null, json!({"run_id":"saved-run"})] {
            let mut fixture = Fixture::new();
            fixture.workspace.page = Page::Experiments;
            fixture.workspace.snapshot["experiments"] =
                json!([{"id":"candidate-a", "status":"created", "replay":replay}]);
            fixture.workspace.experiment_start = 3661.125;
            fixture.workspace.experiment_duration = 12.5;
            let context = egui::Context::default();
            let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
            run_frame(
                &context,
                click_events(text_positions(&output, "Run candidate")[0]),
                |ctx| fixture.workspace.draw_content(ctx),
            );
            let expected = if replay.is_object() {
                json!([])
            } else {
                json!(["--start-time", "01:01:01.125", "-t=12.5"])
            };
            assert_eq!(
                fixture.commands()[0]["args"],
                json!({"experiment_id":"candidate-a", "arguments":expected})
            );
        }
        assert!(bounded_run_arguments(0.0, 0.0).is_empty());
    }

    #[test]
    fn camera_candidates_bind_replay_to_a_recording_run_source_and_range() {
        let mut fixture = Fixture::new();
        fixture.workspace.page = Page::Experiments;
        fixture.workspace.experiment_name = "Slower pan".into();
        fixture.workspace.experiment_kind = "camera".into();
        fixture.workspace.experiment_replay = true;
        fixture.workspace.replay_recording = "/games/game-a/game-a_telemetry-1.db".into();
        fixture.workspace.replay_run = "run-a".into();
        fixture.workspace.replay_source = "/games/game-a/source.mp4".into();
        fixture.workspace.replay_start = 1.25;
        fixture.workspace.replay_end = 5.75;
        fixture.workspace.replay_sources = json!({"runs":[{"run_id":"run-a","producer":"hmtrack","sample_count":180,"start_ms":0,"end_ms":6000}],"sources":[]});
        fixture
            .workspace
            .experiment_changes
            .insert("rink.camera.max_speed_ratio_x".into(), json!(0.7));
        fixture.workspace.calibration = json!({"catalog":[{"key":"rink.camera.max_speed_ratio_x","kind":"number","label":"Pan speed","min":0,"max":10}]});
        let context = egui::Context::default();
        run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        let output = run_frame(&context, vec![], |ctx| fixture.workspace.draw_content(ctx));
        run_frame(
            &context,
            click_events(text_positions(&output, "Create candidate")[0]),
            |ctx| fixture.workspace.draw_content(ctx),
        );
        let command = &fixture.commands()[0];
        assert_eq!(command["operation"], "experiments.create");
        assert_eq!(
            command["args"]["replay"],
            json!({"recording_path":"/games/game-a/game-a_telemetry-1.db","run_id":"run-a","source_path":"/games/game-a/source.mp4","start_ms":1250,"end_ms":5750})
        );
        assert_eq!(
            command["args"]["changes"]["rink.camera.max_speed_ratio_x"],
            0.7
        );
    }

    #[test]
    fn scoreboard_proposal_is_a_revision_checked_unsaved_draft() {
        let mut fixture = Fixture::new();
        fixture.workspace.calibration = json!({"revision":"revision-a"});
        fixture.workspace.accept_result("calibration.scoreboard-proposal", &json!({}), json!({"revision":"revision-a", "points":[[1,2],[3,4],[5,6],[7,8]], "message":"Suggested corners"}));
        assert_eq!(fixture.workspace.geometry.scoreboard.len(), 4);
        assert!(fixture.workspace.geometry.dirty);
        assert!(fixture.workspace.pending.is_empty());
        fixture.workspace.accept_result(
            "calibration.scoreboard-proposal",
            &json!({}),
            json!({"revision":"stale", "points":[]}),
        );
        assert_eq!(fixture.workspace.geometry.scoreboard.len(), 4);
        assert!(fixture
            .workspace
            .error
            .as_deref()
            .unwrap()
            .contains("stale"));
    }

    #[test]
    fn tracked_annotation_capture_uses_source_time_and_preserves_offscreen_coordinates() {
        let mut fixture = Fixture::new();
        fixture.workspace.highlight_source = "/games/game-a/archive-1.mp4".into();
        fixture.workspace.media_time = 2.0;
        fixture.workspace.highlight_document = json!({"archive_offset_ms":1000,"intervals":[{"annotations":[{"kind":"text","text":"Player","start_ms":1000,"end_ms":5000,"motion":"fixed"}]}]});
        let context = egui::Context::default();
        let mut draw = |ctx: &egui::Context| {
            egui::CentralPanel::default().show(ctx, |ui| fixture.workspace.draw_track_capture(ui));
        };
        let output = run_frame(&context, vec![], &mut draw);
        run_frame(
            &context,
            click_events(text_positions(&output, "Follow a recorded player")[0]),
            &mut draw,
        );
        let output = run_frame(&context, vec![], &mut draw);
        run_frame(
            &context,
            click_events(text_positions(&output, "Find players at preview time")[0]),
            &mut draw,
        );
        let command = &fixture.commands()[0];
        assert_eq!(command["operation"], "highlights.track_choices");
        assert_eq!(command["args"]["time_ms"], 3000);
        fixture.respond(json!({"run_id":"recorded-run", "choices":[{"tracking_id":7,"geometry_id":3,"seek_epoch":1,"reset_epoch":0,"source_id":2,"box":{"x":0.2,"y":0.3}}]}), true);
        run_frame(&context, vec![], |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| fixture.workspace.draw_track_capture(ui));
        });
        let output = run_frame(&context, vec![], |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| fixture.workspace.draw_track_capture(ui));
        });
        run_frame(
            &context,
            click_events(text_positions(&output, "Capture player motion")[0]),
            |ctx| {
                egui::CentralPanel::default()
                    .show(ctx, |ui| fixture.workspace.draw_track_capture(ui));
            },
        );
        let commands = fixture.commands();
        assert_eq!(commands[1]["operation"], "highlights.capture_track");
        assert_eq!(commands[1]["args"]["choice"]["tracking_id"], 7);
        assert_eq!(commands[1]["args"]["run_id"], "recorded-run");
        fixture.respond(json!({"motion":"track","positions":[{"time_ms":1000,"x":-0.25,"y":1.2}],"tracking_id":7,"run_id":"recorded-run"}), true);
        let annotation = &fixture.workspace.highlight_document["intervals"][0]["annotations"][0];
        assert_eq!(annotation["text"], "Player");
        assert_eq!(annotation["motion"], "track");
        assert_eq!(annotation["positions"][0]["x"], -0.25);
        assert!(fixture.workspace.highlight_dirty);
    }
}
