use super::*;
use egui::{Event, Modifiers, MouseWheelUnit, PointerButton, Pos2, Rect, Vec2};
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture {
    directory: PathBuf,
    app: HmUiApp,
}

#[test]
fn workspace_startup_waits_for_missing_files_but_surfaces_malformed_files() {
    let mut fixture = Fixture::new();
    fixture.app.workspace = Some(workspace::WorkspaceUi::new(fixture.directory.clone()));
    fs::remove_file(&fixture.app.spec_path).unwrap();
    fixture.app.poll_spec();
    assert_eq!(fixture.app.status, "Preparing pipeline…");
    let runtime = fixture.directory.join("runtime.json");
    fixture.app.spec.runtime_path = Some(runtime.clone());
    fixture.app.poll_runtime();
    assert!(fixture.app.runtime_error.is_none());
    fs::write(&fixture.app.spec_path, "{broken").unwrap();
    fixture.app.last_spec_poll = UNIX_EPOCH;
    fixture.app.poll_spec();
    assert!(fixture.app.status.starts_with("Spec error:"));
    fs::write(runtime, "{broken").unwrap();
    fixture.app.last_runtime_poll = UNIX_EPOCH;
    fixture.app.poll_runtime();
    assert!(fixture.app.runtime_error.is_some());
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "hm-ui-input-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let spec = serde_json::json!({
            "version": 3,
            "title": "Test runtime",
            "action_ack_path": directory.join("ack.json"),
            "previews": [
                {"name": "Stitched", "path": "stitched.jpg"},
                {"name": "Final", "path": "final.jpg"}
            ],
            "windows": [{"name": "camera", "controls": [
                {"name": "Stitched_Gain_x100", "max_value": 300, "value": 100, "view": "Stitched", "group": "Color"},
                {"name": "Exposure_EV_x10", "max_value": 80, "value": 40, "view": "Final", "group": "Color"}
            ]}]
        });
        let spec_path = directory.join("spec.json");
        write_json_atomic(&spec_path, &spec).unwrap();
        let app = HmUiApp::new(spec_path, directory.join("state.json"), "Test".into());
        Self { directory, app }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}

pub(crate) fn run_frame(
    context: &egui::Context,
    events: Vec<Event>,
    draw: impl FnMut(&egui::Context),
) -> egui::FullOutput {
    context.run(
        egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 650.0))),
            events,
            ..Default::default()
        },
        draw,
    )
}

pub(crate) fn click_events(position: Pos2) -> Vec<Event> {
    vec![
        Event::PointerMoved(position),
        Event::PointerButton {
            pos: position,
            button: PointerButton::Primary,
            pressed: true,
            modifiers: Modifiers::NONE,
        },
        Event::PointerButton {
            pos: position,
            button: PointerButton::Primary,
            pressed: false,
            modifiers: Modifiers::NONE,
        },
    ]
}

fn wheel(position: Pos2, unit: MouseWheelUnit, delta: f32) -> Vec<Event> {
    vec![
        Event::PointerMoved(position),
        Event::MouseWheel {
            unit,
            delta: egui::vec2(0.0, delta),
            modifiers: Modifiers::NONE,
        },
    ]
}

pub(crate) fn text_positions(output: &egui::FullOutput, needle: &str) -> Vec<Pos2> {
    fn visit(shape: &egui::epaint::Shape, needle: &str, positions: &mut Vec<Pos2>) {
        match shape {
            egui::epaint::Shape::Text(text) if text.galley.text() == needle => {
                positions.push(text.pos + text.galley.size() * 0.5);
            }
            egui::epaint::Shape::Vec(shapes) => {
                for shape in shapes {
                    visit(shape, needle, positions);
                }
            }
            _ => {}
        }
    }
    let mut positions = Vec::new();
    for shape in &output.shapes {
        visit(&shape.shape, needle, &mut positions);
    }
    positions.sort_by(|left, right| left.x.total_cmp(&right.x));
    positions
}

#[test]
fn preview_and_control_clicks_keep_view_selection_and_wire_names_together() {
    let mut fixture = Fixture::new();
    let context = egui::Context::default();
    run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    let output = run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    let program = text_positions(&output, "Program")[0];
    run_frame(&context, click_events(program), |ctx| fixture.app.draw(ctx));
    assert_eq!(fixture.app.selected_preview, 1);
    assert_eq!(
        control_pages(&fixture.app.spec)[fixture.app.selected_page].0,
        "Final"
    );
    let state: UiState =
        serde_json::from_str(&fs::read_to_string(&fixture.app.state_path).unwrap()).unwrap();
    assert_eq!(state.selected_preview.as_deref(), Some("Final"));
    assert_eq!(state.spec_version, 3);

    let output = run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    let stitched = *text_positions(&output, "Stitched").last().unwrap();
    run_frame(&context, click_events(stitched), |ctx| {
        fixture.app.draw(ctx)
    });
    assert_eq!(fixture.app.selected_preview, 0);
    assert_eq!(
        control_pages(&fixture.app.spec)[fixture.app.selected_page].0,
        "Stitched"
    );
}

#[test]
fn maximize_button_hides_controls_and_escape_restores_them() {
    let mut fixture = Fixture::new();
    let context = egui::Context::default();
    run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    let output = run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    let maximize = text_positions(&output, "Maximize preview")[0];
    run_frame(&context, click_events(maximize), |ctx| {
        fixture.app.draw(ctx)
    });
    let output = run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    assert!(fixture.app.preview_maximized);
    assert!(text_positions(&output, "Stitched · Color").is_empty());
    assert!(!text_positions(&output, "Restore preview").is_empty());
    run_frame(
        &context,
        vec![Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        }],
        |ctx| fixture.app.draw(ctx),
    );
    assert!(!fixture.app.preview_maximized);
    let output = run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    assert!(!text_positions(&output, "Stitched · Color").is_empty());
}

#[test]
fn wheel_over_value_widgets_scrolls_without_editing_values() {
    for unit in [MouseWheelUnit::Point, MouseWheelUnit::Line] {
        for target in 0..4 {
            let context = egui::Context::default();
            let control: ControlSpec =
                serde_json::from_str(r#"{"name":"Gain_x100","max_value":300,"value":100}"#)
                    .unwrap();
            let mut value = 100;
            let mut checked = true;
            let mut selected = 1;
            let mut rectangles = [Rect::NOTHING; 4];
            let mut offset = 0.0;
            let mut draw = |ctx: &egui::Context| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let output = egui::ScrollArea::vertical()
                        .max_height(240.0)
                        .show(ui, |ui| {
                            rectangles[0] = without_value_scroll(ui, |ui| {
                                ui.add(egui::Slider::new(&mut value, 0..=300).show_value(false))
                            })
                            .rect;
                            rectangles[1] = numeric_editor(ui, &control, &mut value).rect;
                            rectangles[2] = ui.checkbox(&mut checked, "Enabled").rect;
                            rectangles[3] = egui::ComboBox::from_id_salt("choice")
                                .selected_text(selected.to_string())
                                .show_ui(ui, |ui| {
                                    for index in 0..3 {
                                        ui.selectable_value(
                                            &mut selected,
                                            index,
                                            index.to_string(),
                                        );
                                    }
                                })
                                .response
                                .rect;
                            ui.add_space(1000.0);
                        });
                    offset = output.state.offset.y;
                });
            };
            run_frame(&context, vec![], &mut draw);
            drop(draw);
            let position = rectangles[target].center();
            let _ = run_frame(&context, wheel(position, unit, -60.0), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let output = egui::ScrollArea::vertical()
                        .max_height(240.0)
                        .show(ui, |ui| {
                            without_value_scroll(ui, |ui| {
                                ui.add(egui::Slider::new(&mut value, 0..=300).show_value(false))
                            });
                            numeric_editor(ui, &control, &mut value);
                            ui.checkbox(&mut checked, "Enabled");
                            egui::ComboBox::from_id_salt("choice")
                                .selected_text(selected.to_string())
                                .show_ui(ui, |ui| {
                                    for index in 0..3 {
                                        ui.selectable_value(
                                            &mut selected,
                                            index,
                                            index.to_string(),
                                        );
                                    }
                                });
                            ui.add_space(1000.0);
                        });
                    offset = output.state.offset.y;
                });
            });
            assert_eq!(value, 100, "target {target}, unit {unit:?}");
            assert!(checked);
            assert_eq!(selected, 1);
            assert!(
                offset > 0.0,
                "wheel did not reach ScrollArea for target {target}, unit {unit:?}"
            );
        }
    }
}

#[test]
fn typed_numeric_values_convert_to_wire_units_including_reversed_degrees() {
    for (json, typed, expected) in [
        (
            r#"{"name":"Exposure_EV_x10","max_value":80,"value":40}"#,
            "1.5",
            55,
        ),
        (
            r#"{"name":"Gain","max_value":300,"value":100,"divisor":100,"offset":0,"unit":"×"}"#,
            "1.25",
            125,
        ),
        (
            r#"{"name":"Angle","max_value":180,"value":90,"divisor":-1,"offset":90,"unit":"deg"}"#,
            "-15",
            105,
        ),
    ] {
        let control: ControlSpec = serde_json::from_str(json).unwrap();
        let mut value = control.value;
        let context = egui::Context::default();
        let mut rectangle = Rect::NOTHING;
        run_frame(&context, vec![], |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                rectangle = numeric_editor(ui, &control, &mut value).rect;
            });
        });
        run_frame(&context, click_events(rectangle.center()), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                numeric_editor(ui, &control, &mut value);
            });
        });
        run_frame(&context, vec![Event::Text(typed.into())], |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                numeric_editor(ui, &control, &mut value);
            });
        });
        run_frame(
            &context,
            vec![Event::Key {
                key: egui::Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::NONE,
            }],
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    numeric_editor(ui, &control, &mut value);
                });
            },
        );
        assert_eq!(value, expected, "typed {typed} in {json}");
    }
}

#[test]
fn preview_wheel_zoom_and_pointer_drag_are_local_display_changes() {
    let context = egui::Context::default();
    let mut transform = PreviewTransform::default();
    let mut rect = Rect::NOTHING;
    run_frame(&context, vec![], |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let response =
                ui.allocate_response(egui::vec2(400.0, 300.0), egui::Sense::click_and_drag());
            rect = response.rect;
            interact_preview(ui, &response, &mut transform);
        });
    });
    let anchor = rect.center() + egui::vec2(80.0, 30.0);
    let mut draw = |ctx: &egui::Context| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let response =
                ui.allocate_response(egui::vec2(400.0, 300.0), egui::Sense::click_and_drag());
            interact_preview(ui, &response, &mut transform);
        });
    };
    run_frame(
        &context,
        wheel(anchor, MouseWheelUnit::Point, 120.0),
        &mut draw,
    );
    run_frame(
        &context,
        vec![Event::PointerButton {
            pos: anchor,
            button: PointerButton::Primary,
            pressed: true,
            modifiers: Modifiers::NONE,
        }],
        &mut draw,
    );
    run_frame(
        &context,
        vec![Event::PointerMoved(anchor + egui::vec2(40.0, 0.0))],
        &mut draw,
    );
    drop(draw);
    assert!(transform.zoom > 1.0);
    assert_ne!(transform.pan, Vec2::ZERO);
    transform.zoom_at(0.001, Vec2::ZERO);
    assert_eq!(transform.zoom, 1.0);
    assert_eq!(transform.pan, Vec2::ZERO);
}

#[test]
fn transport_requires_backend_capability_and_ack_feedback_survives_state_writes() {
    let mut fixture = Fixture::new();
    fixture.app.runtime = Some(RuntimeStatus {
        state: "running".into(),
        updated_ms: now_ms(),
        ..Default::default()
    });
    let context = egui::Context::default();
    let output = run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    assert!(text_positions(&output, "Pause").is_empty());
    assert!(text_positions(&output, "Stop").is_empty());
    fixture.app.runtime.as_mut().unwrap().capabilities = vec!["pause".into(), "stop".into()];
    let output = run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    let pause = text_positions(&output, "Pause")[0];
    run_frame(&context, click_events(pause), |ctx| fixture.app.draw(ctx));
    assert_eq!(fixture.app.actions.len(), 1);
    assert_eq!(fixture.app.actions[0].kind, "pause");
    write_json_atomic(
        &fixture.directory.join("ack.json"),
        &serde_json::json!({"seq": 1, "error": "pause unavailable during shutdown"}),
    )
    .unwrap();
    fixture.app.poll_action_ack();
    assert!(fixture.app.actions.is_empty());
    fixture.app.write_state().unwrap();
    fixture.app.reload_spec(true).unwrap();
    assert_eq!(
        fixture.app.action_feedback.as_deref(),
        Some("Action failed: pause unavailable during shutdown")
    );
}

#[test]
fn malformed_runtime_status_is_visible_and_disables_transport() {
    let mut fixture = Fixture::new();
    let path = fixture.directory.join("runtime.json");
    fixture.app.spec.runtime_path = Some(path.clone());
    fs::write(path, "broken").unwrap();
    fixture.app.poll_runtime();
    assert!(fixture.app.runtime.is_none());
    assert!(fixture
        .app
        .runtime_error
        .as_deref()
        .unwrap()
        .contains("parse runtime status"));
    assert!(parse_runtime_status(r#"{"state":"running","updated_ms":0,"fps":0}"#).is_err());
}

#[test]
fn playback_speed_selection_sends_the_capability_supported_rate_action() {
    let mut fixture = Fixture::new();
    fixture.app.runtime = Some(RuntimeStatus {
        state: "running".into(),
        rate: Some(1.0),
        updated_ms: now_ms(),
        capabilities: vec!["rate".into()],
        ..Default::default()
    });
    let context = egui::Context::default();
    run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    let output = run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    let rate = text_positions(&output, "1×")[0];
    run_frame(&context, click_events(rate), |ctx| fixture.app.draw(ctx));
    let output = run_frame(&context, vec![], |ctx| fixture.app.draw(ctx));
    let fast = text_positions(&output, "2×")[0];
    run_frame(&context, click_events(fast), |ctx| fixture.app.draw(ctx));
    assert_eq!(fixture.app.actions.last().unwrap().kind, "rate:2");
}

#[test]
fn restarted_backend_discards_pending_actions_and_old_control_revisions() {
    let mut fixture = Fixture::new();
    fixture.app.set_action("pause");
    fixture
        .app
        .values
        .get_mut("camera")
        .unwrap()
        .insert("Exposure_EV_x10".into(), 50);
    let path = fixture.directory.join("spec.json");
    let mut spec: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    spec["session_id"] = "new-pipeline".into();
    spec["windows"][0]["controls"][1]["value"] = 30.into();
    write_json_atomic(&path, &spec).unwrap();
    fixture.app.reload_spec(true).unwrap();
    assert!(fixture.app.actions.is_empty());
    assert_eq!(fixture.app.values["camera"]["Exposure_EV_x10"], 30);
    fixture.app.set_action("resume");
    assert_eq!(fixture.app.actions.last().unwrap().seq, 1);
}
