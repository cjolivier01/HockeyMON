-- Hockey telemetry database v1. Keep identical to hmlib/telemetry/schema.sql.
PRAGMA application_id = 1213027156;
PRAGMA user_version = 1;
PRAGMA foreign_keys = ON;
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, game_id TEXT NOT NULL, started_utc TEXT NOT NULL,
  producer TEXT NOT NULL, source_config TEXT NOT NULL, effective_config TEXT NOT NULL,
  completed INTEGER NOT NULL DEFAULT 0 CHECK(completed IN (0,1)),
  outcome TEXT NOT NULL DEFAULT 'incomplete', sample_count INTEGER NOT NULL DEFAULT 0
) WITHOUT ROWID;
CREATE TABLE geometries (
  run_id TEXT NOT NULL REFERENCES runs(run_id), geometry_id INTEGER NOT NULL,
  width INTEGER NOT NULL CHECK(width>0), height INTEGER NOT NULL CHECK(height>0),
  coordinate_space TEXT NOT NULL, revision TEXT NOT NULL, mask_codec TEXT,
  mask BLOB, mask_sha256 TEXT, mask_to_tracking TEXT NOT NULL,
  PRIMARY KEY(run_id,geometry_id)
) WITHOUT ROWID;
CREATE TABLE rink_inputs (
  run_id TEXT NOT NULL, geometry_id INTEGER NOT NULL, input_id TEXT NOT NULL,
  format TEXT NOT NULL, metadata_json TEXT NOT NULL, data BLOB NOT NULL,
  PRIMARY KEY(run_id,geometry_id,input_id),
  FOREIGN KEY(run_id,geometry_id) REFERENCES geometries(run_id,geometry_id)
) WITHOUT ROWID;
CREATE TABLE frames (
  run_id TEXT NOT NULL REFERENCES runs(run_id), sample_id INTEGER NOT NULL CHECK(sample_id>0),
  source_id INTEGER NOT NULL, source_frame INTEGER NOT NULL,
  decoded_source_id INTEGER, decoded_sequence INTEGER, pts_ns INTEGER, ntp_ns INTEGER,
  seek_epoch INTEGER NOT NULL, reset_epoch INTEGER NOT NULL,
  geometry_id INTEGER NOT NULL, detection_count INTEGER NOT NULL, track_count INTEGER NOT NULL,
  PRIMARY KEY(run_id,sample_id),
  FOREIGN KEY(run_id,geometry_id) REFERENCES geometries(run_id,geometry_id)
) WITHOUT ROWID;
CREATE INDEX frames_time ON frames(run_id,pts_ns,sample_id);
CREATE TABLE detections (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
  left REAL NOT NULL, top REAL NOT NULL, width REAL NOT NULL, height REAL NOT NULL,
  score REAL NOT NULL, class_id INTEGER NOT NULL,
  PRIMARY KEY(run_id,sample_id,ordinal),
  FOREIGN KEY(run_id,sample_id) REFERENCES frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE tracks (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
  tracking_id TEXT NOT NULL, left REAL NOT NULL, top REAL NOT NULL,
  width REAL NOT NULL, height REAL NOT NULL, score REAL NOT NULL, class_id INTEGER NOT NULL,
  attributes_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(run_id,sample_id,ordinal),
  FOREIGN KEY(run_id,sample_id) REFERENCES frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE cameras (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL, role TEXT NOT NULL,
  left REAL NOT NULL, top REAL NOT NULL, width REAL NOT NULL, height REAL NOT NULL,
  PRIMARY KEY(run_id,sample_id,role),
  FOREIGN KEY(run_id,sample_id) REFERENCES frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE replay_frames (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL,
  arena_left REAL NOT NULL, arena_top REAL NOT NULL, arena_right REAL NOT NULL, arena_bottom REAL NOT NULL,
  stepped INTEGER NOT NULL, has_received_tracks INTEGER NOT NULL,
  edge_rotation_left REAL NOT NULL, edge_rotation_right REAL NOT NULL,
  PRIMARY KEY(run_id,sample_id),
  FOREIGN KEY(run_id,sample_id) REFERENCES frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE replay_tracks (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, tracking_id TEXT NOT NULL,
  left REAL NOT NULL, top REAL NOT NULL, right REAL NOT NULL, bottom REAL NOT NULL,
  PRIMARY KEY(run_id,sample_id,ordinal),
  FOREIGN KEY(run_id,sample_id) REFERENCES replay_frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE checkpoints (
  run_id TEXT NOT NULL, sample_id INTEGER NOT NULL, schema_version INTEGER NOT NULL,
  state TEXT NOT NULL, base_state TEXT NOT NULL,
  PRIMARY KEY(run_id,sample_id),
  FOREIGN KEY(run_id,sample_id) REFERENCES replay_frames(run_id,sample_id)
) WITHOUT ROWID;
CREATE TABLE config_events (
  run_id TEXT NOT NULL REFERENCES runs(run_id), event_id INTEGER NOT NULL,
  sample_boundary INTEGER NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
  artifact_name TEXT NOT NULL, artifact_contents TEXT NOT NULL,
  PRIMARY KEY(run_id,event_id)
) WITHOUT ROWID;
CREATE INDEX config_boundaries ON config_events(run_id,sample_boundary);
