# Hockey roles detector

Select **Hockey roles — mixed iter 487500 (YOLOv8-m, 1984x736)** in the shared detector menu,
or pass `--detector-model hockey-roles` to hmtrack. The default is the newer
single-class CrowdHuman epoch-90 checkpoint at the same resolution.
The pinned checkpoint downloads from HockeyMONStream's `pretrained-assets-v1`
release and is verified on every cache reuse. It now uses original-background
hockey training with spectator person labels, camera degradation augmentation,
and CrowdHuman, with classes person (0), goalie (1), and official (2). The
inference graph is resized to its trained 1984×736 geometry.
Iteration 487500 has CrowdHuman mAP 0.527, original-background hockey mAP 0.623,
and official AP 0.585. CrowdHuman mAP was 0.530 for the previous gray-background
roles model and 0.534 for the person-only epoch-90 model. The old gray hockey
validation score used different imagery/annotations and is not directly comparable.

The new checkpoint filename includes `hockey_roles_original` and its digest;
selecting the existing roles option downloads it and changes the compiled-engine
cache identity. HStream's roles ONNX is exported from the same checkpoint.
Existing explicit checkpoint overrides continue to use their selected files.

In a small offline comparison of 12 fixed-time Littleton Hawks and CAHA Golden
Bears video frames at confidence 0.25, the new model produced 13 visually confirmed
referee detections versus 2 from the gray model. Some obvious refs still became
person detections. These rendered program frames do not establish native pipeline
recall or tracking stability; review actual game results before relying on role
exclusions.

Program tracking offers separate **Ignore referees for play tracking** and
**Ignore goalies for play tracking** switches. Both default off. They change camera
subjects, including rink-aware framing, without changing the published object
tracks or tracking-box display. The canonical configuration keys are
`rink.tracking.cam_ignore_referees` and `rink.tracking.cam_ignore_goalies`.
Rule-based camera controls apply these exclusions live. Learned transformer/GPT
controllers apply them at startup and require a restart to change them; live
requests fail explicitly before changing the active settings.
Learned controllers also filter their optional pose features by matching pose
boxes to valid labeled tracks (at least 0.3 IoU, as in pose display matching).
When exclusions are active, unmatched poses do not contribute camera features;
published poses and pose overlays remain intact. Live changes to effective role
exclusions create camera-policy boundaries in training exports.
The selected detector supplies the class indices; single-class person detectors
cannot apply role exclusions. Tracked boxes display GOALIE or REF, without a
person/player label. Existing referee-overlay visibility controls remain honored.

Pose-derived detections retain the role of an overlapping source detection so
optional pose-to-detector graphs do not erase the selected model's roles.

Assets and provenance:
https://github.com/cjolivier01/HockeyMONStream/releases/tag/pretrained-assets-v1
