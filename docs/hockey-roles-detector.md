# Epoch-10 hockey roles detector

Select **Hockey roles — epoch 10 (YOLOv8-m, 1408x544)** in the shared detector menu,
or pass `--detector-model hockey-roles` to hmtrack. The existing default is unchanged.
The pinned checkpoint downloads from HockeyMONStream's `pretrained-assets-v1`
release and is verified on every cache reuse. It is COCO initialized and trained
with CrowdHuman and gray-background hockey, with classes person (0), goalie (1),
and official (2). The inference graph is resized to its trained 1408×544 geometry.

Program tracking offers separate **Ignore referees for play tracking** and
**Ignore goalies for play tracking** switches. Both default off. They change camera
subjects, including rink-aware framing, without changing the published object
tracks or tracking-box display. The canonical configuration keys are
`rink.tracking.cam_ignore_referees` and `rink.tracking.cam_ignore_goalies`.
The selected detector supplies the class indices; single-class person detectors
cannot apply role exclusions. Tracked boxes display GOALIE or REF, without a
person/player label. Existing referee-overlay visibility controls remain honored.

Pose-derived detections retain the role of an overlapping source detection so
optional pose-to-detector graphs do not erase the selected model's roles.

Assets and provenance:
https://github.com/cjolivier01/HockeyMONStream/releases/tag/pretrained-assets-v1
