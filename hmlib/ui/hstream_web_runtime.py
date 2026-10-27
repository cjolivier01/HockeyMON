"""Typed web commands over the same frame-owner mailbox as the desktop."""

from __future__ import annotations

from hmlib.ui.hstream_runtime import PREVIEW_NAMES, RuntimeProtocol, read_document
from hmlib.ui.hstream_web import canonical_bytes
from hmlib.ui.hstream_web_control import ACTIONS, MAX_INTEGER, decode_request, validate_payload
from hmlib.ui.hstream_web_preview import read_owner


class WebRuntimeProtocol(RuntimeProtocol):
    """Confined-launch adapter. No legacy text command can reach a web worker."""

    def __init__(self, *args, allow_seek: bool, realtime: bool, stop_applied, **kwargs):
        super().__init__(*args, **kwargs)
        if self.web_binding is None:
            raise ValueError("Web controls require a confined admitted launch")
        self._allow_seek = allow_seek
        self._realtime = realtime
        self._replay_rate = None
        self._stop_applied = stop_applied
        self._web_request = None
        self._last_web_request_id = 0
        self._ready_session = None
        self._ready_actions = []
        self._web_stopping = False
        self._readiness_error = None

    def _owner(self, *, retired: bool = False) -> dict:
        return read_owner(
            self.session,
            expected={
                **self.web_binding,
                "session_id": self._session_id,
                "channel": self._active_channel,
                "generation": self._preview_generation,
                "status": "retired" if retired else "selected",
            },
        )

    def _publish_ready(self) -> None:
        if (
            self._web_stopping
            or self._web_request is not None
            or self._ready_session == self._session_id
        ):
            return
        self._spec()
        runtime = read_document(self.session / "runtime.json")
        if runtime.get("session_id") != self._session_id or runtime.get("state") not in {
            "running",
            "paused",
        }:
            return
        capabilities = runtime.get("capabilities")
        if not isinstance(capabilities, list) or "stop" not in capabilities:
            return
        owner = self._owner()
        self._ready_actions = [
            action
            for action in ACTIONS
            if (
                action in {"preview", "preview-overlays", "properties"}
                or (action == "seek" and self._allow_seek and self.seek is not None)
                or (action in {"pause", "resume", "step", "stop"} and action in capabilities)
                or (action == "rate" and self._realtime and action in capabilities)
            )
        ]
        self._ready_session = self._session_id
        self.emit(
            "HM_WEB_PROGRAM_READY "
            + canonical_bytes(
                {
                    "version": 1,
                    **self.web_binding,
                    "session_id": self._session_id,
                    "actions": self._ready_actions,
                    "owner": owner,
                }
            ).decode()
        )

    def _terminal(
        self, request: dict, *, result: dict | None = None, error: str | None = None
    ) -> None:
        if self._web_request is request:
            self._web_request = None
        message = {
            **{
                key: request[key]
                for key in (
                    "version",
                    "run_id",
                    "request_digest",
                    "session_id",
                    "request_id",
                    "action",
                )
            },
            "status": "failed" if error is not None else "applied",
            "result": None if error is not None else result,
            "error": str(error).replace("\0", " ")[:1024] if error is not None else None,
        }
        self.emit("HM_WEB_PROGRAM_CONTROL " + canonical_bytes(message).decode())

    def _applied(self, request: dict, receipt: dict) -> None:
        if receipt.get("ok") is not True:
            self._terminal(
                request, error=str(receipt.get("error") or "Frame owner rejected the request")
            )
            return
        action, payload = request["action"], request["payload"]
        try:
            result = dict(payload)
            if action in {"pause", "resume", "step"}:
                self._acknowledge_process_pause(action != "resume", cooperative=True)
            if action == "rate":
                self._replay_rate = payload["rate"]
            if action in {"preview", "preview-overlays", "stop"}:
                result["owner"] = self._owner(retired=action == "stop")
            if action == "stop":
                self._web_stopping = True
                self._stop_applied()
            self._terminal(request, result=result)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            self._terminal(request, error=str(error))

    def handle_line(self, line: str) -> bool:
        try:
            request = decode_request(line.rstrip("\r\n"))
        except (ValueError, TypeError, RecursionError) as error:
            self.emit(f"web runtime command rejected: {error}")
            return False
        try:
            if any(request[key] != self.web_binding[key] for key in ("run_id", "request_digest")):
                raise ValueError("Web control belongs to another admitted run")
            if request["request_id"] <= self._last_web_request_id:
                raise ValueError("Web control request identity was already used")
            self._last_web_request_id = request["request_id"]
            validate_payload(request["action"], request["payload"])
            if self._web_request is not None:
                raise ValueError("Another web control is pending")
            if self._web_stopping:
                raise ValueError("Web Program is stopping")
            if (
                self._ready_session != request["session_id"]
                or self._session_id != request["session_id"]
            ):
                raise ValueError("Web control does not own a ready pipeline session")
            self._spec()
            self._owner()
            action, payload = request["action"], request["payload"]
            if action not in self._ready_actions:
                raise ValueError("Web control is unavailable in this session")
            self._web_request = request
            if action == "seek":
                target = payload["position_ns"]
                if self.total_ns is not None and target >= self.total_ns:
                    raise ValueError("Seek position is outside the recording")
                self.seek(target, request["request_id"])
                return True
            if action == "preview":
                generation = self._preview_requested_generation + 1
                if generation > MAX_INTEGER:
                    raise ValueError("Preview generation exhausted")
                channel = payload["channel"]
                mutation = {
                    "kind": "preview",
                    "name": PREVIEW_NAMES[channel],
                    "channel": channel,
                    "generation": generation,
                }
            elif action == "preview-overlays":
                mutation = {
                    "kind": "operation",
                    "operation": {
                        "kind": "preview-overlays",
                        "layers": [payload[key] for key in ("players", "play", "rink")],
                    },
                }
            elif action == "properties":
                values = {}
                for item in payload["assignments"]:
                    self._translate_property(
                        item["element"], item["property"], str(item["value"]), values
                    )
                mutation = {"kind": "controls", "values": values}
            else:
                mutation = {
                    "kind": "playback",
                    "action": f"rate:{payload['rate']}" if action == "rate" else action,
                }
            self._queue(mutation, [], completion=lambda receipt: self._applied(request, receipt))
            if action == "preview":
                self._preview_requested_generation = generation
            return True
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            self._terminal(request, error=str(error))
            return False

    def poll(self) -> None:
        super().poll()
        try:
            self._publish_ready()
            self._readiness_error = None
        except FileNotFoundError:
            return
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            if str(error) != self._readiness_error:
                self.emit(f"web runtime readiness failed: {error}")
                self._readiness_error = str(error)

    def reset_child(self, offset_ns: int = 0) -> None:
        super().reset_child(offset_ns)
        self._ready_session = None
        self._ready_actions = []

    @property
    def replay_state(self) -> dict:
        state = super().replay_state
        if self._replay_rate is not None:
            state["rate"] = self._replay_rate
        return state

    def seek_completed(self, target_ns: int, generation: int) -> bool:
        request = self._web_request
        if (
            request is None
            or request["action"] != "seek"
            or request["request_id"] != generation
            or request["payload"]["position_ns"] != target_ns
        ):
            raise ValueError("Seek completion has no matching web request")
        try:
            self._spec()
            runtime = read_document(self.session / "runtime.json")
            if (
                self._session_id == request["session_id"]
                or runtime.get("session_id") != self._session_id
                or type(runtime.get("frame")) is not int
                or runtime.get("seek_ready") is not True
                or runtime.get("state") not in {"running", "paused"}
            ):
                return False
            owner = self._owner()
        except FileNotFoundError:
            return False
        self._terminal(request, result={"position_ns": target_ns, "owner": owner})
        self._publish_ready()
        return True

    def stopped(self, reason: str = "pipeline stopped") -> None:
        super().stopped(reason)
        self._web_stopping = True
        if self._web_request is not None:
            self._terminal(self._web_request, error=reason)
