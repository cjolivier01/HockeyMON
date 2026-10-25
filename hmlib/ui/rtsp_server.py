"""Standalone PyGObject RTSP relay; launched with an interpreter that provides GI.

Only encoded RTP crosses into this helper. Pipeline frames never pass through
JPEG, a UI preview, or another application's native runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstRtspServer", "1.0")
        from gi.repository import GLib, Gst, GstRtspServer
    except (ImportError, ValueError) as error:
        raise RuntimeError(
            "Local RTSP requires PyGObject and GstRtspServer. Install python3-gi, "
            "gir1.2-gst-rtsp-server-1.0 and gstreamer1.0-plugins-good/bad, or set "
            "HM_GI_PYTHON to an interpreter with these packages."
        ) from error
    config = json.loads(args.config)
    Gst.init(None)
    for name in ("udpsrc", "rtpjitterbuffer", "rtph264depay", "h264parse", "rtph264pay"):
        if Gst.ElementFactory.find(name) is None:
            raise RuntimeError(
                f"RTSP requires GStreamer element {name}; install the GStreamer good/bad plugins"
            )
    server = GstRtspServer.RTSPServer.new()
    server.set_address(config["host"])
    server.set_service(str(config["port"]))
    mounts = server.get_mount_points()
    for route in config["routes"]:
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_shared(True)
        factory.set_stop_on_disconnect(False)
        launch = (
            "udpsrc address=127.0.0.1 port="
            + str(route["udp_port"])
            + ' caps="application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96" '
            "! rtpjitterbuffer latency=100 ! rtph264depay ! h264parse "
            "! rtph264pay name=pay0 pt=96 config-interval=1 "
        )
        if route.get("audio"):
            for name in ("rtpmp4gdepay", "aacparse", "rtpmp4gpay"):
                if Gst.ElementFactory.find(name) is None:
                    raise RuntimeError(f"RTSP audio requires GStreamer element {name}")
            launch += (
                "udpsrc address=127.0.0.1 port="
                + str(route["audio_udp_port"])
                + ' caps="application/x-rtp,media=audio,encoding-name=MPEG4-GENERIC,clock-rate=48000,'
                "payload=97,streamtype=(string)5,profile-level-id=(string)1,mode=(string)AAC-hbr,config=(string)1190,"
                'sizelength=13,indexlength=3,indexdeltalength=3" '
                "! rtpjitterbuffer latency=100 ! rtpmp4gdepay ! aacparse ! rtpmp4gpay name=pay1 pt=97 "
            )
        factory.set_launch("( " + launch + ")")
        mounts.add_factory(route["mount"], factory)
    if server.attach(None) == 0:
        raise RuntimeError(
            f"Cannot listen on RTSP {config['host']}:{config['port']}; choose an unused port"
        )
    loop = GLib.MainLoop()
    parent = os.getppid()

    def check_parent() -> bool:
        if os.getppid() != parent:
            loop.quit()
            return False
        return True

    GLib.timeout_add(500, check_parent)
    for signum in (signal.SIGTERM, signal.SIGINT):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signum, lambda: loop.quit())
    print("READY", flush=True)
    loop.run()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
