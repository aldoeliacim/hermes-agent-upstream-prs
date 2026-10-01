"""Real native-decoder boundary tests. Run under the home-hiding harness.

No skips stand in for successful isolation: Linux qualification requires bwrap
and system FFmpeg. All source media and adversarial references are synthetic.
"""
import http.server
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import threading

import pytest

pytestmark = pytest.mark.linux_only


def boundary():
    assert importlib.util.find_spec("tools.ffmpeg_artifact"), "artifact-only FFmpeg boundary missing"
    return importlib.import_module("tools.ffmpeg_artifact")


def ffmpeg(*args):
    return subprocess.run(["/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                          capture_output=True, timeout=30)


@pytest.fixture
def media(tmp_path):
    outside = tmp_path / "private"
    outside.mkdir()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    tone = outside / "tone.ts"
    result = ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", "-f", "mpegts", tone)
    assert result.returncode == 0, result.stderr
    return artifact, tone


def playlist(path, reference):
    path.write_text("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n"
                    "#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1.0,\n" + reference + "\n#EXT-X-ENDLIST\n")


@pytest.mark.parametrize("protocols", [[], ["-protocol_whitelist", "file,pipe"]])
def test_replay_bridge_audio_includes_second_local_file_without_namespace(media, tmp_path, protocols):
    artifact, tone = media
    source = artifact / "input.m3u8"
    playlist(source, str(tone))
    result = ffmpeg(*protocols, "-i", source, "-ar", "48000", "-ac", "1", "-c:a", "libopus", tmp_path / "unsafe.ogg")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "unsafe.ogg").read_bytes().startswith(b"OggS")


@pytest.mark.parametrize("reference", ["absolute", "relative", "sibling"])
def test_secondary_file_is_not_visible(media, tmp_path, reference):
    mod = boundary()
    artifact, tone = media
    if reference == "sibling":
        sibling = artifact / "tone.ts"
        sibling.write_bytes(tone.read_bytes())
        ref = "tone.ts"
    else:
        ref = str(tone) if reference == "absolute" else "../private/tone.ts"
    source = artifact / "input.m3u8"
    playlist(source, ref)
    output = tmp_path / "blocked.ogg"
    with pytest.raises(subprocess.CalledProcessError) as caught:
        mod.convert_artifact(source, output, operation="opus")
    assert b"/input/artifact.m3u8" in caught.value.stderr
    assert not output.exists()


@pytest.mark.parametrize("suffix,codec", [(".wav", "pcm_s16le"), (".mp3", "libmp3lame"), (".mp4", "aac")])
def test_seekable_local_audio_transcodes(media, tmp_path, suffix, codec):
    mod = boundary()
    artifact, tone = media
    source = artifact / ("input with spaces" + suffix)
    result = ffmpeg("-i", tone, "-c:a", codec, source)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "converted.ogg"
    mod.convert_artifact(source, output, operation="opus")
    assert output.read_bytes().startswith(b"OggS")
    decoded = ffmpeg("-i", output, "-f", "s16le", "pipe:1")
    assert decoded.returncode == 0, decoded.stderr
    assert len(decoded.stdout) > 48000


@pytest.mark.parametrize("operation,extension,signature", [("thumbnail", ".jpg", b"\xff\xd8"), ("gif", ".mp4", None), ("sticker", ".webp", b"RIFF")])
def test_bridge_visual_conversion_shapes(tmp_path, operation, extension, signature):
    mod = boundary()
    source = tmp_path / "video.mp4"
    result = ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64:d=1", "-c:v", "mpeg4", source)
    assert result.returncode == 0, result.stderr
    if operation == "gif":
        gif_source = tmp_path / "animation.gif"
        result = ffmpeg("-i", source, "-vf", "fps=4", gif_source)
        assert result.returncode == 0, result.stderr
        source = gif_source
    output = tmp_path / ("out" + extension)
    mod.convert_artifact(source, output, operation=operation)
    data = output.read_bytes()
    assert data
    if signature:
        assert data.startswith(signature)
    else:
        assert b"ftyp" in data[:32]


@pytest.mark.parametrize("allow_http_protocol", [False, True])
def test_no_http_secondary_fetch(media, tmp_path, monkeypatch, allow_http_protocol):
    mod = boundary()
    artifact, tone = media
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            data = tone.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        source = artifact / "network.m3u8"
        playlist(source, f"http://127.0.0.1:{server.server_port}/tone.ts")
        # Positive control proves the loopback listener and the secondary URL work.
        control = ffmpeg("-protocol_whitelist", "file,http,tcp", "-i", source, "-c:a", "libopus", tmp_path / "control.ogg")
        assert control.returncode == 0, control.stderr
        assert requests
        requests.clear()
        if allow_http_protocol:
            # Exercise the SAME real namespace with FFmpeg's protocol guard
            # deliberately widened. Otherwise a protocol refusal could mask a
            # missing network namespace. This never adds a mount/network grant.
            real_run = subprocess.run

            def probe_run(command, **kwargs):
                if Path(command[0]).name == "bwrap":
                    index = command.index("-i")
                    command = [*command[:index], "-protocol_whitelist", "file,http,tcp", *command[index:]]
                return real_run(command, **kwargs)

            monkeypatch.setattr(mod.subprocess, "run", probe_run)
        with pytest.raises(subprocess.CalledProcessError) as caught:
            mod.convert_artifact(source, tmp_path / "network.ogg", operation="opus")
        assert b"/input/artifact.m3u8" in caught.value.stderr
        if allow_http_protocol:
            assert b"Connection refused" in caught.value.stderr or b"Network is unreachable" in caught.value.stderr
        assert requests == []
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_missing_bwrap_fails_closed(media, tmp_path, monkeypatch):
    mod = boundary()
    monkeypatch.setattr(mod.shutil, "which", lambda *a, **kw: None)
    with pytest.raises(mod.ArtifactSandboxUnavailable, match="bwrap"):
        mod.convert_artifact(media[1], tmp_path / "out.ogg", operation="opus")
    assert not (tmp_path / "out.ogg").exists()


@pytest.mark.parametrize("output", ["-", "pipe:1", "http://localhost/out.ogg", "out-%d.ogg"])
def test_rejects_non_single_file_output(media, tmp_path, output):
    mod = boundary()
    with pytest.raises(ValueError):
        mod.convert_artifact(media[1], output, operation="opus")


def test_rejects_unknown_operation_and_preserves_existing_destination(media, tmp_path):
    mod = boundary()
    dest = tmp_path / "out.ogg"
    dest.write_bytes(b"do not overwrite")
    with pytest.raises(ValueError):
        mod.convert_artifact(media[1], dest, operation="custom")
    with pytest.raises(FileExistsError):
        mod.convert_artifact(media[1], dest, operation="opus")
    assert dest.read_bytes() == b"do not overwrite"


def test_namespace_launch_failure_never_retries_privileged(media, tmp_path, monkeypatch):
    mod = boundary()
    # A failing launcher is an observable failure, never a reason to run ffmpeg
    # directly. Use a real failing process, not a mocked subprocess result.
    monkeypatch.setattr(mod.shutil, "which", lambda *a, **kw: "/usr/bin/false")
    dest = tmp_path / "out.ogg"
    with pytest.raises(subprocess.CalledProcessError):
        mod.convert_artifact(media[1], dest, operation="opus")
    assert not dest.exists()


def test_timeout_never_publishes_output(media, tmp_path):
    mod = boundary()
    dest = tmp_path / "out.ogg"
    with pytest.raises(subprocess.TimeoutExpired):
        mod.convert_artifact(media[1], dest, operation="opus", timeout=0.000001)
    assert not dest.exists()


def test_output_symlink_is_not_followed(media, tmp_path):
    mod = boundary()
    target = tmp_path / "keep"
    target.write_bytes(b"keep")
    dest = tmp_path / "link.ogg"
    dest.symlink_to(target)
    with pytest.raises(FileExistsError):
        mod.convert_artifact(media[1], dest, operation="opus")
    assert target.read_bytes() == b"keep"


def test_input_symlink_and_fifo_rejected(media, tmp_path):
    mod = boundary()
    source = tmp_path / "link.ts"
    source.symlink_to(media[1])
    with pytest.raises(OSError):
        mod.convert_artifact(source, tmp_path / "symlink.ogg", operation="opus")
    fifo = tmp_path / "fifo.ts"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        mod.convert_artifact(fifo, tmp_path / "fifo.ogg", operation="opus")
