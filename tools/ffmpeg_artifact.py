"""File-minimal FFmpeg execution for parent-authorized media artifacts.

This is an opt-in library boundary, NOT a general FFmpeg command runner or a
path-authorization service. Callers must authorize the source and destination.
Only one regular input and one NEW regular output are supported; no raw argv,
URLs, pipes, multiple outputs, custom runtimes or privileged fallback.

Linux/system FFmpeg only: bwrap must support unprivileged namespaces. Public
system libraries are visible read-only; home, /etc, host /tmp, the input's
siblings and the caller's destination directory are not. This contains demuxer
secondary reads, not CPU/memory/disk exhaustion or kernel vulnerabilities.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile


class ArtifactSandboxUnavailable(RuntimeError):
    """The selected boundary cannot execute; callers must not retry unsandboxed."""


# These are parent-owned recipes, never options supplied by artifact contents.
# Thumbnail matches Baileys' default first-frame, 32-pixel JPEG operation.
_RECIPES = {
    "opus": (".ogg", [], ["-ar", "48000", "-ac", "1", "-c:a", "libopus"]),
    "thumbnail": (".jpg", ["-ss", "0"], ["-vf", "scale=32:-1", "-vframes", "1", "-f", "image2"]),
    "gif": (".mp4", [], ["-movflags", "faststart", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]),
    "sticker": (".webp", [], [
        "-vf", "scale=512:512:force_original_aspect_ratio=decrease,format=rgba,pad=512:512:(ow-iw)/2:(oh-ih)/2:color=0x00000000",
        "-c:v", "libwebp", "-quality", "80", "-preset", "picture", "-compression_level", "6", "-vsync", "0",
    ]),
}


def _regular_reader(path: Path):
    # Pin the opened inode, reject symlinks/devices and avoid blocking on FIFOs.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("FFmpeg artifacts must be regular files")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def _library_search_path(executable: Path) -> str:
    # Inspect only our fixed, trusted system executable, never an input artifact.
    # Debian's BLAS/LAPACK links traverse /etc/alternatives. Resolve their public
    # library directories rather than mounting /etc (or the host loader cache).
    result = subprocess.run(["/usr/bin/ldd", str(executable)], check=True,
                            capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
                            env={"PATH": "/usr/bin:/bin", "LANG": "C"})
    roots = [Path(p).resolve() for p in ("/usr/lib", "/usr/lib64", "/lib", "/lib64")]
    directories = set()
    for match in re.finditer(r"(?:=>\s+|^\s*)(/\S+)\s+\(", result.stdout, re.MULTILINE):
        library = Path(match.group(1)).resolve(strict=True)
        if not any(library.is_relative_to(root) for root in roots):
            raise ArtifactSandboxUnavailable("FFmpeg library outside public runtime roots")
        directories.add(str(library.parent))
    if not directories:
        raise ArtifactSandboxUnavailable("Unable to resolve FFmpeg public runtime libraries")
    return ":".join(sorted(directories))


def convert_artifact(
    source: str | os.PathLike,
    destination: str | os.PathLike,
    *,
    operation: str,
    timeout: float = 30,
) -> Path:
    """Convert a parent-authorized file; publish only after sandbox success.

    ``destination`` must be a new file in a parent-controlled directory. The
    parent, not a worker/model, selects it. An existing file or symlink is never
    overwritten. Raises on all failures; there is deliberately no fallback.
    Stdout/pipe outputs and arbitrary FFmpeg option shapes are not supported.
    """
    if operation not in _RECIPES:
        raise ValueError("Unsupported artifact conversion operation")
    suffix, before, after = _RECIPES[operation]
    for value in (source, destination):
        value = os.fspath(value)
        if not isinstance(value, str) or value == "-" or ":" in value or "%" in value:
            raise ValueError("Only single local file paths are supported")
    src, dest = Path(source), Path(destination)
    if dest.suffix.lower() != suffix:
        raise ValueError(f"{operation} output requires {suffix}")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,16}", src.suffix):
        raise ValueError("Input requires a simple preserved file extension")
    if not (0 < timeout <= 300):
        raise ValueError("Timeout must be between 0 and 300 seconds")
    if sys.platform != "linux":
        raise ArtifactSandboxUnavailable("Artifact FFmpeg requires Linux namespaces")
    bwrap = shutil.which("bwrap", path="/usr/bin:/bin")
    if not bwrap:
        raise ArtifactSandboxUnavailable("Artifact FFmpeg requires bwrap")
    executable = Path("/usr/bin/ffmpeg")
    if not executable.is_file():
        raise ArtifactSandboxUnavailable("Artifact FFmpeg requires /usr/bin/ffmpeg")

    with tempfile.TemporaryDirectory(prefix="hermes-ffmpeg-") as temporary:
        scratch = Path(temporary)
        staged = scratch / ("artifact" + src.suffix)
        with _regular_reader(src) as reader, staged.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        outdir = scratch / "output"
        outdir.mkdir(mode=0o700)
        result_file = outdir / ("result" + suffix)
        input_name = "/input/" + staged.name
        output_name = "/output/" + result_file.name
        command = [
            bwrap, "--unshare-all", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL", "--clearenv",
            "--setenv", "PATH", "/usr/bin", "--setenv", "LANG", "C",
            "--setenv", "LD_LIBRARY_PATH", _library_search_path(executable),
            "--ro-bind", str(executable), str(executable),
        ]
        # Explicit public runtime roots only, never a host-root or /etc bind.
        # Bind the directory target at its original path (works on merged-usr
        # and split-usr systems without exposing any additional root trees).
        for runtime in ("/usr/lib", "/usr/lib64", "/lib", "/lib64"):
            if Path(runtime).is_dir():
                command.extend(["--ro-bind", runtime, runtime])
        command.extend([
            # no-tmp: ok — bubblewrap mounts a fresh tmpfs at the SANDBOX's own /tmp (inside the jail), not host scratch
            "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/input",
            "--ro-bind", str(staged), input_name,
            "--bind", str(outdir), "/output", "--chdir", "/output", "--",
            str(executable), "-hide_banner", "-nostdin", "-y", *before,
            "-i", input_name, *after, output_name,
        ])
        # No protocol whitelist is relied on: namespace isolation must stop
        # secondary file AND network opens, even for an autodetected playlist.
        subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                       check=True, timeout=timeout, close_fds=True,
                       env={"PATH": "/usr/bin:/bin", "LANG": "C"})
        # No destination mount: FFmpeg never receives root-side output authority.
        # Exclusive creation also closes the exists()/create race for the leaf.
        with _regular_reader(result_file) as reader:
            with dest.open("xb") as writer:
                try:
                    shutil.copyfileobj(reader, writer)
                except BaseException:
                    dest.unlink(missing_ok=True)
                    raise
    return dest
