"""Download CIC-IDS 2017 PCAPs + CSV archives from the CIC research portal.

The CIC mirror gates downloads behind a registration session: the user must register
at https://www.unb.ca/cic/datasets/ids-2017.html, log into cicresearch.ca, and supply
the resulting Token cookie via --cookie-token (or a Netscape cookie file via
--cookie-file). Without auth, all direct file URLs 302 to the registration page.

Default subset: Tuesday + Wednesday + Friday PCAPs, covers Brute Force / DoS Slowloris /
DoS Hulk / Heartbleed / DDoS / Botnet / Port Scan. See Idea.md §3.1, §7.1.

Authoritative MD5 hashes are fetched from sibling `.md5` files on the server, so no
hashes are hard-coded in this script.

Note: cicresearch.ca's download.php endpoint does NOT honor HTTP Range, so partial
downloads cannot resume — a failed transfer will restart from byte 0.

Usage:
    uv run python scripts/download_cicids2017.py --dry-run
    uv run python scripts/download_cicids2017.py --cookie-token <Token> --yes
    uv run python scripts/download_cicids2017.py --cookie-file cookies.txt --days all --yes
"""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm

from nid_video.utils import logger, project_root, setup_logger

DEFAULT_BASE_URL = "https://cicresearch.ca/CICDataset/CIC-IDS-2017"
USER_AGENT = "nid-video/0.1 (cicids2017 fetcher)"


@dataclass(frozen=True)
class CICFile:
    """One downloadable artifact in the CIC-IDS 2017 dataset."""

    name: str               # e.g. "Tuesday-WorkingHours.pcap"
    day: str                # mon | tue | wed | thu | fri | "any" (for non-day-specific zips)
    kind: str               # "pcap" | "csv_zip"
    approx_size_mb: int

    @property
    def subfolder(self) -> str:
        return "PCAPs" if self.kind == "pcap" else "CSVs"

    @property
    def remote_path(self) -> str:
        """Logical path used inside the download.php `file=` query parameter."""
        return f"CIC-IDS-2017/{self.subfolder}/{self.name}"

    @property
    def md5_remote_path(self) -> str:
        """Sibling .md5 manifest path on the server (basename + .md5)."""
        stem = self.name.rsplit(".", 1)[0]
        return f"CIC-IDS-2017/{self.subfolder}/{stem}.md5"


# Authoritative manifest verified against the cicresearch.ca browser.
# Note: Wednesday filename uses lowercase 'w' in 'workingHours' on the server.
MANIFEST: list[CICFile] = [
    CICFile("Monday-WorkingHours.pcap",     "mon", "pcap", 10_500),
    CICFile("Tuesday-WorkingHours.pcap",    "tue", "pcap", 10_500),
    CICFile("Wednesday-workingHours.pcap",  "wed", "pcap", 12_500),  # lowercase w
    CICFile("Thursday-WorkingHours.pcap",   "thu", "pcap",  7_500),
    CICFile("Friday-WorkingHours.pcap",     "fri", "pcap",  8_200),
    CICFile("GeneratedLabelledFlows.zip",   "any", "csv_zip",  240),
    CICFile("MachineLearningCSV.zip",       "any", "csv_zip",  220),
]

ALL_DAYS = {"mon", "tue", "wed", "thu", "fri"}
DEFAULT_DAYS = {"tue", "wed", "fri"}


# ---------------------------------------------------------------------------
# Selection / planning
# ---------------------------------------------------------------------------


def parse_days(spec: str) -> set[str]:
    if spec == "all":
        return set(ALL_DAYS)
    days = {d.strip().lower() for d in spec.split(",") if d.strip()}
    bad = days - ALL_DAYS
    if bad:
        raise ValueError(f"Unknown day(s): {sorted(bad)}; expected from {sorted(ALL_DAYS)}")
    return days


def select_files(days: set[str], include_pcaps: bool, include_csvs: bool) -> list[CICFile]:
    out: list[CICFile] = []
    for f in MANIFEST:
        if f.kind == "pcap" and include_pcaps and f.day in days:
            out.append(f)
        elif f.kind == "csv_zip" and include_csvs:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# URLs / hashing / disk
# ---------------------------------------------------------------------------


def build_url(base_url: str, remote_path: str) -> str:
    return f"{base_url}/download.php?file={quote(remote_path, safe='')}"


def check_disk_space(target_dir: Path, required_gb: float) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target_dir).free / (1024 ** 3)
    logger.info(f"Free disk on {target_dir}: {free_gb:.1f} GB (need >= {required_gb:.1f} GB)")
    if free_gb < required_gb:
        raise RuntimeError(
            f"Not enough free disk: {free_gb:.1f} GB < required {required_gb:.1f} GB"
        )


def compute_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_md5_file_text(text: str) -> str:
    """Pick the hash from a `md5sum` formatted file: `<hex32>  <name>`."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        if len(token) == 32 and all(c in "0123456789abcdefABCDEF" for c in token):
            return token.lower()
    raise ValueError(f"No md5 hash found in: {text!r}")


# ---------------------------------------------------------------------------
# HTTP session / cookies
# ---------------------------------------------------------------------------


def build_session(cookie_token: str | None, cookie_file: Path | None) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    if cookie_file is not None:
        jar = http.cookiejar.MozillaCookieJar(str(cookie_file))
        jar.load(ignore_discard=True, ignore_expires=True)
        s.cookies.update(jar)
        logger.info(f"Loaded {len(s.cookies)} cookies from {cookie_file}")
    if cookie_token:
        s.cookies.set("Token", cookie_token, domain="cicresearch.ca", path="/")
        logger.info("Auth: Token cookie attached for cicresearch.ca")
    return s


# ---------------------------------------------------------------------------
# Download backends
# ---------------------------------------------------------------------------


def have_aria2c() -> bool:
    return shutil.which("aria2c") is not None


def download_with_aria2c(
    url: str,
    dest: Path,
    cookie_token: str | None,
    cookie_file: Path | None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        "-c",                   # continue (server may ignore Range; aria still tries)
        "-x", "8", "-s", "8",
        "--max-tries=5",
        "--retry-wait=10",
        "--auto-file-renaming=false",
        f"--user-agent={USER_AGENT}",
        "-d", str(dest.parent),
        "-o", dest.name,
    ]
    if cookie_file is not None:
        cmd.extend(["--load-cookies", str(cookie_file)])
    if cookie_token:
        cmd.extend([f"--header=Cookie: Token={cookie_token}"])
    cmd.append(url)
    logger.info(f"aria2c -> {dest}")
    subprocess.run(cmd, check=True)


def download_with_requests(
    url: str,
    dest: Path,
    session: requests.Session,
    timeout: int = 60,
) -> None:
    """Streamed download with optional Range resume + tqdm progress.

    cicresearch.ca's download.php returns 200 (full body) for Range requests, so
    resume is detected as failed and the file restarts from 0. Other endpoints
    that honor Range (e.g. mirrors) will continue from the existing offset.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with session.get(url, stream=True, headers=headers, timeout=timeout, allow_redirects=False) as r:
        if 300 <= r.status_code < 400:
            raise RuntimeError(
                f"Got redirect {r.status_code} -> {r.headers.get('Location')!r}; "
                "auth likely failed (Token cookie missing/expired). "
                "Re-export the cookie and pass --cookie-token / --cookie-file."
            )
        if r.status_code == 416:
            logger.info(f"Already complete: {dest.name}")
            return
        if existing and r.status_code != 206:
            logger.warning(
                f"Server ignored Range header (status={r.status_code}); "
                f"restarting {dest.name} from 0"
            )
            existing = 0
            dest.unlink(missing_ok=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise RuntimeError(
                f"Server returned HTML for {dest.name} (likely auth/registration page). "
                "Check --cookie-token / --cookie-file."
            )
        total = int(r.headers.get("Content-Length", 0)) + existing
        mode = "ab" if existing else "wb"
        with dest.open(mode) as fh, tqdm(
            total=total or None,
            initial=existing,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name,
            ncols=80,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))


def fetch_md5_from_server(
    f: CICFile, base_url: str, session: requests.Session, timeout: int = 30
) -> str | None:
    """Fetch and parse the sibling .md5 manifest. Returns None if unavailable."""
    url = build_url(base_url, f.md5_remote_path)
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=False)
        if resp.status_code != 200 or "text/html" in resp.headers.get("Content-Type", ""):
            logger.warning(f"No .md5 available for {f.name} (status={resp.status_code})")
            return None
        return parse_md5_file_text(resp.text)
    except Exception as exc:
        logger.warning(f"Failed to fetch .md5 for {f.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Plan / dry-run
# ---------------------------------------------------------------------------


def print_plan(files: list[CICFile], base_url: str, out_dir: Path, has_auth: bool) -> None:
    total_mb = sum(f.approx_size_mb for f in files)
    logger.info("=" * 78)
    logger.info("CIC-IDS 2017 download plan")
    logger.info("=" * 78)
    logger.info(f"Output directory : {out_dir}")
    logger.info(f"Base URL         : {base_url}")
    logger.info(f"Backend          : {'aria2c' if have_aria2c() else 'requests'}")
    logger.info(f"Auth             : {'cookie attached' if has_auth else 'NONE — will fail!'}")
    logger.info(f"Files            : {len(files)}  (total ~{total_mb / 1024:.1f} GB)")
    logger.info(
        "Note: cicresearch.ca does NOT support HTTP Range; a failed transfer "
        "will restart from byte 0."
    )
    logger.info("-" * 78)
    for f in files:
        logger.info(
            f"  [{f.day:<3}] {f.kind:<8} ~{f.approx_size_mb:>6} MB  {f.name}\n"
            f"           url : {build_url(base_url, f.remote_path)}\n"
            f"           md5 : will fetch from {f.md5_remote_path}"
        )
    logger.info("=" * 78)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def download_one(
    f: CICFile,
    base_url: str,
    out_dir: Path,
    session: requests.Session,
    cookie_token: str | None,
    cookie_file: Path | None,
    prefer_aria2c: bool,
) -> None:
    url = build_url(base_url, f.remote_path)
    dest = out_dir / f.subfolder / f.name
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"Found existing {dest.name} ({size_mb:.1f} MB) — will retry / verify")

    if prefer_aria2c and have_aria2c():
        download_with_aria2c(url, dest, cookie_token, cookie_file)
    else:
        download_with_requests(url, dest, session)

    expected = fetch_md5_from_server(f, base_url, session)
    actual = compute_md5(dest)
    if expected is None:
        logger.warning(f"No server-side MD5 to verify {f.name}; computed = {actual}")
    elif actual.lower() != expected.lower():
        raise RuntimeError(f"MD5 mismatch for {f.name}: expected {expected}, got {actual}")
    else:
        logger.info(f"MD5 OK: {f.name} = {actual}")


def run(
    files: Iterable[CICFile],
    base_url: str,
    out_dir: Path,
    session: requests.Session,
    cookie_token: str | None,
    cookie_file: Path | None,
    prefer_aria2c: bool,
) -> None:
    for f in files:
        try:
            download_one(f, base_url, out_dir, session, cookie_token, cookie_file, prefer_aria2c)
        except Exception as exc:
            logger.error(f"Failed: {f.name} -> {exc}")
            raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", default=",".join(sorted(DEFAULT_DAYS)),
                   help="Comma-separated days (mon,tue,wed,thu,fri) or 'all'. "
                        f"Default: {','.join(sorted(DEFAULT_DAYS))}")
    p.add_argument("--no-pcaps", action="store_true", help="Skip PCAP files")
    p.add_argument("--no-csvs", action="store_true", help="Skip CSV ZIP archives")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"Base URL for the CIC mirror. Default: {DEFAULT_BASE_URL}")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Default: <project>/data/raw/cicids2017")
    p.add_argument("--cookie-token", default=None,
                   help="Token cookie value from cicresearch.ca after registration")
    p.add_argument("--cookie-file", default=None, type=Path,
                   help="Netscape-format cookies.txt with cicresearch.ca cookies")
    p.add_argument("--use-aria2c", dest="prefer_aria2c", action="store_true", default=True)
    p.add_argument("--no-aria2c", dest="prefer_aria2c", action="store_false")
    p.add_argument("--required-gb", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    p.add_argument("--yes", action="store_true", help="Required to start a real download")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logger(level="INFO")
    args = parse_args(argv)

    days = parse_days(args.days)
    files = select_files(
        days,
        include_pcaps=not args.no_pcaps,
        include_csvs=not args.no_csvs,
    )
    if not files:
        logger.error("No files matched. Check --days / --no-pcaps / --no-csvs.")
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else project_root() / "data" / "raw" / "cicids2017"
    has_auth = bool(args.cookie_token or args.cookie_file)
    print_plan(files, args.base_url, out_dir, has_auth)

    if args.dry_run:
        logger.info("Dry run complete — no files written.")
        return 0
    if not args.yes:
        logger.error("Real download requires --yes (safety belt).")
        return 3
    if not has_auth:
        logger.error(
            "Real download requires authentication. Pass --cookie-token <Token> "
            "(after registering at https://www.unb.ca/cic/datasets/ids-2017.html) "
            "or --cookie-file cookies.txt."
        )
        return 4

    check_disk_space(out_dir, required_gb=args.required_gb)
    session = build_session(args.cookie_token, args.cookie_file)
    run(files, args.base_url, out_dir, session, args.cookie_token, args.cookie_file,
        prefer_aria2c=args.prefer_aria2c)
    logger.info("All downloads finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
