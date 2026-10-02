#!/usr/bin/env python3
"""Fetch far-side active-region catalogs -> data/farside/ (idempotent).

Two independent 12-hourly detection catalogs (see FARSIDE_SCOPING.md):
  f6x   GONG helioseismic far-side AR catalog (NSO), 2010-01 -> present.
        https://gong2.nso.edu/oQR/f6x/YYYYMM/mrf6xYYMMDD/mrf6xYYMMDDtHHMM.txt
  sard  SDO/HMI Strong-AR-Discriminator lists (Stanford/JSOC), 2010-04 ->
        present. Plain HTTP only (HTTPS 403s). Years <=2023 sit in yearly
        subdirs, 2024-> at the top level; we try both.

GONG's WAF rate-limits hard (a 12-thread crawl earned an hours-long IP-wide
403 on 2026-08-16), so f6x is fetched SERIALLY at <= 1 req/s with backoff on
403; sard runs a small thread pool. Usage:

  fetch_farside.py [f6x|sard|all]     (default all)

Not every map exists (GONG discards duty-cycle<0.8 maps; JSOC has outages):
a 404 is recorded as an empty <name>.404 marker so reruns don't re-ask.
Markers younger than RETRY_DAYS (data may still be posted late) are retried.
Rerun any time to extend to today; safe to interrupt.
"""

import concurrent.futures
import datetime as dt
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "farside")
F6X_START = dt.date(2010, 1, 1)
SARD_START = dt.date(2010, 4, 25)
RETRY_DAYS = 7
GONG_DELAY = 1.0  # seconds between GONG requests — do not lower
GONG_BACKOFFS = [120, 300, 600, 1200]  # on 403: wait, retry; then give up
UA = {"User-Agent": "F107-farside-fetch/1.0 (research; connordimarco@gmail.com)"}

TODAY = dt.date.today()


def f6x_jobs():
    d = F6X_START
    while d <= TODAY:
        for hh in ("0000", "1200"):
            name = f"mrf6x{d:%y%m%d}t{hh}.txt"
            yield (
                os.path.join(DATA, "f6x", name),
                [f"https://gong2.nso.edu/oQR/f6x/{d:%Y%m}/mrf6x{d:%y%m%d}/{name}"],
                d,
            )
        d += dt.timedelta(days=1)


def sard_jobs():
    d = SARD_START
    while d <= TODAY:
        for hh in ("00", "12"):
            remote = f"AR_LIST_{d:%Y.%m.%d}_{hh}:00:00.txt"
            base = "http://jsoc.stanford.edu/data/farside/AR_Lists/"
            urls = [base + remote, f"{base}{d:%Y}/{remote}"]
            if d.year <= 2023:
                urls.reverse()
            yield (os.path.join(DATA, "sard", remote.replace(":", "")), urls, d)
        d += dt.timedelta(days=1)


def wanted(job):
    dest, _, day = job
    if os.path.exists(dest):
        return False
    if os.path.exists(dest + ".404") and (TODAY - day).days > RETRY_DAYS:
        return False
    return True


def fetch(job):
    """-> 'fetched' | '404' | 'blocked' | 'error ...'"""
    dest, urls, _ = job
    marker = dest + ".404"
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            with open(dest + ".tmp", "wb") as f:
                f.write(body)
            os.replace(dest + ".tmp", dest)
            if os.path.exists(marker):
                os.remove(marker)
            return "fetched"
        except urllib.error.HTTPError as e:
            if e.code == 403:
                return "blocked"
            if e.code != 404:
                return f"error {e.code} {url}"
        except Exception as e:  # timeouts, resets — leave for the next rerun
            return f"error {e} {url}"
    open(marker, "w").close()
    return "404"


def run_serial_gong(jobs):
    counts = {}
    for i, job in enumerate(jobs):
        res = fetch(job)
        if res == "blocked":
            for wait in GONG_BACKOFFS:
                print(f"GONG 403 — backing off {wait}s", flush=True)
                time.sleep(wait)
                res = fetch(job)
                if res != "blocked":
                    break
            else:
                print("GONG still 403 after all backoffs — try again later")
                counts["blocked"] = counts.get("blocked", 0) + 1
                break
        counts[res.split()[0]] = counts.get(res.split()[0], 0) + 1
        if (i + 1) % 500 == 0:
            print(f"f6x {i + 1}/{len(jobs)}: {counts}", flush=True)
        time.sleep(GONG_DELAY)
    return counts


def run_pool_sard(jobs):
    counts = {}
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        for i, res in enumerate(ex.map(fetch, jobs)):
            counts[res.split()[0]] = counts.get(res.split()[0], 0) + 1
            if (i + 1) % 1000 == 0:
                print(f"sard {i + 1}/{len(jobs)}: {counts}", flush=True)
    return counts


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(os.path.join(DATA, "f6x"), exist_ok=True)
    os.makedirs(os.path.join(DATA, "sard"), exist_ok=True)
    ok = True
    if which in ("sard", "all"):
        jobs = [j for j in sard_jobs() if wanted(j)]
        print(f"sard: {len(jobs)} files to try")
        counts = run_pool_sard(jobs)
        print(f"sard complete: {counts}")
        ok &= not counts.get("error") and not counts.get("blocked")
    if which in ("f6x", "all"):
        jobs = [j for j in f6x_jobs() if wanted(j)]
        print(f"f6x: {len(jobs)} files to try (serial, ~{len(jobs) * (GONG_DELAY + 0.3) / 3600:.1f} h)")
        counts = run_serial_gong(jobs)
        print(f"f6x complete: {counts}")
        ok &= not counts.get("error") and not counts.get("blocked")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
