#!/usr/bin/env python3
import argparse
import os
import re
import json
from datetime import datetime
from glob import glob


def parse_time_flexible(ts: str):
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            continue
    return None


def load_decisions(gem_dir: str):
    decisions = []  # list of (n:int, ts:datetime, path:str)
    if not os.path.isdir(gem_dir):
        return decisions
    for p in glob(os.path.join(gem_dir, "decision_*.json")):
        try:
            base = os.path.basename(p)
            n_str = base.split("_")[-1].split(".")[0]
            n = int(n_str)
        except Exception:
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ts_str = data.get('request', {}).get('timestamp')
            ts = parse_time_flexible(ts_str) if ts_str else None
        except Exception:
            ts = None
        decisions.append((n, ts, p))
    decisions.sort(key=lambda x: (x[1] or datetime.min, x[0]))
    return decisions


def nearest_decision(event_ts, decisions):
    if event_ts is None or not decisions:
        return None
    for n, ts, path in decisions:
        if ts and ts >= event_ts:
            return n
    return decisions[-1][0] if decisions[-1][1] else None


def scan_strategy_log(log_path: str):
    events = []  # list of dict(ts,parent,child,raw_line)
    # Pattern: "YYYY-mm-dd HH:MM:SS,ms - StreetViewStrategy - INFO - Emitted dead_end event P->C"
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'Emitted dead_end event' in line:
                try:
                    ts_part = line.split(' - ', 1)[0].strip()
                    ts = parse_time_flexible(ts_part)
                except Exception:
                    ts = None
                try:
                    msg = line.rsplit(' - ', 1)[-1]
                    after = msg.split('Emitted dead_end event', 1)[1].strip()
                    parts = after.split('->')
                    parent = parts[0].strip().strip('{}[]()<> "\'')
                    child = parts[1].strip().strip('{}[]()<> "\'') if len(parts) > 1 else None
                except Exception:
                    parent = child = None
                if parent and child:
                    events.append({'ts': ts, 'parent': parent, 'child': child, 'raw': line.strip()})
            elif 'Dead end detected!' in line and 'Emitted dead_end event' not in line:
                # Fallback: record detection without pano IDs
                try:
                    ts_part = line.split(' - ', 1)[0].strip()
                    ts = parse_time_flexible(ts_part)
                except Exception:
                    ts = None
                events.append({'ts': ts, 'parent': None, 'child': None, 'raw': line.strip()})
    return events


def summarize_logs(root: str, output_json: str | None, infer_prev_decision: bool):
    runs = []
    for log_path in glob(os.path.join(root, "**", "strategy.log"), recursive=True):
        run_dir = os.path.dirname(log_path)
        gem_dir = os.path.join(run_dir, 'gemini_calls')
        decisions = load_decisions(gem_dir)
        evs = scan_strategy_log(log_path)
        if not evs:
            continue
        items = []
        for ev in evs:
            n = nearest_decision(ev['ts'], decisions)
            prev_n = (n - 1) if (infer_prev_decision and isinstance(n, int) and n > 0) else None
            items.append({
                'time': ev['ts'].isoformat() if ev['ts'] else None,
                'decision': n,
                'dead_end_on_previous_decision': prev_n,
                'parent_pano_id': ev['parent'],
                'child_pano_id': ev['child'],
                'log_line': ev['raw']
            })
        runs.append({
            'run_dir': os.path.abspath(run_dir),
            'log_path': os.path.abspath(log_path),
            'events': items
        })

    # Print human-readable
    for r in runs:
        print(f"Run: {r['run_dir']}")
        for it in r['events']:
            dn = it['decision'] if it['decision'] is not None else 'N/A'
            prev = it['dead_end_on_previous_decision'] if it['dead_end_on_previous_decision'] is not None else 'N/A'
            pp = it['parent_pano_id'] or 'unknown'
            cp = it['child_pano_id'] or 'unknown'
            t  = it['time'] or 'unknown time'
            print(f"  [{t}] decision~{dn} (prev {prev}) dead_end: {pp} -> {cp}")
        print()

    if output_json:
        out = {
            'root': os.path.abspath(root),
            'runs': runs
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f"Wrote JSON to {os.path.abspath(output_json)}")


def main():
    ap = argparse.ArgumentParser(description='Scan strategy logs and summarize dead-end events per run')
    ap.add_argument('--root', required=True, help='Root directory containing run folders (searched recursively for strategy.log)')
    ap.add_argument('--json', dest='output_json', default=None, help='Optional JSON output path')
    ap.add_argument('--current-decision', action='store_true', help='Associate event with the current decision instead of previous (default maps to previous)')
    args = ap.parse_args()

    summarize_logs(args.root, args.output_json, infer_prev_decision=(not args.current_decision))


if __name__ == '__main__':
    main()


