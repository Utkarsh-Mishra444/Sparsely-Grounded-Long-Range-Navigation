#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

from infrastructure.cache import PanoCache


def summarize_dead_ends(db_path: str, output_json: str | None, show_coords: bool, limit: int | None):
    cache = PanoCache(path=db_path)

    summary = {
        'db_path': os.path.abspath(db_path),
        'total_parents': 0,
        'total_dead_edges': 0,
        'parents': []
    }

    # Iterate dead_edges DB directly for completeness
    with cache.env.begin(db=cache.dead_edges) as txn:
        cur = txn.cursor()
        for key, val in cur:
            try:
                parent = key.decode()
            except Exception:
                continue
            try:
                children = json.loads(val) if val else []
            except Exception:
                children = []

            # Optional enrich with coords and fully-blocked flag
            lat, lng = cache.coord_for(parent)
            try:
                links = cache.links_for(parent) or []
            except Exception:
                links = []
            link_targets = {ln.get('pano') for ln in links}
            dead_children = set(children)
            fully_blocked = bool(link_targets) and link_targets.issubset(dead_children)

            summary['parents'].append({
                'parent_pano_id': parent,
                'dead_children': sorted(dead_children),
                'dead_count': len(dead_children),
                'total_links': len(link_targets),
                'fully_blocked': fully_blocked,
                **({'lat': lat, 'lng': lng} if show_coords else {})
            })

    # Aggregate totals
    summary['total_parents'] = len(summary['parents'])
    summary['total_dead_edges'] = sum(p['dead_count'] for p in summary['parents'])

    # Sort parents by dead_count desc, then parent id
    summary['parents'].sort(key=lambda p: (-p['dead_count'], p['parent_pano_id']))

    # Print human-readable table
    print(f"Dead-end summary for LMDB: {summary['db_path']}")
    print(f"Total parents with dead edges: {summary['total_parents']}")
    print(f"Total dead edges: {summary['total_dead_edges']}")
    print()

    rows = summary['parents']
    if limit is not None:
        rows = rows[:max(0, int(limit))]

    for rec in rows:
        header = (
            f"Parent {rec['parent_pano_id']} — dead={rec['dead_count']} / links={rec['total_links']}"
            + (f" — coords=({rec['lat']:.6f},{rec['lng']:.6f})" if show_coords and rec.get('lat') is not None else "")
            + (" — FULLY BLOCKED" if rec.get('fully_blocked') else "")
        )
        print(header)
        if rec['dead_children']:
            print("  Dead children:")
            for ch in rec['dead_children']:
                print(f"    - {ch}")
        print()

    # Optionally write JSON
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote JSON summary to {os.path.abspath(output_json)}")


def main():
    ap = argparse.ArgumentParser(description="Summarize dead-end edges from the LMDB pano cache")
    ap.add_argument('--db', default=os.path.join('cache', 'pano'), help='Path to LMDB pano cache directory (default: cache/pano)')
    ap.add_argument('--json', dest='output_json', default=None, help='Optional path to write JSON summary')
    ap.add_argument('--coords', action='store_true', help='Include parent coordinates in output')
    ap.add_argument('--limit', type=int, default=None, help='Limit number of parents printed')
    args = ap.parse_args()

    summarize_dead_ends(args.db, args.output_json, args.coords, args.limit)


if __name__ == '__main__':
    main()


