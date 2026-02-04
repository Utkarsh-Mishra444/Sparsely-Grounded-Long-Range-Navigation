# streetview_cache_lmdb.py
import lmdb, json, os, gzip, io
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

def _haversine_m(a,b,c,d):
    R=6371000; dlat=radians(c-a); dlon=radians(d-b)
    return 2*R*asin(sqrt(sin(dlat/2)**2+cos(radians(a))*cos(radians(c))*sin(dlon/2)**2))

class PanoCache:
    def __init__(self, path=os.path.join("cache", "pano"), map_size=8<<30):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Increase max_dbs to accommodate a dead-edges database
        self.env = lmdb.open(path, map_size=map_size, max_dbs=3, lock=True)
        self.coords = self.env.open_db(b"coords")   # key → b"lat,lng"
        self.links  = self.env.open_db(b"links")    # key → JSON string
        # parent_pano_id → JSON list[str] of child pano IDs marked as dead-end from that parent
        # Use a versioned sub-DB name so we can start fresh without deleting older data.
        _dead_db_name = os.environ.get("DEAD_EDGES_DB_NAME", "dead_edges_v2").encode()
        self.dead_edges = self.env.open_db(_dead_db_name)

    # -------- lookup ----------
    def coord_for(self, pid):
        with self.env.begin(db=self.coords) as txn:
            val = txn.get(pid.encode())
        if not val: return (None, None)
        lat,lng = map(float, val.split(b","))
        return lat,lng

    def links_for(self, pid):
        with self.env.begin(db=self.links) as txn:
            j = txn.get(pid.encode())
        return json.loads(j) if j else []

    def nearest_pano(self, lat,lng,radius_m=30):
        with self.env.begin(db=self.coords) as txn:
            cur = txn.cursor()
            for k,v in cur:                         # linear scan ; OK for <1M
                plat,plng = map(float, v.split(b","))
                if _haversine_m(lat,lng,plat,plng) <= radius_m:
                    return k.decode()
        return None

    # -------- write -----------
    def insert_pano(self, pid, lat, lng, links):
        with self.env.begin(write=True) as txn:
            txn.put(pid.encode(),
                    f"{lat},{lng}".encode(),
                    db=self.coords, overwrite=False)
            txn.put(pid.encode(),
                    json.dumps(links).encode(),
                    db=self.links,  overwrite=False)

    def update_links(self, pid, links):
        with self.env.begin(write=True) as txn:
            txn.put(pid.encode(),
                    json.dumps(links).encode(),
                    db=self.links, overwrite=True)

    # -------- dead-end edges (shared across runs) -----------
    def dead_children_for(self, parent_pid):
        """Return a set of child pano IDs marked dead-end from parent_pid."""
        with self.env.begin(db=self.dead_edges) as txn:
            j = txn.get(parent_pid.encode())
        if not j:
            return set()
        try:
            return set(json.loads(j))
        except Exception:
            return set()

    def mark_dead_edge(self, parent_pid, child_pid):
        """Mark the directed edge parent→child as a dead-end choice.

        Idempotent and safe under concurrent writers.
        """
        with self.env.begin(write=True, db=self.dead_edges) as txn:
            key = parent_pid.encode()
            existing = txn.get(key)
            try:
                s = set(json.loads(existing)) if existing else set()
            except Exception:
                s = set()
            if child_pid not in s:
                s.add(child_pid)
                txn.put(key, json.dumps(sorted(s)).encode(), overwrite=True)

    # -------- cleanup -----------
    def close(self):
        """Close the underlying LMDB environment."""
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Path/Run store (separate LMDB under cache/, not a sub-db of pano cache)
# ──────────────────────────────────────────────────────────────────────────────
class PathRunDB:
    """LMDB-backed store for backend runs and their points.

    Schema:
      - DB "runs": key=b"run:{jobId}" → JSON dict with run metadata
      - DB "points": key=b"pt:{jobId}:{idx}" → JSON dict for a single point
    """

    def __init__(self, path=os.path.join("cache", "paths"), map_size=8 << 30):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.env = lmdb.open(path, map_size=map_size, max_dbs=3, lock=True)
        self.runs_db = self.env.open_db(b"runs")
        self.points_db = self.env.open_db(b"points")
        # manual batches of user-committed paths
        self.manual_db = self.env.open_db(b"manual")

    # ---- runs ----
    def get_run(self, job_id: str):
        key = f"run:{job_id}".encode()
        with self.env.begin(db=self.runs_db) as txn:
            val = txn.get(key)
        return json.loads(val) if val else None

    def put_run(self, job_id: str, data: dict):
        key = f"run:{job_id}".encode()
        buf = json.dumps(data, separators=(',', ':')).encode()
        with self.env.begin(write=True, db=self.runs_db) as txn:
            txn.put(key, buf, overwrite=True)

    def update_run(self, job_id: str, patch: dict):
        current = self.get_run(job_id) or {}
        current.update(patch)
        self.put_run(job_id, current)

    # ---- points ----
    def put_points(self, job_id: str, points: list[dict]):
        with self.env.begin(write=True, db=self.points_db) as txn:
            for idx, p in enumerate(points):
                key = f"pt:{job_id}:{idx}".encode()
                buf = json.dumps(p, separators=(',', ':')).encode()
                txn.put(key, buf, overwrite=True)

    def delete_run(self, job_id: str) -> int:
        """Delete a run and all its stored points. Returns number of keys removed."""
        removed = 0
        with self.env.begin(write=True) as txn:
            # Delete run record
            rkey = f"run:{job_id}".encode()
            if txn.delete(rkey, db=self.runs_db):
                removed += 1
            # Delete points with prefix
            cur = txn.cursor(db=self.points_db)
            prefix = f"pt:{job_id}:".encode()
            it = cur.set_range(prefix)
            while cur.key() and cur.key().startswith(prefix):
                try:
                    cur.delete()
                    removed += 1
                except Exception:
                    pass
                if not cur.next():
                    break
        return removed

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass

    # ---- manual user-committed paths ----
    def put_manual_batch(self, batch_id: str, paths: list[dict], meta: dict | None = None):
        key = f"batch:{batch_id}".encode()
        payload = {
            "batchId": batch_id,
            "paths": paths,
            "meta": meta or {},
        }
        buf = json.dumps(payload, separators=(',', ':')).encode()
        with self.env.begin(write=True, db=self.manual_db) as txn:
            txn.put(key, buf, overwrite=True)