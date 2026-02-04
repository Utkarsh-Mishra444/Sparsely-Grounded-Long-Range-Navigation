# streetview_cache_lmdb.py
import lmdb, json, os, gzip, io
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from PIL import Image

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
        self.dead_edges = self.env.open_db(b"dead_edges")

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

class DistanceCache:
    """LMDB-backed cache for Google Maps Directions API walking distance results.

    Stores walking distance calculations keyed by "start_lat,start_lng:end_lat,end_lng".
    Values are JSON-encoded dictionaries containing distance, duration, and status.
    """

    def __init__(self, path: str | Path = os.path.join("cache", "distances"), *, map_size: int = 8 << 30):
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # One database for distance results
        self.env = lmdb.open(str(path), map_size=map_size,
                             lock=True, readahead=False, max_dbs=1)
        self.db = self.env.open_db(b"distances")

    def get(self, start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> dict | None:
        """Return cached distance result for the given coordinate pair or None when missing."""
        key = f"{start_lat},{start_lng}:{end_lat},{end_lng}".encode()
        with self.env.begin(db=self.db) as txn:
            val = txn.get(key)
            if val is None:
                return None
            try:
                return json.loads(val.decode())
            except Exception:
                return None

    def put(self, start_lat: float, start_lng: float, end_lat: float, end_lng: float, result: dict):
        """Insert a distance calculation result into the cache."""
        key = f"{start_lat},{start_lng}:{end_lat},{end_lng}".encode()
        value = json.dumps(result).encode()
        with self.env.begin(write=True, db=self.db) as txn:
            txn.put(key, value, overwrite=True)

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass

    # Allow `with DistanceCache(...) as cache:` syntax
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()