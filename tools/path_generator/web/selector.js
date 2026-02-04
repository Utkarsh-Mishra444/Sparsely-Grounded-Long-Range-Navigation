// Unified Selector logic: integrates the LMDB-backed path selector from path_selector.html
// into the existing map created by script.js (initMap).

(function() {
  const SERVICE_BASE = 'http://127.0.0.1:8765';  // API server port

  // Selector state (kept separate from script.js globals)
  let selectorInitialized = false;
  let selectorMarkers = [];
  let selectorPolylines = [];
  let selectedRunIds = new Set();
  let runs = [];
  let runIdToPoints = {}; // jobId -> points[]
  let polygons = {}; // key -> { polygon, coordinates, city, name }
  let polygonColors = {}; // key -> color string
  let jobIdToColor = {}; // jobId -> color
  let selectedCity = 'ALL';
  let selectedDestination = 'ALL';
  let sortBy = 'displacement';
  let sortOrder = 'desc';
  // Live (no-persist) jobs tracked client-side
  let liveJobIds = new Set(); // jobIds started with noPersist
  let jobIdToLive = {}; // jobId -> { status, points:[{lat,lng,panoId}] }
  let suppressFitBounds = false; // avoid zoom reset during live updates
  let radiusCircles = []; // Array of circles showing bulk run radii for selected destinations
  // Destination filter state
  let destinationFilter = null; // { city: string, name: string } or null
  // Bulk operations state
  let showActiveOnly = true; // Whether to show only active runs during bulk operations
  let allPathsLoaded = false; // Whether all paths are currently loaded (for toggle button)
  let selectedBiasDirection = null; // Selected bias direction in degrees (0-360) or null for no bias
  let biasDecisionCount = 3; // Number of decisions to apply bias to

  // Initialize compass needle state
  function initializeCompassRose() {
    // Reset to no bias by default
    setCompassAngle(null);
  }

  // Update the bias direction indicator
  function updateBiasIndicator() {
    const biasText = document.getElementById('current-bias-text');
    const biasIndicator = document.querySelector('.bias-indicator');

    if (!biasText || !biasIndicator) return;

    if (selectedBiasDirection === null) {
      biasText.textContent = 'Current: No bias';
      biasIndicator.classList.remove('active');
    } else {
      // For precise angles, show the exact degree
      biasText.textContent = `Current: ${selectedBiasDirection.toFixed(1)}°`;
      biasIndicator.classList.add('active');
    }
  }

  // Set compass angle (defined later in the event listener section)
  let setCompassAngle;

  const baseColors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#9C27B0', '#00ACC1', '#8D6E63', '#7CB342', '#FF7043', '#5C6BC0'];

  function assignPolygonColors(keys) {
    for (let i = 0; i < keys.length; i++) {
      polygonColors[keys[i]] = baseColors[i % baseColors.length];
    }
  }

  function assignRunColors(runList) {
    const n = runList.length || 1;
    for (let i = 0; i < runList.length; i++) {
      const jid = runList[i].jobId;
      const hue = Math.round((360 * i) / Math.max(1, n));
      jobIdToColor[jid] = `hsl(${hue}, 85%, 50%)`;
    }
  }

  function ensureRunColor(jobId) {
    if (!jobIdToColor[jobId]) {
      const hue = Math.round(Math.random() * 360);
      jobIdToColor[jobId] = `hsl(${hue}, 85%, 50%)`;
    }
  }

  // Geometry helpers copied from path_selector
  function pointInPolygon(lat, lng, polygon) {
    let inside = false;
    const n = polygon.length;
    if (n < 3) return false;

    // Check if polygon is clockwise (most GIS polygons are clockwise)
    let sum_val = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      sum_val += (polygon[j][1] - polygon[i][1]) * (polygon[j][0] + polygon[i][0]);
    }
    const is_clockwise = sum_val > 0;

    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const yi = polygon[i][0], xi = polygon[i][1];
      const yj = polygon[j][0], xj = polygon[j][1];
      const intersect = ((xi > lng) !== (xj > lng)) && (lat < (yj - yi) * (lng - xi) / (xj - xi + 1e-12) + yi);
      if (intersect) inside = !inside;
    }

    // If polygon is clockwise, flip the result
    return is_clockwise ? !inside : inside;
  }

  function inferPolygonKey(startPoint) {
    if (!startPoint) return null;
    const lat = startPoint.lat, lng = startPoint.lng;
    // Sort keys to ensure deterministic ordering (prefer higher indices)
    const keys = Object.keys(polygons).sort((a, b) => {
      const aIdx = parseInt(a.split('_').pop()) || 0;
      const bIdx = parseInt(b.split('_').pop()) || 0;
      return bIdx - aIdx; // Higher indices first
    });
    for (const k of keys) {
      const poly = polygons[k].polygon;
      if (Array.isArray(poly) && pointInPolygon(lat, lng, poly)) return k;
    }
    return null;
  }

  async function loadPolygons() {
    try {
      const file = (window.POLYGON_FILE || null);
      const url = file ? `${SERVICE_BASE}/api/paths/polygons?file=${encodeURIComponent(file)}` : `${SERVICE_BASE}/api/paths/polygons`;
      const res = await fetch(url);
      const json = await res.json();
      polygons = json.polygons || {};
      assignPolygonColors(Object.keys(polygons));
      buildLegend();
      buildCityOptions();
    } catch (e) {
      console.error('Failed to load polygons', e);
    }
  }

  // React to polygon file changes from main UI
  try {
    window.addEventListener('polygonFileChanged', () => { loadPolygons().then(renderAll).catch(console.error); });
  } catch(_) {}

  function buildLegend() {
    const legend = document.getElementById('selector-legend-items');
    if (!legend) return;
    legend.innerHTML = '';
    const source = (window.POLYGON_FILE || 'data/polygons.json');
    const srcDiv = document.createElement('div');
    srcDiv.className = 'muted';
    srcDiv.style.marginBottom = '6px';
    srcDiv.textContent = `Polygons source: ${source}`;
    legend.appendChild(srcDiv);
    Object.entries(polygons).forEach(([key, meta]) => {
      const color = polygonColors[key] || '#9E9E9E';
      const name = meta.name || key;
      const div = document.createElement('div');
      div.className = 'legend-item';
      div.innerHTML = `<div class="legend-color" style="background-color: ${color};"></div><div>${name}</div>`;
      legend.appendChild(div);
    });
  }

  function buildCityOptions() {
    const sel = document.getElementById('selector-city-filter');
    if (!sel) return;
    const cities = Array.from(new Set(Object.values(polygons).map(p => p.city).filter(Boolean))).sort();
    sel.innerHTML = '';
    const optAll = document.createElement('option');
    optAll.value = 'ALL'; optAll.textContent = 'All Cities'; sel.appendChild(optAll);
    cities.forEach(city => { const o = document.createElement('option'); o.value = city; o.textContent = city; sel.appendChild(o); });
    buildDestinationOptions();
  }

  function buildDestinationOptions() {
    const destSel = document.getElementById('selector-destination-filter');
    if (!destSel) return;
    const dests = [];
    Object.entries(polygons).forEach(([key, meta]) => {
      if (selectedCity !== 'ALL' && meta.city !== selectedCity) return;
      const name = meta.name || key;
      dests.push({ key, name });
    });
    destSel.innerHTML = '';
    const optAll = document.createElement('option');
    optAll.value = 'ALL'; optAll.textContent = 'All Destinations'; destSel.appendChild(optAll);
    dests.sort((a,b)=> a.name.localeCompare(b.name)).forEach(({key,name}) => {
      const o = document.createElement('option'); o.value = name; o.textContent = name; destSel.appendChild(o);
    });
    destSel.disabled = dests.length === 0;
  }

  async function loadRuns() {
    try {
      const res = await fetch(`${SERVICE_BASE}/api/paths/runs`);
      const json = await res.json();
      runs = (json.runs || []).filter(r => r && typeof r === 'object' && r.jobId);
      selectedRunIds = new Set(runs.map(r => r.jobId));
      assignRunColors(runs);
    } catch (e) {
      console.error('Failed to load runs', e);
      runs = [];
    }
  }

  async function loadAllPointsForAllRuns() {
    runIdToPoints = {};
    await Promise.all(runs.map(async (r) => {
      try {
        const res = await fetch(`${SERVICE_BASE}/api/paths/points?jobId=${encodeURIComponent(r.jobId)}`);
        const json = await res.json();
        runIdToPoints[r.jobId] = json.points || [];
      } catch (e) {
        console.error('Failed to load points for', r.jobId, e);
        runIdToPoints[r.jobId] = [];
      }
    }));
  }


  async function loadAllPathsWithFeedback() {
    const btn = document.getElementById('selector-load-all-paths');
    if (!btn) return;

    // Toggle between full view and optimized view
    if (allPathsLoaded) {
      // Return to optimized view (only live paths)
      btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Optimizing...';
      btn.disabled = true;

      try {
        // Switch to live-only loading for bulk operations
        if (liveJobIds.size > 0) {
          await pollLiveBulkJobs();
        } else {
          // If no live jobs, load a minimal set or clear
          runIdToPoints = {};
        }

        allPathsLoaded = false;
        renderAll();

        // Update button text
        btn.innerHTML = '<i class="fas fa-database"></i> Load All Paths';
        btn.disabled = false;

      } catch (e) {
        console.error('Failed to optimize paths:', e);
        btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error';
        setTimeout(() => {
          btn.innerHTML = '<i class="fas fa-database"></i> Load All Paths';
          btn.disabled = false;
        }, 3000);
      }
    } else {
      // Load all paths
      const originalText = btn.innerHTML;
      btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
      btn.disabled = true;

      try {
        // Load all runs and their points
        await loadRuns();
        await loadAllPointsForAllRuns();

        allPathsLoaded = true;
        renderAll();

        // Show success and update to "optimized" state
        btn.innerHTML = '<i class="fas fa-eye-slash"></i> Show Live Only';
        btn.disabled = false;

      } catch (e) {
        console.error('Failed to load all paths:', e);
        btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Failed';
        setTimeout(() => {
          btn.innerHTML = originalText;
          btn.disabled = false;
        }, 3000);
      }
    }
  }

  function updateLoadAllPathsButton() {
    const btn = document.getElementById('selector-load-all-paths');
    if (!btn) return;

    if (allPathsLoaded) {
      btn.innerHTML = '<i class="fas fa-eye-slash"></i> Show Live Only';
    } else {
      btn.innerHTML = '<i class="fas fa-database"></i> Load All Paths';
    }
  }

  function selectorClear() {
    selectorMarkers.forEach(m => { try { m.setMap(null); } catch(e){} });
    selectorPolylines.forEach(p => { try { p.setMap(null); } catch(e){} });
    selectorMarkers = [];
    selectorPolylines = [];
  }

  function isOnBulkTab() {
    const bulkTab = document.getElementById('bulk-tab');
    return bulkTab && bulkTab.classList.contains('active');
  }

  function getVisibleRuns() {
    const baseRuns = runs.slice();
    const ephemeral = computeEphemeralRuns();

    // Check if we're on the bulk operations tab
    const onBulkTab = isOnBulkTab();

    // Determine which runs to include
    let allRuns;
    if (onBulkTab) {
      // On bulk tab: show only live jobs (jobs in memory)
      allRuns = ephemeral;
    } else if (liveJobIds.size > 0 && showActiveOnly) {
      // During bulk operations with active-only filter, show only live runs
      allRuns = ephemeral;
    } else {
      // Normal mode: show all runs (database + live)
      allRuns = baseRuns.concat(ephemeral);
    }

    // First filter by city
    let filteredRuns = allRuns;
    if (selectedCity !== 'ALL') {
      const keysByCity = new Set(Object.entries(polygons).filter(([k, v]) => v.city === selectedCity).map(([k]) => k));
      filteredRuns = allRuns.filter(r => {
        const pts = runIdToPoints[r.jobId] || [];
        const end = pts[0];
        const key = r.polygonKey || (end ? inferPolygonKey({ lat: end.lat, lng: end.lng }) : null);
        return key && keysByCity.has(key);
      });
    }

    // Destination filter from dropdown
    if (selectedDestination !== 'ALL') {
      filteredRuns = filteredRuns.filter(r => {
        const pts = runIdToPoints[r.jobId] || [];
        const end = pts[0];
        const key = r.polygonKey || (end ? inferPolygonKey({ lat: end.lat, lng: end.lng }) : null);
        if (!key) return false;
        const poly = polygons[key];
        const name = poly && (poly.name || key);
        return name === selectedDestination;
      });
    }

    // Sort by criteria
    filteredRuns.sort((a, b) => {
      const aPts = runIdToPoints[a.jobId] || [];
      const bPts = runIdToPoints[b.jobId] || [];
      const aPoints = aPts.length;
      const bPoints = bPts.length;
      const aDec = Math.max(0, aPoints - 1);
      const bDec = Math.max(0, bPoints - 1);
      const aDisp = computeDisplacementMetersSafe(aPts);
      const bDisp = computeDisplacementMetersSafe(bPts);
      let cmp = 0;
      if (sortBy === 'points') cmp = aPoints - bPoints;
      else if (sortBy === 'decisions') cmp = aDec - bDec;
      else cmp = aDisp - bDisp;
      return sortOrder === 'asc' ? cmp : -cmp;
    });

    return filteredRuns;
  }

  function computeEphemeralRuns() {
    const out = [];
    try {
      liveJobIds.forEach((jid) => {
        const pts = runIdToPoints[jid] || (jobIdToLive[jid] && Array.isArray(jobIdToLive[jid].points) ? jobIdToLive[jid].points : []);
        if (pts.length >= 2) out.push({ jobId: jid });
      });
    } catch(_) {}
    return out;
  }

  function renderAll(renderMapPaths = true) {
    // renderMapPaths: whether to render paths on the map (default: true)
    // Set to false during initialization to avoid cluttering the map
    selectorClear();
    const visible = getVisibleRuns();
    if (renderMapPaths) {
      renderPaths(visible);
    }
    populateSidebar(visible);
  }

  // Expose renderAll to global window for HTML script access
  window.renderAll = renderAll;

  function drawRadiusCircle(radiusMeters) {
    // Clear existing circles
    clearRadiusCircle();

    // Check if map is available (use global map variable, not window.map)
    if (!window.google || typeof map === 'undefined' || !map) {
      console.log('Map not ready for radius circles');
      return;
    }

    // Get selected destinations
    const selectedDestinations = getSelectedBulkDestinations();
    if (selectedDestinations.length === 0) {
      console.log('No destinations selected for radius circles');
      return;
    }

    const seeds = (typeof window !== 'undefined' && window.SEED_PANOS_BY_LANDMARK) ? window.SEED_PANOS_BY_LANDMARK : {};

    // Draw circles for each selected destination
    selectedDestinations.forEach((dest, index) => {
      const key = `${dest.city}_${dest.name}`;
      const seedList = seeds[key];

      if (!Array.isArray(seedList) || seedList.length === 0) {
        console.log(`No seeds found for ${key}`);
        return;
      }

      // Use the first seed pano as center for this destination
      const firstSeed = seedList[0];
      if (!firstSeed.position) {
        console.log(`No position found for first seed in ${key}`);
        return;
      }

      try {
        // Use different colors for different destinations
        const colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'];
        const color = colors[index % colors.length];

        const circle = new google.maps.Circle({
          map: map,
          center: new google.maps.LatLng(firstSeed.position.lat, firstSeed.position.lng),
          radius: radiusMeters,
          fillColor: color,
          fillOpacity: 0.1,
          strokeColor: color,
          strokeOpacity: 0.8,
          strokeWeight: 2,
          clickable: false
        });

        radiusCircles.push(circle);
      } catch (e) {
        console.log(`Error creating radius circle for ${key}:`, e);
      }
    });

    if (radiusCircles.length === 0) {
      console.log('No radius circles were created');
    } else {
      console.log(`Created ${radiusCircles.length} radius circles`);
    }
  }

  function clearRadiusCircle() {
    radiusCircles.forEach(circle => {
      try {
        if (circle) {
          circle.setMap(null);
        }
      } catch (e) {
        console.log('Error clearing radius circle:', e);
      }
    });
    radiusCircles = [];
  }

  function computeDisplacementMetersSafe(pts) {
    try {
      if (!Array.isArray(pts) || pts.length < 2) return 0;
      // Consistent with renderPaths: start is tail, end is destination (index 0)
      const end = pts[0];
      const start = pts[pts.length - 1];
      // Haversine distance (meters)
      const R = 6371000;
      const toRad = (d) => d * Math.PI / 180;
      const dLat = toRad(end.lat - start.lat);
      const dLng = toRad(end.lng - start.lng);
      const a = Math.sin(dLat/2)**2 + Math.cos(toRad(start.lat)) * Math.cos(toRad(end.lat)) * Math.sin(dLng/2)**2;
      const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
      return R * c;
    } catch(_) { return 0; }
  }

  function renderPaths(visibleRuns) {
    if (!window.google || typeof map === 'undefined' || !map) return;
    const bounds = new google.maps.LatLngBounds();
    // Re-index for labels in current view
    const idxMap = {};
    visibleRuns.forEach((r, i) => { idxMap[r.jobId] = i + 1; });
    visibleRuns.forEach((run) => {
      const jobId = run.jobId;
      const points = runIdToPoints[jobId] || [];
      if (!points.length) return;
      const end = points[0]; const start = points[points.length - 1]; // Fixed: end is destination, start is tail of path
      const color = jobIdToColor[jobId] || '#9C27B0';
      const isSelected = selectedRunIds.has(jobId);
      // Gradient segments for directionality (much more visible)
      const segs = [];
      try {
        const n = points.length;
        const maxSeg = n - 1;
        for (let i = 0; i < maxSeg; i++) {
          const a = points[i], b = points[i+1];
          const frac = i / Math.max(1, maxSeg);
          const opacity = (isSelected ? 0.8 : 0.5) + 0.2 * frac; // Much more visible tails
          const hue = 200 + Math.round(140 * frac); // teal→purple
          const gradColor = `hsl(${hue}, 85%, 65%)`; // Brighter colors
          const seg = new google.maps.Polyline({
            map,
            path: [new google.maps.LatLng(a.lat, a.lng), new google.maps.LatLng(b.lat, b.lng)],
            strokeColor: gradColor,
            strokeWeight: isSelected ? 6 : 4, // Slightly thicker
            strokeOpacity: opacity
          });
          segs.push(seg);
          selectorPolylines.push(seg);
        }
      } catch(_) {}
      // Start marker (circle) at actual starting point, End marker (arrow) at destination
      const sm = new google.maps.Marker({ map, position: { lat: start.lat, lng: start.lng }, icon: { path: google.maps.SymbolPath.CIRCLE, scale: 8, fillColor: color, fillOpacity: isSelected ? 1.0 : 0.5, strokeColor: '#fff', strokeWeight: 2 }, label: String(idxMap[jobId] || ''), title: `Start ${start.panoId}` });
      const em = new google.maps.Marker({ map, position: { lat: end.lat, lng: end.lng }, icon: { path: google.maps.SymbolPath.BACKWARD_CLOSED_ARROW, scale: 5, fillColor: color, fillOpacity: isSelected ? 1.0 : 0.5, strokeColor: '#fff', strokeWeight: 2 }, title: `End ${end.panoId}` });
      selectorMarkers.push(sm, em);
      try { bounds.extend(sm.getPosition()); bounds.extend(em.getPosition()); } catch(_){}
      segs.forEach(seg => seg.addListener('click', () => toggleRun(jobId)));
      sm.addListener('click', () => toggleRun(jobId));
      em.addListener('click', () => toggleRun(jobId));
    });
    try {
      if (!suppressFitBounds && !bounds.isEmpty()) {
        map.fitBounds(bounds, 60);
      }
    } catch(_){ }
    suppressFitBounds = false;
  }

  function populateSidebar(visibleRuns) {
    const list = document.getElementById('selector-path-list');
    if (!list) return;
    list.innerHTML = '';
    visibleRuns.forEach((run, idx) => {
      const jobId = run.jobId;
      const pts = runIdToPoints[jobId] || [];
      if (!pts.length) return;
      const end = pts[0]; const start = pts[pts.length - 1]; // Fixed: end is destination polygon, start is tail of path
      const polyKey = run.polygonKey || inferPolygonKey({ lat: end.lat, lng: end.lng }) || 'Other';
      const color = jobIdToColor[jobId] || '#9C27B0';
      const isSelected = selectedRunIds.has(jobId);
      const item = document.createElement('div');
      item.className = `path-item ${isSelected ? 'selected' : ''}`;
      item.dataset.jobId = jobId;
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox'; checkbox.className = 'path-checkbox'; checkbox.checked = isSelected;
      const destName = (polygons[polyKey] && polygons[polyKey].name) || polyKey;
      const cityName = (polygons[polyKey] && polygons[polyKey].city) || 'Unknown';
      const idxNum = (idx + 1);
      const numPoints = pts.length;
      const numDecisions = Math.max(0, numPoints - 1);
      const dispMeters = computeDisplacementMetersSafe(pts);
      const dispStr = (dispMeters >= 1000) ? `${(dispMeters/1000).toFixed(2)} km` : `${Math.round(dispMeters)} m`;
      item.innerHTML = `
        <div style="border-left: 4px solid ${color}; padding-left: 10px;">
          <h3>#${idxNum} ${cityName} → ${destName}</h3>
          <div class="path-details" style="display:flex; gap:10px; flex-wrap:wrap;">
            <div><strong>Points</strong>: ${numPoints}</div>
            <div><strong>Decisions</strong>: ${numDecisions}</div>
            <div><strong>Displacement</strong>: ${dispStr}</div>
          </div>
          <div style="margin-top:6px; display:flex; gap:8px;">
            <button class="btn btn-small" data-action="delete" data-job="${jobId}" style="background:#c62828;">Delete</button>
          </div>
        </div>`;
      item.appendChild(checkbox);
      item.addEventListener('click', (e) => { if (e.target !== checkbox) toggleRun(jobId); });
      checkbox.addEventListener('change', () => toggleRun(jobId));
      list.appendChild(item);
    });
    updateSelectAllButton(visibleRuns.length);

    // Wire delete buttons
    list.querySelectorAll('button[data-action="delete"]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const jid = e.currentTarget.getAttribute('data-job');
        if (!jid) return;
        if (!confirm(`Delete run ${jid}? This will remove it from the database.`)) return;
        try {
          const res = await fetch(`${SERVICE_BASE}/api/paths/delete`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ jobId: jid }) });
          const json = await res.json();
          if (!res.ok) { alert('Delete failed: ' + (json.error || res.status)); return; }
          // Remove from in-memory state and re-render
          runs = runs.filter(r => r.jobId !== jid);
          delete runIdToPoints[jid];
          selectedRunIds.delete(jid);
          renderAll();
        } catch (err) {
          alert('Delete error: ' + (err && err.message ? err.message : String(err)));
        }
      });
    });
  }

  function toggleRun(jobId) {
    if (selectedRunIds.has(jobId)) selectedRunIds.delete(jobId); else selectedRunIds.add(jobId);
    renderAll();
  }

  function updateSelectAllButton(visibleCount) {
    const btn = document.getElementById('selector-select-all');
    if (!btn) return;
    btn.textContent = (selectedRunIds.size === visibleCount && visibleCount) ? 'Deselect All' : 'Select All';
  }

  function toggleSelectAll() {
    const allIds = getVisibleRuns().map(r => r.jobId);
    if (selectedRunIds.size === allIds.length) {
      selectedRunIds.clear();
    } else {
      selectedRunIds = new Set(allIds);
    }
    renderAll();
  }

  function unselectAll() {
    selectedRunIds.clear();

    // Also uncheck all checkboxes in the sidebar
    const pathItems = document.querySelectorAll('#selector-path-list .path-item');
    pathItems.forEach(item => {
      item.classList.remove('selected');
      const checkbox = item.querySelector('.path-checkbox input[type="checkbox"]');
      if (checkbox) {
        checkbox.checked = false;
      }
    });

    renderAll();
  }

  function downloadJSON() {
    if (!selectedRunIds.size) { alert('Please select at least one path'); return; }
    const out = [];
    const visible = getVisibleRuns();
    visible.forEach(run => {
      if (!selectedRunIds.has(run.jobId)) return;
      const pts = runIdToPoints[run.jobId] || [];
      if (!pts.length) return;
      const end = pts[0]; // Destination polygon (where pathfinding started)
      const start = pts[pts.length - 1]; // Actual starting point (tail of path)
      const polyKey = run.polygonKey || inferPolygonKey({ lat: end.lat, lng: end.lng });
      if (!polyKey) return;
      const meta = polygons[polyKey] || {};
      out.push({
        jobId: run.jobId,
        startPanoId: start.panoId,
        start: { lat: start.lat, lng: start.lng },
        endPanoId: end.panoId,
        end: { lat: end.lat, lng: end.lng },
        destinationPolygonKey: polyKey,
        destinationName: meta.name || polyKey,
        destinationCity: meta.city || null,
        destinationPolygon: meta.polygon || null
      });
    });
    const blob = new Blob([JSON.stringify({ routes: out }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'selected_routes.json';
    document.body.appendChild(a); a.click();
    setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
  }

  async function pollLiveBulkJobs() {
    const ids = Array.from(liveJobIds);
    if (!ids.length) return;
    await Promise.all(ids.map(async (jid) => {
      try {
        const res = await fetch(`${SERVICE_BASE}/api/route/status?jobId=${encodeURIComponent(jid)}`);
        if (res.status === 404) { liveJobIds.delete(jid); return; }
        const js = await res.json();
        const status = js && js.status ? js.status : 'unknown';
        const pts = Array.isArray(js.points) ? js.points.map(p => ({ lat: p.position.lat, lng: p.position.lng, panoId: p.panoId })) : [];
        jobIdToLive[jid] = { status, points: pts };
        runIdToPoints[jid] = pts;
      } catch(_) {}
    }));
  }

  async function commitLiveJobs() {
    const finished = Array.from(liveJobIds).filter(jid => {
      const m = jobIdToLive[jid];
      return m && m.status && m.status !== 'running' && Array.isArray(m.points) && m.points.length >= 2;
    });
    if (!finished.length) { alert('No finished live runs to commit.'); return; }
    try {
      const res = await fetch(`${SERVICE_BASE}/api/paths/manual_commit`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ jobIds: finished }) });
      const js = await res.json();
      if (!res.ok) { alert('Commit failed: ' + (js.error || res.status)); return; }
      finished.forEach(jid => { liveJobIds.delete(jid); delete jobIdToLive[jid]; });

      // Stop bulk polling if no live jobs remain
      if (liveJobIds.size === 0 && window.__bulkPoller) {
        clearInterval(window.__bulkPoller);
        window.__bulkPoller = null;
      }

      await loadRuns();
      await loadAllPointsForAllRuns();
      suppressFitBounds = true;
      renderAll();

      // After committing, all paths are loaded
      allPathsLoaded = true;
      updateLoadAllPathsButton();

      alert(`Committed ${js.committed || finished.length} run(s) to DB.`);
    } catch (e) {
      alert('Commit error: ' + (e && e.message ? e.message : String(e)));
    }
  }

  async function showLiveStatsPopup() {
    // Create or update stats popup
    let statsPopup = document.getElementById('live-stats-popup');
    if (!statsPopup) {
      statsPopup = document.createElement('div');
      statsPopup.id = 'live-stats-popup';
      statsPopup.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        border: 2px solid #333;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        z-index: 10000;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
        font-family: monospace;
      `;
      document.body.appendChild(statsPopup);

      // Add close button
      const closeBtn = document.createElement('button');
      closeBtn.textContent = '×';
      closeBtn.style.cssText = `
        position: absolute;
        top: 10px;
        right: 10px;
        background: #f44336;
        color: white;
        border: none;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        cursor: pointer;
        font-size: 18px;
        font-weight: bold;
      `;
      closeBtn.onclick = () => statsPopup.remove();
      statsPopup.appendChild(closeBtn);

      // Add refresh button
      const refreshBtn = document.createElement('button');
      refreshBtn.textContent = '🔄 Refresh';
      refreshBtn.style.cssText = `
        margin-left: 10px;
        padding: 5px 10px;
        background: #2196F3;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
      `;
      refreshBtn.onclick = () => updateStatsContent(statsPopup);
      statsPopup.appendChild(refreshBtn);
    }

    updateStatsContent(statsPopup);
  }

  async function updateStatsContent(popup) {
    // Clear previous content (except buttons)
    const buttons = popup.querySelectorAll('button');
    popup.innerHTML = '';
    buttons.forEach(btn => popup.appendChild(btn));

    // Title
    const title = document.createElement('h3');
    title.textContent = 'Live Run Statistics';
    title.style.marginTop = '10px';
    popup.appendChild(title);

    // Stats content
    const statsDiv = document.createElement('div');
    statsDiv.style.marginTop = '10px';

    // Count running jobs
    let runningCount = 0;
    let doneCount = 0;
    let errorCount = 0;
    let totalPoints = 0;
    let totalDistance = 0;

    // Check live jobs
    for (const jid of liveJobIds) {
      const meta = jobIdToLive[jid];
      if (!meta) continue;

      const pts = Array.isArray(meta.points) ? meta.points : [];
      const status = meta.status || 'unknown';

      if (status === 'running') runningCount++;
      else if (status === 'done') doneCount++;
      else if (status === 'error') errorCount++;

      totalPoints += pts.length;
      if (pts.length >= 2) {
        const end = pts[0];
        const start = pts[pts.length - 1];
        totalDistance += computeDisplacementMetersSafe([end, start]);
      }
    }

    // Add stats
    statsDiv.innerHTML = `
      <div style="margin-bottom: 15px;">
        <strong>Live Jobs:</strong> ${liveJobIds.size}<br>
        <strong>Running:</strong> <span style="color:#2196F3;">${runningCount}</span><br>
        <strong>Done:</strong> <span style="color:#4CAF50;">${doneCount}</span><br>
        <strong>Errors:</strong> <span style="color:#f44336;">${errorCount}</span><br>
        <strong>Total Points:</strong> ${totalPoints}<br>
        <strong>Total Distance:</strong> ${(totalDistance / 1000).toFixed(2)} km
      </div>

      <div style="border-top: 1px solid #ccc; padding-top: 10px;">
        <strong>Individual Job Status:</strong>
      </div>
    `;

    // Individual job details
    const jobDetailsDiv = document.createElement('div');
    jobDetailsDiv.style.marginTop = '10px';
    jobDetailsDiv.style.maxHeight = '300px';
    jobDetailsDiv.style.overflowY = 'auto';

    for (const jid of liveJobIds) {
      const meta = jobIdToLive[jid];
      if (!meta) continue;

      const pts = Array.isArray(meta.points) ? meta.points : [];
      const status = meta.status || 'unknown';
      const color = status === 'running' ? '#2196F3' :
                   status === 'done' ? '#4CAF50' :
                   status === 'error' ? '#f44336' : '#9E9E9E';

      const jobDiv = document.createElement('div');
      jobDiv.style.cssText = `
        margin: 5px 0;
        padding: 8px;
        border-left: 3px solid ${color};
        background: #f9f9f9;
        font-size: 12px;
      `;

      let dist = 0;
      if (pts.length >= 2) {
        const end = pts[0];
        const start = pts[pts.length - 1];
        dist = computeDisplacementMetersSafe([end, start]);
      }

      jobDiv.innerHTML = `
        <strong>${jid.slice(-8)}:</strong> ${status}<br>
        Points: ${pts.length} | Distance: ${(dist / 1000).toFixed(2)} km
      `;
      jobDetailsDiv.appendChild(jobDiv);
    }

    statsDiv.appendChild(jobDetailsDiv);
    popup.appendChild(statsDiv);

    // Auto-refresh every 2 seconds if popup is still open
    if (document.getElementById('live-stats-popup')) {
      setTimeout(() => {
        if (document.getElementById('live-stats-popup')) {
          updateStatsContent(popup);
        }
      }, 2000);
    }
  }

  async function initSelectorOnce() {
    if (selectorInitialized) return;
    selectorInitialized = true;
    // Wire UI
    const dl = document.getElementById('selector-download-json');
    const selAll = document.getElementById('selector-select-all');
    const unselAll = document.getElementById('selector-unselect-all');
    const delSel = document.getElementById('selector-delete-selected');
    const citySel = document.getElementById('selector-city-filter');
    const bulkList = document.getElementById('bulk-list');
    const bulkRefreshBtn = document.getElementById('bulk-refresh');
    const bulkValidateBtn = document.getElementById('bulk-validate-seeds');
    const bulkStartBtn = document.getElementById('bulk-start');
    const bulkSelectAllBtn = document.getElementById('bulk-select-all');
    const bulkUnselectAllBtn = document.getElementById('bulk-unselect-all');
    const bulkCommitBtn = document.getElementById('bulk-commit-live');
    const bulkTargetInput = document.getElementById('bulk-target-distance');
    const bulkRunsInput = document.getElementById('bulk-runs-per-dest');
    const bulkMinExistingInput = document.getElementById('bulk-min-existing');
    const bulkBiasDecisionsInput = document.getElementById('bulk-bias-decisions');
    const compassRose = document.getElementById('compass-rose');
    const bulkStatsBtn = document.getElementById('bulk-stats');
    const bulkShowActiveOnlyToggle = document.getElementById('bulk-show-active-only');
    const loadAllPathsBtn = document.getElementById('selector-load-all-paths');
    // Sort & destination filter inputs
    const destFilterSel = document.getElementById('selector-destination-filter');
    const sortBySel = document.getElementById('selector-sort-by');
    const sortOrderSel = document.getElementById('selector-sort-order');
    if (dl) dl.addEventListener('click', downloadJSON);
    if (selAll) selAll.addEventListener('click', toggleSelectAll);
    if (unselAll) unselAll.addEventListener('click', unselectAll);
    if (delSel) delSel.addEventListener('click', deleteSelectedRuns);
    if (loadAllPathsBtn) loadAllPathsBtn.addEventListener('click', loadAllPathsWithFeedback);
    if (bulkRefreshBtn) bulkRefreshBtn.addEventListener('click', () => {
      if (typeof window.manualRefreshBulkStatus === 'function') {
        window.manualRefreshBulkStatus();
      } else {
        renderBulkPanel();
      }
    });
    if (bulkStartBtn) bulkStartBtn.addEventListener('click', startBulkFromSeeds);
    if (bulkValidateBtn) bulkValidateBtn.addEventListener('click', validateSeeds);
    if (bulkSelectAllBtn) bulkSelectAllBtn.addEventListener('click', () => {
      toggleBulkSelection(true);
      // Update circles after selection change
      setTimeout(() => {
        const radius = parseFloat(bulkTargetInput?.value) || 0;
        if (radius > 0) {
          drawRadiusCircle(radius);
        }
      }, 10);
    });
    if (bulkUnselectAllBtn) bulkUnselectAllBtn.addEventListener('click', () => {
      toggleBulkSelection(false);
      clearRadiusCircle();
    });
    if (bulkCommitBtn) bulkCommitBtn.addEventListener('click', commitLiveJobs);
    if (bulkStatsBtn) bulkStatsBtn.addEventListener('click', showLiveStatsPopup);
    if (bulkShowActiveOnlyToggle) {
      bulkShowActiveOnlyToggle.checked = showActiveOnly;
      bulkShowActiveOnlyToggle.addEventListener('change', (e) => {
        showActiveOnly = e.target.checked;
        renderAll(); // Re-render with new filter setting
      });
    }

    // Destination dropdown change
    if (destFilterSel) {
      destFilterSel.addEventListener('change', (e) => {
        selectedDestination = e.target.value || 'ALL';
        renderAll();
      });
    }

    // Sort controls
    if (sortBySel) {
      sortBySel.value = sortBy;
      sortBySel.addEventListener('change', (e) => { sortBy = e.target.value; renderAll(); });
    }
    if (sortOrderSel) {
      sortOrderSel.value = sortOrder;
      sortOrderSel.addEventListener('change', (e) => { sortOrder = e.target.value; renderAll(); });
    }

    // Compass needle functionality
    const compassSvg = document.getElementById('compass-svg');
    const compassNeedle = document.getElementById('compass-needle');
    const resetCompassBtn = document.getElementById('reset-compass');
    const compassPresets = document.querySelectorAll('.compass-preset');

    let isDragging = false;
    let currentAngle = null;

    // Compass interaction handlers
    if (compassSvg) {
      const handlePointerDown = (e) => {
        isDragging = true;
        updateCompassAngle(e);
        document.addEventListener('pointermove', handlePointerMove);
        document.addEventListener('pointerup', handlePointerUp);
        e.preventDefault();
      };

      const handlePointerMove = (e) => {
        if (isDragging) {
          updateCompassAngle(e);
        }
      };

      const handlePointerUp = () => {
        isDragging = false;
        document.removeEventListener('pointermove', handlePointerMove);
        document.removeEventListener('pointerup', handlePointerUp);
      };

      const updateCompassAngle = (e) => {
        const rect = compassSvg.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        const deltaX = e.clientX - centerX;
        const deltaY = e.clientY - centerY;

        // Calculate angle in degrees (0° = North, clockwise)
        let angle = Math.atan2(deltaX, -deltaY) * (180 / Math.PI);
        angle = (angle + 360) % 360; // Ensure positive angle

        setCompassAngle(angle);
      };

      compassSvg.addEventListener('pointerdown', handlePointerDown);

      // Touch support
      compassSvg.addEventListener('touchstart', (e) => {
        e.preventDefault();
        if (e.touches.length === 1) {
          const touch = e.touches[0];
          const pointerEvent = new PointerEvent('pointerdown', {
            clientX: touch.clientX,
            clientY: touch.clientY,
            pointerId: touch.identifier
          });
          handlePointerDown(pointerEvent);
        }
      });
    }

    // Reset compass button
    if (resetCompassBtn) {
      resetCompassBtn.addEventListener('click', () => {
        setCompassAngle(null);
      });
    }

    // Preset buttons
    if (compassPresets) {
      compassPresets.forEach(button => {
        button.addEventListener('click', () => {
          const angle = parseFloat(button.getAttribute('data-angle'));
          setCompassAngle(angle);
        });
      });
    }

    setCompassAngle = function(angle) {
      currentAngle = angle;
      selectedBiasDirection = angle;

      // Update needle rotation
      if (compassNeedle) {
        compassNeedle.setAttribute('transform', `rotate(${angle || 0}, 100, 100)`);
      }

      // Update preset button states
      compassPresets.forEach(btn => {
        const btnAngle = parseFloat(btn.getAttribute('data-angle'));
        if (angle !== null && Math.abs(btnAngle - angle) < 1) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });

      // Update bias indicator
      updateBiasIndicator();
    };

    // Bias decisions input
    if (bulkBiasDecisionsInput) {
      bulkBiasDecisionsInput.value = biasDecisionCount;
      bulkBiasDecisionsInput.addEventListener('input', (e) => {
        biasDecisionCount = Math.max(0, parseInt(e.target.value) || 0);
      });
    }
    if (citySel) citySel.addEventListener('change', (e) => {
      selectedCity = e.target.value || 'ALL';
      // Clear destination filter when city changes
      destinationFilter = null;
      // Remove filter message if it exists
      const message = document.getElementById('destination-filter-message');
      if (message) {
        message.remove();
      }
      // Rebuild destination options and reset selection
      buildDestinationOptions();
      selectedDestination = 'ALL';
      const destSelect = document.getElementById('selector-destination-filter');
      if (destSelect) destSelect.value = 'ALL';
      renderAll();
    });

    // Add radius circle drawing for bulk target distance and destination selection
    if (bulkTargetInput) {
      const updateCircles = () => {
        const radius = parseFloat(bulkTargetInput.value) || 0;
        if (radius > 0) {
          drawRadiusCircle(radius);
        } else {
          clearRadiusCircle();
        }
      };

      bulkTargetInput.addEventListener('input', updateCircles);
      bulkTargetInput.addEventListener('blur', () => {
        // Clear circles when input loses focus
        clearRadiusCircle();
      });

      // Also update circles when destination checkboxes change
      const bulkList = document.getElementById('bulk-list');
      if (bulkList) {
        bulkList.addEventListener('change', (e) => {
          if (e.target.classList.contains('bulk-dest')) {
            // Small delay to let checkbox state update
            setTimeout(updateCircles, 10);
          }
        });
      }
    }

    await loadPolygons();
    await loadRuns();
    await loadAllPointsForAllRuns();
    renderAll(false); // Don't render paths on map initially
    renderBulkPanel();
    initializeCompassRose();
  }

  function groupSeedPanosByCity() {
    const seeds = (typeof window !== 'undefined' && window.SEED_PANOS_BY_LANDMARK) ? window.SEED_PANOS_BY_LANDMARK : {};
    const grouped = {};
    Object.entries(seeds).forEach(([key, list]) => {
      if (!Array.isArray(list) || !list.length) return;
      const [city, landmark] = key.split('_', 2);
      if (!grouped[city]) grouped[city] = {};
      grouped[city][landmark] = list;
    });
    return grouped;
  }

  function validateSeeds() {
    const report = [];
    const seeds = (typeof window !== 'undefined' && window.SEED_PANOS_BY_LANDMARK) ? window.SEED_PANOS_BY_LANDMARK : {};
    const keys = Object.keys(seeds);
    if (!keys.length) { alert('No seed panos saved.'); return; }
    // Build quick polygon map for inference
    keys.forEach(key => {
      const list = seeds[key] || [];
      let ok = 0, bad = 0;
      list.forEach(s => {
        if (!s || !s.position) { bad++; return; }
        const k = inferPolygonKey({ lat: s.position.lat, lng: s.position.lng });
        if (k) ok++; else bad++;
      });
      report.push(`${key}: valid ${ok}, invalid ${bad}`);
    });
    alert(report.join('\n'));
  }

  function computeExistingRunsByDest() {
    // Use polygonKey inferred from endpoints to count runs per destination key
    const counts = {}; // city -> landmark -> count
    runs.forEach(r => {
      const pts = runIdToPoints[r.jobId] || [];
      if (!pts.length) return;
      const end = pts[0];
      const key = r.polygonKey || inferPolygonKey({ lat: end.lat, lng: end.lng });
      if (!key) return;
      const meta = polygons[key];
      const city = meta && meta.city ? meta.city : 'Unknown';
      const name = meta && meta.name ? meta.name : key;
      counts[city] = counts[city] || {};
      counts[city][name] = (counts[city][name] || 0) + 1;
    });
    return counts;
  }

  function renderBulkPanel() {
    const container = document.getElementById('bulk-list');
    if (!container) return;
    container.innerHTML = '';
    const seedsByCity = groupSeedPanosByCity();
    const runCounts = computeExistingRunsByDest();
    const cities = Object.keys(seedsByCity).sort();
    if (!cities.length) { container.textContent = 'No seed panos saved.'; return; }
    const wrap = document.createElement('div');
    cities.forEach(city => {
      const cityDiv = document.createElement('div');
      cityDiv.style.marginBottom = '8px';
      const title = document.createElement('div');
      title.innerHTML = `<strong>${city}</strong>`;
      cityDiv.appendChild(title);
      const table = document.createElement('table');
      table.style.margin = '4px 0 0 16px';
      table.style.width = '100%';
      table.style.borderCollapse = 'collapse';
      table.innerHTML = `
        <thead>
          <tr style="border-bottom: 1px solid #ccc;">
            <th style="text-align: left; padding: 4px;">Select</th>
            <th style="text-align: left; padding: 4px;">Landmark</th>
            <th style="text-align: left; padding: 4px;">Runs</th>
            <th style="text-align: left; padding: 4px;">Seeds</th>
          </tr>
        </thead>
        <tbody>
      `;
      Object.keys(seedsByCity[city]).sort().forEach(name => {
        const count = (runCounts[city] && runCounts[city][name]) ? runCounts[city][name] : 0;
        const id = `bulk-${city}-${name}`.replace(/[^a-z0-9_-]/gi,'_');
        const row = document.createElement('tr');
        const nameCell = document.createElement('td');
        nameCell.style.padding = '4px';
        const nameLink = document.createElement('span');
        nameLink.textContent = name;
        nameLink.style.cursor = 'pointer';
        nameLink.style.color = '#2563eb';
        nameLink.style.textDecoration = 'underline';
        nameLink.addEventListener('click', () => showRunsForDestination(city, name));
        nameCell.appendChild(nameLink);

        row.innerHTML = `
          <td style="padding: 4px;"><label><input type="checkbox" class="bulk-dest" id="${id}" data-city="${city}" data-name="${name}" checked /></label></td>
        `;
        row.appendChild(nameCell);
        row.innerHTML += `
          <td style="padding: 4px;">${count}</td>
          <td style="padding: 4px;">${seedsByCity[city][name].length}</td>
        `;
        table.querySelector('tbody').appendChild(row);
      });
      cityDiv.appendChild(table);
      wrap.appendChild(cityDiv);
    });
    container.appendChild(wrap);
    // Show quick status for live (no-persist) jobs
    try {
      if (liveJobIds.size) {
        let runningCount = 0, doneCount = 0, errorCount = 0;
        for (const jid of liveJobIds) {
          const meta = jobIdToLive[jid];
          const status = meta ? meta.status : 'unknown';
          if (status === 'running') runningCount++;
          else if (status === 'done') doneCount++;
          else if (status === 'error') errorCount++;
        }

        const info = document.createElement('div');
        info.className = 'muted';
        info.style.marginTop = '6px';
        info.innerHTML = `
          Live runs: <strong>${liveJobIds.size}</strong> |
          <span style="color:#2196F3;">Running: ${runningCount}</span> |
          <span style="color:#4CAF50;">Done: ${doneCount}</span> |
          <span style="color:#f44336;">Errors: ${errorCount}</span>
          <br><small>Use "Commit Live to DB" to persist finished runs.</small>
        `;
        container.appendChild(info);
      }
    } catch(_) {}
  }

  function getSelectedBulkDestinations() {
    const container = document.getElementById('bulk-list');
    if (!container) return [];
    const nodes = Array.from(container.querySelectorAll('input.bulk-dest'));
    return nodes.filter(n => n.checked).map(n => ({ city: n.getAttribute('data-city'), name: n.getAttribute('data-name') }));
  }

  function toggleBulkSelection(value) {
    const container = document.getElementById('bulk-list');
    if (!container) return;
    container.querySelectorAll('input.bulk-dest').forEach(n => { n.checked = !!value; });
  }

  function showRunsForDestination(city, landmarkName) {
    // Switch to the Paths tab
    const pathsTab = document.querySelector('[data-tab="paths"]');
    if (pathsTab) {
      pathsTab.click();
    }

    // Set destination filter
    destinationFilter = { city: city, name: landmarkName };

    // Set city filter to the selected city
    const cityFilter = document.getElementById('selector-city-filter');
    if (cityFilter) {
      cityFilter.value = city;
      selectedCity = city;
    }

    // Clear current selections to avoid cluttering the map
    selectedRunIds.clear();

    // Get filtered runs for display count
    const visibleRuns = getVisibleRuns();

    // Re-render everything with the filter applied
    renderAll();

    // Show a message about the filter
    const filterMessage = document.createElement('div');
    filterMessage.id = 'destination-filter-message';
    filterMessage.style.cssText = `
      background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
      border: 1px solid #0ea5e9;
      border-radius: 8px;
      padding: 1rem;
      margin: 1rem 0;
      color: #0ea5e9;
      font-weight: 500;
    `;
    filterMessage.innerHTML = `
      <i class="fas fa-filter"></i>
      Showing ${visibleRuns.length} runs for: <strong>${landmarkName}</strong> in <strong>${city}</strong>
      <span style="margin-left: 1rem; font-size: 0.9rem; color: #64748b;">(Click paths individually to plot them)</span>
      <button id="clear-destination-filter" style="margin-left: 1rem; background: none; border: none; color: #0ea5e9; text-decoration: underline; cursor: pointer;">
        <i class="fas fa-times"></i> Clear filter
      </button>
    `;

    // Remove any existing filter message
    const existingMessage = document.getElementById('destination-filter-message');
    if (existingMessage) {
      existingMessage.remove();
    }

    // Add the filter message at the top of the paths section
    const pathsTabContent = document.getElementById('paths-tab');
    if (pathsTabContent) {
      pathsTabContent.insertBefore(filterMessage, pathsTabContent.firstChild);

      // Add event listener for clear button
      const clearBtn = document.getElementById('clear-destination-filter');
      if (clearBtn) {
        clearBtn.addEventListener('click', clearDestinationFilter);
      }
    }
  }

  function clearDestinationFilter() {
    // Clear destination filter
    destinationFilter = null;

    // Remove filter message
    const message = document.getElementById('destination-filter-message');
    if (message) {
      message.remove();
    }

    // Reset city filter to ALL
    const cityFilter = document.getElementById('selector-city-filter');
    if (cityFilter) {
      cityFilter.value = 'ALL';
      selectedCity = 'ALL';
    }

    // Re-render everything with filters cleared
    renderAll();
  }

  async function startBulkFromSeeds() {
    // Clear the radius circle when starting bulk run
    clearRadiusCircle();

    const targetMeters = Math.max(10, Number(document.getElementById('bulk-target-distance')?.value || 2000) || 2000);
    const runsPerDest = Math.max(1, Number(document.getElementById('bulk-runs-per-dest')?.value || 10) || 10);
    const minExisting = Math.max(0, Number(document.getElementById('bulk-min-existing')?.value || 10) || 10);
    const seeds = (typeof window !== 'undefined' && window.SEED_PANOS_BY_LANDMARK) ? window.SEED_PANOS_BY_LANDMARK : {};
    if (!Object.keys(seeds).length) { alert('No seed panos saved.'); return; }

    // Build worklist of destinations (respect selection)
    const runCounts = computeExistingRunsByDest();
    let totalScheduled = 0;
    const selected = getSelectedBulkDestinations();
    const selectedSet = new Set(selected.map(({city,name}) => `${city}_${name}`));
    for (const [key, seedList] of Object.entries(seeds)) {
      if (!Array.isArray(seedList) || !seedList.length) continue;
      const [city, landmark] = key.split('_', 2);
      if (selectedSet.size && !selectedSet.has(key)) continue;
      const existing = (runCounts[city] && runCounts[city][landmark]) ? runCounts[city][landmark] : 0;
      if (existing >= minExisting) continue; // skip sufficiently covered
      const needed = Math.max(0, runsPerDest - existing);
      if (!needed) continue;
      // Schedule exactly `needed` runs per destination using seeds round-robin
      for (let i = 0; i < needed; i++) {
        const s = seedList[i % seedList.length];
        try {
          const res = await fetch(`${SERVICE_BASE}/api/route/start`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              seedPanoId: s.panoId,
              targetMeters,
              randomForward: true,
              regressWindow: 25,
              minFinalDegree: 3,
              polygonFile: (window.POLYGON_FILE || undefined),
              expectedPolygonKey: key,  // Add the expected landmark key
              expectedLandmarkName: landmark,  // Add landmark name for special handling
              noPersist: true,
              biasDirection: selectedBiasDirection,
              biasDecisionCount: biasDecisionCount
            })
          });
          if (res.ok) {
            totalScheduled++;
            const js = await res.json().catch(() => ({}));
            if (js && js.jobId) { liveJobIds.add(js.jobId); ensureRunColor(js.jobId); selectedRunIds.add(js.jobId); }
          }
        } catch(_) {}
      }
    }
    if (totalScheduled) {
      alert(`Started ${totalScheduled} run(s).`);
      // For live jobs, only poll status - don't try to load from database
      await pollLiveBulkJobs();
      suppressFitBounds = true;
      renderAll();
      renderBulkPanel();
      // Reset button state during bulk operations
      allPathsLoaded = false;
      updateLoadAllPathsButton();
    } else {
      alert('No runs started (check min existing threshold or seeds).');
    }
  }

  async function deleteSelectedRuns() {
    const ids = Array.from(selectedRunIds);
    if (!ids.length) { alert('No runs selected.'); return; }
    if (!confirm(`Delete ${ids.length} selected run(s)? This cannot be undone.`)) return;
    let ok = 0, fail = 0;
    for (const jid of ids) {
      try {
        const res = await fetch(`${SERVICE_BASE}/api/paths/delete`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ jobId: jid })
        });
        if (res.ok) {
          ok++;
          runs = runs.filter(r => r.jobId !== jid);
          delete runIdToPoints[jid];
          selectedRunIds.delete(jid);
          // Also drop from live tracking if present
          liveJobIds.delete(jid);
          delete jobIdToLive[jid];
        } else {
          fail++;
        }
      } catch (_) { fail++; }
    }
    renderAll();
    alert(`Deleted ${ok} run(s). ${fail ? fail + ' failed.' : ''}`);
  }

  // Manual refresh function for bulk operations (no automatic polling)
  window.manualRefreshBulkStatus = async function() {
    try {
      if (liveJobIds.size > 0) {
        // For live jobs, only poll status - don't try to load from database
        await pollLiveBulkJobs();
        suppressFitBounds = true;
        renderAll();
        renderBulkPanel();
        console.log('Bulk status refreshed manually');
      } else {
        console.log('No live jobs to refresh');
        // If no live jobs, load regular database runs
        await loadRuns();
        await loadAllPointsForAllRuns();
        renderAll();
        renderBulkPanel();
      }
    } catch (e) {
      console.error('Manual bulk refresh failed:', e);
    }
  };

  // Keyboard shortcut for bulk refresh (Ctrl+U or Cmd+U)
  document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
      e.preventDefault();
      window.manualRefreshBulkStatus();
    }
  });

  // Hook into Google Maps init lifecycle: call original initMap then our setup
  const _origInit = window.initMap;
  window.initMap = function() {
    if (typeof _origInit === 'function') {
      _origInit();
    }
    // Delay selector initialization to ensure DOM is ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => { initSelectorOnce().catch(console.error); });
    } else {
      initSelectorOnce().catch(console.error);
    }
  };
})();

