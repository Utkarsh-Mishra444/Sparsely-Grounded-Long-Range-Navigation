let map;
let panorama;
let streetViewService;
let currentPanoData = null;

let clickMarker = null;
let panoMarker = null;
let mark_pos = null;
// Markers for the current (not yet finalized) path:
let currentStartMarker = null;
let currentEndMarker = null;
let destinationCircle = null;

// Data for the current path and for the collection of paths
let currentPath = { start: null, end: null };
let paths = [];
let routes = [];

// Polygon source used by both frontend and backend
let POLYGON_FILE = null; // Will be auto-detected deterministically

// A per‑path counter. Each new path gets a number.
let pathCounter = 0;

// Local storage key for draft paths
const LOCAL_PATHS_KEY = 'unified_paths_local';

function saveLocalPaths() {
  try {
    localStorage.setItem(LOCAL_PATHS_KEY, JSON.stringify({ paths }));
  } catch (e) {
    console.error('Failed to save local paths', e);
  }
}

function loadLocalPaths() {
  try {
    const raw = localStorage.getItem(LOCAL_PATHS_KEY);
    if (!raw) return;
    const obj = JSON.parse(raw);
    if (obj && Array.isArray(obj.paths)) {
      paths = obj.paths;
    }
  } catch (e) {
    console.error('Failed to load local paths', e);
  }
}

// Deterministic polygon file detection - no fallbacks
async function autoDetectLatestPolygonFile() {
  const targetFile = 'data/polygons.json';

  try {
    const response = await fetch(`http://127.0.0.1:8765/api/paths/polygons?file=${encodeURIComponent(targetFile)}`);
    if (response.ok) {
      const data = await response.json();
      if (data.polygons && Object.keys(data.polygons).length > 0) {
        console.log(`✅ Found latest polygon file: ${targetFile}`);
        return targetFile;
      } else {
        console.error(`❌ ${targetFile} exists but contains no polygons`);
        return null;
      }
    } else {
      console.error(`❌ ${targetFile} not found or server error (${response.status})`);
      return null;
    }
  } catch (e) {
    console.error(`❌ Failed to load ${targetFile}:`, e.message);
    return null;
  }
}

function initMap() {
  // Initialize the map and Street View services.
  map = new google.maps.Map(document.getElementById('map'), {
    center: { lat: 40.7128, lng: -74.0060 },
    zoom: 12,
  });

  const coverageLayer = new google.maps.StreetViewCoverageLayer();
  coverageLayer.setMap(map);

  // Directions Service & Renderer with custom polyline options.
  const directionsService = new google.maps.DirectionsService();
  const directionsRenderer = new google.maps.DirectionsRenderer({
    suppressMarkers: true,
    polylineOptions: {
      strokeColor: '#FF0000',
      strokeWeight: 4,
      strokeOpacity: 0.8
    }
  });
  directionsRenderer.setMap(map);

  // "Calculate Distance" button event listener.
  document.getElementById('calcDistance').addEventListener('click', () => {
    if (!currentPath.start || !currentPath.end) {
      showNotification('Please set both start and end points first!');
      return;
    }
    const startPos = currentPath.start.position;
    const endPos = currentPath.end.position;
    const request = {
      origin: new google.maps.LatLng(startPos.lat, startPos.lng),
      destination: new google.maps.LatLng(endPos.lat, endPos.lng),
      travelMode: google.maps.TravelMode.DRIVING,
    };
    directionsService.route(request, (result, status) => {
      if (status === google.maps.DirectionsStatus.OK) {
        directionsRenderer.setDirections(result);
        const leg = result.routes[0].legs[0];
        const distanceInKm = (leg.distance.value / 1000).toFixed(2);
        // Store the distance and duration in the current path.
        currentPath.distance = distanceInKm;
        currentPath.duration = leg.duration.text;
        showNotification('Driving distance: ' + distanceInKm + ' km (' + leg.duration.text + ')');
        updateJsonPreview();
      } else {
        showNotification('Error calculating route: ' + status);
      }
    });
  });

  // Notification control.
  const notifDiv = document.getElementById('notification');
  map.controls[google.maps.ControlPosition.BOTTOM_LEFT].push(notifDiv);

  streetViewService = new google.maps.StreetViewService();
  panorama = new google.maps.StreetViewPanorama(document.getElementById('pano'));

  // Load any previously saved draft paths
  loadLocalPaths();
  if (typeof updateJsonPreview === 'function') {
    try { updateJsonPreview(); } catch(_) {}
  }

  // Auto-route state
  let selectedLandmark = null;
  let autoPolyline = null;
  let autoStopRequested = false;
  let autoActive = false;

  // Seed panos per landmark: key = `${city}_${landmarkName}`; value = array of seed entries
  // seed entry: { panoId, position:{lat,lng}, links:Int, addedAt, city, landmarkName }
  const seedPanosByLandmark = {};
  try { window.SEED_PANOS_BY_LANDMARK = seedPanosByLandmark; } catch(_) {}

  // Places Autocomplete for searching arbitrary destinations
  const placeInput = document.getElementById('placeSearch');
  if (placeInput && google.maps.places) {
    const autocomplete = new google.maps.places.Autocomplete(placeInput, {
      fields: ['geometry', 'name'],
    });
    autocomplete.bindTo('bounds', map);
    const searchMarker = new google.maps.Marker({ map });
    autocomplete.addListener('place_changed', () => {
      const place = autocomplete.getPlace();
      if (!place.geometry || !place.geometry.location) {
        showNotification('No details for input: ' + placeInput.value);
        return;
      }
      map.panTo(place.geometry.location);
      map.setZoom(16);
      searchMarker.setPosition(place.geometry.location);
      searchMarker.setTitle(place.name || 'Selected place');
    });
  }

  const setStartBtn = document.getElementById('setStart');
  const setEndBtn = document.getElementById('setEnd');
  const addSeedPanoBtn = document.getElementById('addSeedPano');

  // Initially disable the start and end buttons.
  setStartBtn.disabled = true;
  setEndBtn.disabled = true;
  if (addSeedPanoBtn) addSeedPanoBtn.disabled = true;

  // When the map is clicked, fetch the nearest panorama.
  map.addListener('click', (event) => {
    setStartBtn.disabled = true;
    setEndBtn.disabled = true;
    mark_pos = event.latLng;
    if (clickMarker) clickMarker.setMap(null);
    if (panoMarker) panoMarker.setMap(null);

    clickMarker = new google.maps.Marker({
      position: event.latLng,
      map: map,
      icon: {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 5,
        fillColor: '#FF0000',
        fillOpacity: 1,
        strokeWeight: 0,
      },
      title: 'Click Location',
    });

    streetViewService.getPanorama(
      {
        location: event.latLng,
        radius: 50,
        preference: google.maps.StreetViewPreference.NEAREST,
      },
      (data, status) => {
        if (status === 'OK' && data.links && data.links.length > 0) {
          currentPanoData = {
            panoId: data.location.pano,
            position: data.location.latLng,
            description: data.location.description || ''
          };
          panoMarker = new google.maps.Marker({
            position: currentPanoData.position,
            map: map,
            icon: {
              path: google.maps.SymbolPath.CIRCLE,
              scale: 5,
              fillColor: '#4285F4',
              fillOpacity: 1,
              strokeWeight: 0,
            },
            title: 'Panorama Location',
          });
          panorama.setPano(currentPanoData.panoId);
          updateCoordsDisplay();
          if (addSeedPanoBtn) addSeedPanoBtn.disabled = false;
          panorama.addListener('position_changed', () => {
            currentPanoData.position = panorama.getPosition();
            if (panoMarker) panoMarker.setPosition(currentPanoData.position);
            updateCoordsDisplay();
            if (addSeedPanoBtn) addSeedPanoBtn.disabled = !(currentPanoData && currentPanoData.panoId);
          });
          // Keep panoId in sync with viewer navigation
          panorama.addListener('pano_changed', () => {
            try {
              const pid = typeof panorama.getPano === 'function' ? panorama.getPano() : null;
              if (pid) {
                if (!currentPanoData) {
                  currentPanoData = { panoId: pid, position: panorama.getPosition() };
                } else {
                  currentPanoData.panoId = pid;
                  // Update position too, if available
                  const p = typeof panorama.getPosition === 'function' ? panorama.getPosition() : null;
                  if (p) currentPanoData.position = p;
                }
                if (addSeedPanoBtn) addSeedPanoBtn.disabled = false;
              }
            } catch (_) {}
          });
          setStartBtn.disabled = false;
          setEndBtn.disabled = false;
          console.log('Number of links:', (data && data.links) ? data.links.length : 0);
        } else {
          if (clickMarker) { clickMarker.setMap(null); clickMarker = null; }
          if (panoMarker) { panoMarker.setMap(null); panoMarker = null; }
          showNotification('No navigable Street View found at this location!');
        }
      }
    );
  });

  // Helper: point-in-polygon (ray casting), polygon as [[lat,lng], ...]
  function isPointInPolygon(lat, lng, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const yi = polygon[i][0], xi = polygon[i][1];
      const yj = polygon[j][0], xj = polygon[j][1];
      const intersect = ((xi > lng) !== (xj > lng)) && (lat < (yj - yi) * (lng - xi) / (xj - xi + 1e-12) + yi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  // Find landmark for a point using current polygons (edited first, then base file)
  function findLandmarkForPoint(lat, lng) {
    // Prefer file polygons (LatestPoly.json) - these are the committed/authoritative version
    const fileData = (typeof window !== 'undefined' && window.LM_POLYGON_DATA) ? window.LM_POLYGON_DATA : (lmPolygonData || {});
    for (const [city, landmarks] of Object.entries(fileData)) {
      for (const [landmarkName, obj] of Object.entries(landmarks || {})) {
        const poly = obj && obj.polygon;
        if (Array.isArray(poly) && poly.length >= 3 && isPointInPolygon(lat, lng, poly)) {
          return { city, landmarkName, key: `${city}_${landmarkName}` };
        }
      }
    }
    // Fall back to localStorage edited polygons only if not found in file
    const edited = (typeof window !== 'undefined' && window.LM_EDITED_POLYGONS) ? window.LM_EDITED_POLYGONS : (lmEditedPolygons || {});
    for (const [key, poly] of Object.entries(edited)) {
      const [city, landmarkName] = key.split('_', 2);
      if (Array.isArray(poly) && poly.length >= 3 && isPointInPolygon(lat, lng, poly)) {
        return { city, landmarkName, key };
      }
    }
    return null;
  }

  // Enable seed button when pano is available
  function updateSeedButtonState() {
    if (!addSeedPanoBtn) return;
    addSeedPanoBtn.disabled = !(currentPanoData && currentPanoData.panoId);
  }
  updateSeedButtonState();

  const toggleCoverageBtn = document.getElementById('toggleCoverage');
  toggleCoverageBtn.addEventListener('click', () => {
    if (coverageLayer.getMap()) {
      coverageLayer.setMap(null);
      showNotification('Street View Coverage Layer hidden');
    } else {
      coverageLayer.setMap(map);
      showNotification('Street View Coverage Layer shown');
    }
  });

  // Set Start: Save the current panorama as the start for the current path.
  setStartBtn.addEventListener('click', () => {
    if (currentPanoData) {
      currentPath.start = {
        panoId: currentPanoData.panoId,
        position: mark_pos ? mark_pos.toJSON() : currentPanoData.position.toJSON(),
        panoPosition: currentPanoData.position.toJSON(),
        description: currentPanoData.description
      };
      if (currentStartMarker) currentStartMarker.setMap(null);
      currentStartMarker = new google.maps.Marker({
        position: currentPanoData.position,
        map: map,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 8,
          fillColor: '#00FF00',
          fillOpacity: 1,
          strokeWeight: 0,
        },
        title: 'Start Point',
      });
      showNotification('Start point set at panorama location!');
    }
  });

  // Set End: Save the current panorama as the end for the current path.
  setEndBtn.addEventListener('click', () => {
    if (currentPanoData) {
      currentPath.end = {
        panoId: currentPanoData.panoId,
        position: mark_pos ? mark_pos.toJSON() : currentPanoData.position.toJSON(),
        panoPosition: currentPanoData.position.toJSON(),
        description: currentPanoData.description
      };
      // Read desired radius (meters) from input
      const radiusInput = document.getElementById('destinationRadius');
      const radiusMeters = radiusInput ? Math.max(0, Number(radiusInput.value) || 0) : 0;
      if (radiusMeters > 0) {
        currentPath.end.destinationRadius = radiusMeters;
      } else {
        delete currentPath.end.destinationRadius;
      }
      if (currentEndMarker) currentEndMarker.setMap(null);
      currentEndMarker = new google.maps.Marker({
        position: currentPanoData.position,
        map: map,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 8,
          fillColor: '#FF00FF',
          fillOpacity: 1,
          strokeWeight: 0,
        },
        title: 'End Point',
      });
      // Draw or update destination circle around the chosen end position
      const centerLatLng = new google.maps.LatLng(
        currentPath.end.position.lat,
        currentPath.end.position.lng
      );
      if (!destinationCircle) {
        destinationCircle = new google.maps.Circle({
          map,
          center: centerLatLng,
          radius: radiusMeters,
          strokeColor: '#00BFA5',
          strokeOpacity: 0.9,
          strokeWeight: 2,
          fillColor: '#00BFA5',
          fillOpacity: 0.15,
        });
      } else {
        destinationCircle.setCenter(centerLatLng);
        destinationCircle.setRadius(radiusMeters);
        destinationCircle.setMap(map);
      }
      showNotification('End point set at panorama location!');
    }
  });

  // Add as Seed Pano button logic
  if (addSeedPanoBtn) {
    addSeedPanoBtn.addEventListener('click', async () => {
      try {
        if (!currentPanoData || !currentPanoData.panoId) {
          showNotification('No panorama selected. Click on the map to pick one.');
          return;
        }
        const pos = currentPanoData.position && currentPanoData.position.toJSON ? currentPanoData.position.toJSON() : null;
        if (!pos) { showNotification('Panorama has no position.'); return; }

        // Identify landmark (must be inside a polygon)
        const lm = findLandmarkForPoint(pos.lat, pos.lng);
        if (!lm) { showNotification('Selected pano is not inside any landmark polygon.'); return; }

        // Fetch links count for confirmation
        const panoMeta = await getPanoramaAsync({ pano: currentPanoData.panoId });
        const linksCount = (panoMeta && Array.isArray(panoMeta.links)) ? panoMeta.links.length : 0;

        const details = `Add this as seed pano?\n\n` +
          `Landmark: ${lm.city} / ${lm.landmarkName}\n` +
          `Pano ID: ${currentPanoData.panoId}\n` +
          `Coords: ${pos.lat.toFixed(6)}, ${pos.lng.toFixed(6)}\n` +
          `Links: ${linksCount}`;
        const ok = window.confirm(details);
        if (!ok) return;

        // Save in memory
        const entry = {
          panoId: currentPanoData.panoId,
          position: { lat: pos.lat, lng: pos.lng },
          links: linksCount,
          addedAt: new Date().toISOString(),
          city: lm.city,
          landmarkName: lm.landmarkName
        };
        if (!seedPanosByLandmark[lm.key]) seedPanosByLandmark[lm.key] = [];
        // Avoid duplicates by panoId
        if (!seedPanosByLandmark[lm.key].some(s => s.panoId === entry.panoId)) {
          seedPanosByLandmark[lm.key].push(entry);
          showNotification(`Seed pano added for ${lm.landmarkName}. Total: ${seedPanosByLandmark[lm.key].length}`);
          try { localStorage.setItem('landmarkSeedPanos', JSON.stringify(seedPanosByLandmark)); } catch(_) {}
          try { window.dispatchEvent(new Event('seedPanosChanged')); } catch(_) {}
        } else {
          showNotification('This seed pano is already added for that landmark.');
        }
      } catch (e) {
        showNotification('Failed to add seed pano: ' + e.message);
      }
    });
  }

  // "Add Path" button: Finalize the current start–end pair.
  document.getElementById('addPath').addEventListener('click', () => {
    if (!currentPath.start || !currentPath.end) {
      showNotification('Please set both start and end points for the current path!');
      return;
    }
    const destInput = document.getElementById('destinationName');
    const destinationName = destInput ? destInput.value.trim() : '';
    pathCounter++;
    if (currentStartMarker)
      currentStartMarker.setLabel({ text: pathCounter.toString(), color: 'white' });
    if (currentEndMarker)
      currentEndMarker.setLabel({ text: pathCounter.toString(), color: 'white' });
    
    // Push a copy of the current path (including any distance/duration data).
    const toPush = { ...currentPath };
    if (destinationName) {
      // store destination on end point, keeping parity with existing path formats
      toPush.end = { ...toPush.end, destination: destinationName };
    }
    paths.push(toPush);
    updateJsonPreview();
    saveLocalPaths();
    
    // Reset for the next path.
    currentPath = { start: null, end: null };
    currentStartMarker = null;
    currentEndMarker = null;
    showNotification('Path added to collection!');
  });

  // "Done" button: Export the JSON with extra data if selected.
  document.getElementById('done').addEventListener('click', () => {
    if (paths.length === 0) {
      showNotification('No paths have been added!');
      return;
    }
    
    const filteredPaths = paths.map(path => {
      const filteredPath = {
        start: filterPanoData(path.start),
        end: filterPanoData(path.end)
      };
      if (document.getElementById('includeDistance').checked && path.distance) {
        filteredPath.distance = path.distance;
        filteredPath.duration = path.duration;
      }
      if (document.getElementById('includeDescription').checked) {
        filteredPath.start.description = path.start.description;
        filteredPath.end.description = path.end.description;
      }
      if (document.getElementById('includeDestination').checked && path.end && path.end.destination) {
        filteredPath.end.destination = path.end.destination;
      }
      if (path.end && typeof path.end.destinationRadius === 'number') {
        filteredPath.end.destinationRadius = path.end.destinationRadius;
      }
      return filteredPath;
    });
    
    const output = { paths: filteredPaths };
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(output, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute('href', dataStr);
    downloadAnchorNode.setAttribute('download', 'paths.json');
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  });

  // Commit locally stored paths to backend DB
  const commitBtn = document.getElementById('commitLocal');
  if (commitBtn) {
    commitBtn.addEventListener('click', async () => {
      if (!paths.length) { showNotification('No local paths to commit.'); return; }
      // Build export similar to Done, honoring current checkboxes
      const filteredPaths = paths.map(path => {
        const filteredPath = {
          start: filterPanoData(path.start),
          end: filterPanoData(path.end)
        };
        if (document.getElementById('includeDistance').checked && path.distance) {
          filteredPath.distance = path.distance;
          filteredPath.duration = path.duration;
        }
        if (document.getElementById('includeDescription').checked) {
          filteredPath.start.description = path.start.description;
          filteredPath.end.description = path.end.description;
        }
        if (document.getElementById('includeDestination').checked && path.end && path.end.destination) {
          filteredPath.end.destination = path.end.destination;
        }
        if (path.end && typeof path.end.destinationRadius === 'number') {
          filteredPath.end.destinationRadius = path.end.destinationRadius;
        }
        return filteredPath;
      });
      try {
        const res = await fetch('http://127.0.0.1:8765/api/paths/manual_commit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ paths: filteredPaths })
        });
        const json = await res.json();
        if (!res.ok) {
          showNotification('Commit failed: ' + (json && (json.error || json.message) || res.status));
          return;
        }
        showNotification('Committed ' + (json.count || filteredPaths.length) + ' paths. Batch ' + (json.batchId || 'n/a'));
      } catch (e) {
        showNotification('Commit error: ' + e.message);
      }
    });
  }

  // Keyboard shortcuts for setting start (S), end (E), adding path (A),
  // toggling coverage (T), and calculating distance (D).
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    switch(e.key.toLowerCase()) {
      case 's':
        document.getElementById('setStart').click();
        break;
      case 'e':
        document.getElementById('setEnd').click();
        break;
      case 'a':
        document.getElementById('addPath').click();
        break;
      case 't':
        if (coverageLayer.getMap()) {
          coverageLayer.setMap(null);
          showNotification('Street View Coverage Layer hidden');
        } else {
          coverageLayer.setMap(map);
          showNotification('Street View Coverage Layer shown');
        }
        break;
      case 'd':
        document.getElementById('calcDistance').click();
        break;
      default:
        break;
    }
  });

  // Helper function: Update the JSON preview to show only what is being exported.
  function updateJsonPreview() {
    const filteredPaths = paths.map(path => {
      const filteredPath = {
        start: filterPanoData(path.start),
        end: filterPanoData(path.end)
      };
      if (document.getElementById('includeDistance').checked && path.distance) {
        filteredPath.distance = path.distance;
        filteredPath.duration = path.duration;
      }
      if (document.getElementById('includeDescription').checked) {
        filteredPath.start.description = path.start.description;
        filteredPath.end.description = path.end.description;
      }
      if (document.getElementById('includeDestination').checked && path.end && path.end.destination) {
        filteredPath.end.destination = path.end.destination;
      }
      if (path.end && typeof path.end.destinationRadius === 'number') {
        filteredPath.end.destinationRadius = path.end.destinationRadius;
      }
      return filteredPath;
    });
    const output = { paths: filteredPaths };
    document.getElementById('output').textContent = JSON.stringify(output, null, 2);
  }

  // Load persisted seed panos at init
  try {
    const storedSeeds = localStorage.getItem('landmarkSeedPanos');
    if (storedSeeds) {
      const obj = JSON.parse(storedSeeds);
      if (obj && typeof obj === 'object') {
        Object.assign(seedPanosByLandmark, obj);
        try { window.SEED_PANOS_BY_LANDMARK = seedPanosByLandmark; } catch(_) {}
      }
    }
  } catch(_) {}
  
  // Helper function: Filter a panorama object based on checkbox settings.
  function filterPanoData(pano) {
    const result = {};
    if (document.getElementById('includePanoId').checked) {
      result.panoId = pano.panoId;
    }
    if (document.getElementById('includePosition').checked) {
      result.position = pano.position;
      if (pano.panoPosition) result.panoPosition = pano.panoPosition;
    }
    return result;
  }
  
  // Helper function: Update the coordinate display.
  function updateCoordsDisplay() {
    const pos = mark_pos //panorama.getPosition();
    if (pos) {
      const lat = pos.lat();
      const lng = pos.lng();
      document.getElementById('coords-json').textContent = `{ lat: ${lat.toFixed(6)}, lng: ${lng.toFixed(6)} }`;
      document.getElementById('coords-string').textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
    }
  }

  // Landmarks sidebar (from index_inline) – initialize only if the elements and data exist
  (function setupLandmarks() {
    const citySel = document.getElementById('city');
    const fitBtn = document.getElementById('fit');
    const filterInput = document.getElementById('filter');
    const listDiv = document.getElementById('list');
    const errorDiv = document.getElementById('error');
    const resetBtn = document.getElementById('resetPolygons');
    if (!citySel || !window.ALL_DATA) return;

    let lmMarkers = [];
    let lmMarkerByKey = {}; // key -> marker
    let lmSeedMarkers = []; // temporary markers for seed panos of selected landmark
    let lmPolygons = [];
    let lmCurrentCity = null;
    let lmEditMode = false;
    let lmPolygonData = {}; // Store polygon data from current polygon file: "city_landmark_name" -> polygon_coords
    let lmEditedPolygons = {}; // Store manually edited polygons: "city_landmark_name" -> polygon_coords
    // Expose for other logic (seed panos)
    try { window.LM_POLYGON_DATA = lmPolygonData; } catch(_) {}
    try { window.LM_EDITED_POLYGONS = lmEditedPolygons; } catch(_) {}
    const lmInfo = new google.maps.InfoWindow();

    // Load polygon data from selected file
    function lmLoadPolygonData() {
      if (!POLYGON_FILE) {
        console.log('No polygon file detected - landmark polygons will use defaults');
        return Promise.resolve({});
      }
      const targetFile = POLYGON_FILE;
      return fetch(`http://127.0.0.1:8765/api/paths/polygons?file=${encodeURIComponent(targetFile)}&full=true`)
        .then(response => response.json())
        .then(apiResponse => {
          const data = apiResponse.data;
          lmPolygonData = data.editedPolygons || {};
          try { window.LM_POLYGON_DATA = lmPolygonData; } catch(_) {}

          // Extract seedPanos from polygon data and merge into seedPanosByLandmark
          try {
            for (const [city, landmarks] of Object.entries(lmPolygonData)) {
              for (const [landmarkName, landmarkData] of Object.entries(landmarks)) {
                if (Array.isArray(landmarkData.seedPanos) && landmarkData.seedPanos.length) {
                  const key = `${city}_${landmarkName}`;
                  if (!seedPanosByLandmark[key]) seedPanosByLandmark[key] = [];
                  const existingIds = new Set(seedPanosByLandmark[key].map(s => s.panoId));
                  landmarkData.seedPanos.forEach(s => {
                    if (!existingIds.has(s.panoId)) {
                      seedPanosByLandmark[key].push({
                        ...s,
                        city: city,
                        landmarkName: landmarkName
                      });
                    }
                  });
                }
              }
            }
            try { window.SEED_PANOS_BY_LANDMARK = seedPanosByLandmark; } catch(_) {}
          } catch(e) { console.warn('Failed to extract seedPanos:', e); }
        })
        .catch(error => {
          console.warn('Could not load polygon file:', targetFile, error);
          lmPolygonData = {};
          try { window.LM_POLYGON_DATA = lmPolygonData; } catch(_) {}
          throw error; // Re-throw to trigger catch in caller
        });
    }

    // Load edited polygons from localStorage
    function lmLoadEditedPolygons() {
      const stored = localStorage.getItem('landmarkEditedPolygons');
      if (stored) {
        lmEditedPolygons = JSON.parse(stored);
        try { window.LM_EDITED_POLYGONS = lmEditedPolygons; } catch(_) {}
      }
    }

    // Migrate renamed landmarks so edited polygons and seed panos follow the new name
    function lmMigrateRenamedLandmarks() {
      const RENAME_KEYS = {
        'New York City_Washington Square Arch': 'New York City_Washington Square Park'
      };
      let changed = false;
      // Migrate edited polygons
      Object.entries(RENAME_KEYS).forEach(([oldKey, newKey]) => {
        if (lmEditedPolygons && lmEditedPolygons[oldKey]) {
          if (!lmEditedPolygons[newKey]) {
            lmEditedPolygons[newKey] = lmEditedPolygons[oldKey];
          }
          delete lmEditedPolygons[oldKey];
          changed = true;
        }
      });
      if (changed) {
        lmSaveEditedPolygons();
        try { window.LM_EDITED_POLYGONS = lmEditedPolygons; } catch(_) {}
      }
      // Migrate seed panos stored in localStorage and memory
      try {
        const seedsRaw = localStorage.getItem('landmarkSeedPanos');
        const seedsObj = seedsRaw ? JSON.parse(seedsRaw) : {};
        let seedsChanged = false;
        Object.entries(RENAME_KEYS).forEach(([oldKey, newKey]) => {
          if (seedsObj && seedsObj[oldKey]) {
            const oldList = seedsObj[oldKey];
            const newList = Array.isArray(seedsObj[newKey]) ? seedsObj[newKey] : [];
            // Merge without duplicate panoIds
            const have = new Set(newList.map(s => s.panoId));
            oldList.forEach(s => { if (!have.has(s.panoId)) newList.push(s); });
            seedsObj[newKey] = newList;
            delete seedsObj[oldKey];
            seedsChanged = true;
          }
        });
        if (seedsChanged) {
          localStorage.setItem('landmarkSeedPanos', JSON.stringify(seedsObj));
          // Update in-memory map as well
          if (typeof window !== 'undefined' && window.SEED_PANOS_BY_LANDMARK) {
            // Clear then copy
            Object.keys(window.SEED_PANOS_BY_LANDMARK).forEach(k => { delete window.SEED_PANOS_BY_LANDMARK[k]; });
            Object.entries(seedsObj).forEach(([k, v]) => { window.SEED_PANOS_BY_LANDMARK[k] = v; });
          }
        }
      } catch(_) {}
    }

    // Save edited polygons to localStorage
    function lmSaveEditedPolygons() {
      localStorage.setItem('landmarkEditedPolygons', JSON.stringify(lmEditedPolygons));
    }

    function lmClearMarkers() {
      lmMarkers.forEach(m => m.setMap(null));
      lmPolygons.forEach(p => p.setMap(null));
      lmMarkers = [];
      lmPolygons = [];
    }
    function lmFitToMarkers() {
      if (!lmMarkers.length) return;
      const b = new google.maps.LatLngBounds();
      lmMarkers.forEach(m => b.extend(m.getPosition()));
      map.fitBounds(b, 60);
    }
    function lmRenderList() {
      const filter = (filterInput && filterInput.value ? filterInput.value : '').toLowerCase();
      if (listDiv) listDiv.innerHTML = '';
      const allPlaces = window.ALL_DATA[lmCurrentCity] || [];
      const filteredPlaces = allPlaces.filter(p => p[0].toLowerCase().includes(filter));

      filteredPlaces.forEach((p) => {
        const originalIdx = allPlaces.indexOf(p);
        const div = document.createElement('div');
        div.className = 'place';

        // Check if this landmark has been manually edited
        const landmarkName = p[0];
        const editKey = `${lmCurrentCity}_${landmarkName}`;
        const isEdited = editKey in lmEditedPolygons;
        const seedCount = (seedPanosByLandmark[editKey] && seedPanosByLandmark[editKey].length) ? seedPanosByLandmark[editKey].length : 0;

        const editedBadge = isEdited ? ' <span style="color: #4CAF50; font-weight: bold;">[EDITED]</span>' : '';
        const seedsBadge = seedCount > 0 ? ` <span style="color:#F9A825; font-weight:bold;">[SEEDS:${seedCount}]</span>` : '';
        div.innerHTML = `${p[0]}${editedBadge}${seedsBadge}`;
        div.title = p[1].toFixed(6) + ', ' + p[2].toFixed(6);
        div.onclick = () => google.maps.event.trigger(lmMarkers[originalIdx], 'click');
        if (listDiv) listDiv.appendChild(div);
      });
    }
    function lmShowCity(city) {
      lmCurrentCity = city;
      lmClearMarkers();
      lmLoadEditedPolygons(); // Load any manually edited polygons
      lmMigrateRenamedLandmarks(); // Ensure keys match current landmark names

      // Load polygon data asynchronously
      lmLoadPolygonData().then(() => {
        lmRenderCityPolygons(city);
      }).catch(() => {
        // If custom file fails to load, render with default polygons
        lmRenderCityPolygons(city);
      });
    }

    function lmRenderCityPolygons(city) {
      // Use polygon file data first (has correct names), fall back to landmarks_data.js
      let places = [];
      const fileData = (typeof window !== 'undefined' && window.LM_POLYGON_DATA) ? window.LM_POLYGON_DATA : (lmPolygonData || {});

      if (fileData[city]) {
        // Use data from LatestPoly.json - convert to same format as ALL_DATA
        places = Object.entries(fileData[city]).map(([landmarkName, data]) => {
          return [landmarkName, data.coordinates[0], data.coordinates[1], data.polygon];
        });
      } else {
        // Fall back to landmarks_data.js
        places = window.ALL_DATA[city] || [];
      }

      const bounds = new google.maps.LatLngBounds();
      places.forEach((p, idx) => {
        const lat = p[1], lng = p[2];
        const pos = new google.maps.LatLng(lat, lng);
        bounds.extend(pos);
        const key = `${city}_${p[0]}`;
        const seedCount = (seedPanosByLandmark[key] && seedPanosByLandmark[key].length) ? seedPanosByLandmark[key].length : 0;
        const markerOptions = { position: pos, map, title: p[0] };
        if (seedCount > 0) {
          markerOptions.label = { text: '★', color: '#F9A825', fontWeight: 'bold' };
        }
        const marker = new google.maps.Marker(markerOptions);
        marker.addListener('click', () => lmOnMarkerClick(marker, p[0], lat, lng));
        lmMarkers.push(marker);
        lmMarkerByKey[key] = marker;

        // Create polygon if coordinates are available
        if (p[3] && Array.isArray(p[3])) {
          const landmarkName = p[0];
          const editKey = `${city}_${landmarkName}`;
          const isEdited = editKey in lmEditedPolygons;

          // Check if we have custom polygon data from current polygon file
          const cityPolygons = lmPolygonData[city] || {};
          const customPolygon = cityPolygons[landmarkName];

          let polygonCoords;
          if (customPolygon && customPolygon.polygon) {
            // Prefer polygon from current polygon file (LatestPoly)
            polygonCoords = customPolygon.polygon.map(coord => new google.maps.LatLng(coord[0], coord[1]));
          } else if (isEdited) {
            // Fallback to manually edited polygon from localStorage
            polygonCoords = lmEditedPolygons[editKey].map(coord => new google.maps.LatLng(coord[0], coord[1]));
          } else {
            // Fallback to default polygon from landmarks_data.js
            polygonCoords = p[3].map(coord => new google.maps.LatLng(coord[0], coord[1]));
          }

          const polygon = new google.maps.Polygon({
            paths: polygonCoords,
            strokeColor: isEdited ? '#00FF00' : (customPolygon ? '#0000FF' : '#FF0000'), // Green for edited, blue for custom, red for auto
            strokeOpacity: 0.8,
            strokeWeight: 2,
            fillColor: isEdited ? '#00FF00' : (customPolygon ? '#0000FF' : '#FF0000'),
            fillOpacity: 0.1,
            editable: lmEditMode,
            clickable: lmEditMode, // Only clickable when editing to allow vertex dragging
            map: map
          });

          // Add listener for polygon edits
          google.maps.event.addListener(polygon, 'mouseup', () => {
            if (lmEditMode) {
              const newCoords = polygon.getPath().getArray().map(latLng => [latLng.lat(), latLng.lng()]);
              lmEditedPolygons[editKey] = newCoords;
              lmSaveEditedPolygons();
              // Update polygon color to show it's been edited
              polygon.setOptions({
                strokeColor: '#00FF00',
                fillColor: '#00FF00'
              });
            }
          });

          lmPolygons.push(polygon);
        }
      });
      if (!bounds.isEmpty()) map.fitBounds(bounds, 60);
      lmRenderList();
    }
    function lmOnMarkerClick(marker, name, lat, lng) {
      const ll = lat.toFixed(6) + ',' + lng.toFixed(6);
      const mapsUrl = `https://maps.google.com/?q=${lat},${lng}`;
      lmInfo.setContent(
        `<div>
           <strong>${name}</strong><br/>
           <span class="muted">${ll}</span><br/>
           <a href="${mapsUrl}" target="_blank" rel="noreferrer">Open in Google Maps</a>
           <div id="sv-status" class="muted">Checking Street View...</div>
           <div style="margin-top:8px;">
             <div style="font-weight:600; margin-bottom:4px;">Seed Panos:</div>
             <div id="seed-list"><em>No seed panos saved.</em></div>
           </div>
         </div>`
      );
      lmInfo.open({ anchor: marker, map });
      // Track selected landmark for Auto Route
      selectedLandmark = { name, position: new google.maps.LatLng(lat, lng) };
      const key = `${lmCurrentCity}_${name}`;
      try { lmRenderSeedList(key); } catch(_) {}
      showNotification('Selected landmark: ' + name);
      if (!streetViewService) return;
      streetViewService.getPanorama({ location: marker.getPosition(), radius: 50 }, (data, status) => {
        const el = document.getElementById('sv-status');
        if (!el) return;
        el.textContent = (status === 'OK') ? 'Street View: available ✅' : 'Street View: not found within 50m ❌';
      });
    }

    function lmClearSeedMarkers() {
      lmSeedMarkers.forEach(m => { try { m.setMap(null); } catch(_) {} });
      lmSeedMarkers = [];
    }

    function lmRenderSeedList(key) {
      const container = document.getElementById('seed-list');
      if (!container) return;
      const seeds = seedPanosByLandmark[key] || [];
      lmClearSeedMarkers();
      if (!seeds.length) { container.innerHTML = '<em>No seed panos saved.</em>'; return; }
      const ul = document.createElement('ul');
      ul.style.margin = '0';
      ul.style.paddingLeft = '16px';
      seeds.forEach((s, idx) => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = '#';
        a.textContent = `${s.panoId}  (${s.position.lat.toFixed(5)}, ${s.position.lng.toFixed(5)})  [links:${s.links}]`;
        a.addEventListener('click', (e) => {
          e.preventDefault();
          try { panorama.setPano(s.panoId); } catch(_) {}
          try { map.panTo(new google.maps.LatLng(s.position.lat, s.position.lng)); } catch(_) {}
        });
        li.appendChild(a);
        const delBtn = document.createElement('button');
        delBtn.textContent = 'Delete';
        delBtn.title = 'Remove this seed pano';
        delBtn.style.marginLeft = '6px';
        delBtn.style.padding = '2px 6px';
        delBtn.style.fontSize = '11px';
        delBtn.addEventListener('click', () => {
          const msg = `Remove seed ${s.panoId} for ${key}?\nThis only deletes this one seed.`;
          if (!confirm(msg)) return;
          const list = Array.isArray(seedPanosByLandmark[key]) ? seedPanosByLandmark[key] : [];
          seedPanosByLandmark[key] = list.filter(t => t && t.panoId !== s.panoId);
          try { localStorage.setItem('landmarkSeedPanos', JSON.stringify(seedPanosByLandmark)); } catch(_) {}
          try {
            if (typeof window !== 'undefined' && window.SEED_PANOS_BY_LANDMARK) {
              window.SEED_PANOS_BY_LANDMARK[key] = seedPanosByLandmark[key];
              window.dispatchEvent(new Event('seedPanosChanged'));
            }
          } catch(_) {}
          lmRenderSeedList(key);
        });
        li.appendChild(delBtn);
        ul.appendChild(li);
        // Add temporary marker on map
        try {
          const sm = new google.maps.Marker({
            map,
            position: new google.maps.LatLng(s.position.lat, s.position.lng),
            icon: { path: google.maps.SymbolPath.CIRCLE, scale: 5, fillColor: '#F9A825', fillOpacity: 1, strokeColor: '#333', strokeWeight: 1 },
            title: `Seed ${idx+1}: ${s.panoId}`
          });
          sm.addListener('click', () => { try { panorama.setPano(s.panoId); } catch(_) {} });
          lmSeedMarkers.push(sm);
        } catch(_) {}
      });
      container.innerHTML = '';
      container.appendChild(ul);
    }

    // Populate cities
    Object.keys(window.ALL_DATA).forEach(city => {
      const opt = document.createElement('option');
      opt.value = city; opt.textContent = city;
      citySel.appendChild(opt);
    });

    citySel.addEventListener('change', () => lmShowCity(citySel.value));
    if (fitBtn) fitBtn.addEventListener('click', lmFitToMarkers);

    // Polygon editing toggle
    const togglePolygonEditBtn = document.getElementById('togglePolygonEdit');
    if (togglePolygonEditBtn) {
      togglePolygonEditBtn.addEventListener('click', () => {
        lmEditMode = !lmEditMode;
        togglePolygonEditBtn.textContent = lmEditMode ? 'Exit Edit Mode' : 'Edit Polygons';
        togglePolygonEditBtn.classList.toggle('active', lmEditMode);

        // Re-show the current city to apply edit mode changes
        if (lmCurrentCity) {
          lmShowCity(lmCurrentCity);
        }

        showNotification(lmEditMode ?
          'Polygon editing enabled. Drag polygon vertices to edit boundaries.' :
          'Polygon editing disabled.');
      });
    }

    // Export polygons functionality
    const exportPolygonsBtn = document.getElementById('exportPolygons');
    if (exportPolygonsBtn) {
      exportPolygonsBtn.addEventListener('click', () => {
        lmLoadEditedPolygons(); // Ensure we have latest data

        if (Object.keys(lmEditedPolygons).length === 0) {
          showNotification('No edited polygons to export.');
          return;
        }

        // Build export data grouped by city with landmark names as keys
        const exportPolygons = {};
        for (const [key, polygon] of Object.entries(lmEditedPolygons)) {
          const [city, landmarkName] = key.split('_', 2); // Split only on first underscore to handle landmark names with underscores

          // Find the landmark in ALL_DATA to get coordinates; skip unknown (clean export)
          const landmarks = window.ALL_DATA[city] || [];
          const landmarkData = landmarks.find(l => l[0] === landmarkName);
          if (!landmarkData) {
            // Landmark no longer exists in canonical list → skip from export
            continue;
          }

          // Initialize city group if it doesn't exist
          if (!exportPolygons[city]) {
            exportPolygons[city] = {};
          }

          // Add landmark data with name as key, include seed panos if any
          const lmKey = `${city}_${landmarkName}`;
          const seeds = seedPanosByLandmark[lmKey] || [];
          exportPolygons[city][landmarkName] = {
            index: landmarkData ? landmarks.indexOf(landmarkData) : -1,
            coordinates: landmarkData ? [landmarkData[1], landmarkData[2]] : [0, 0],
            polygon: polygon,
            seedPanos: seeds.map(s => ({
              panoId: s.panoId,
              position: s.position,
              links: s.links,
              addedAt: s.addedAt
            }))
          };
        }

        const exportData = {
          exportedAt: new Date().toISOString(),
          description: "Manually edited landmark polygon boundaries - grouped by city and landmark name",
          editedPolygons: exportPolygons,
          totalEdits: Object.keys(lmEditedPolygons).length
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `edited-landmark-polygons-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showNotification(`Exported ${Object.keys(lmEditedPolygons).length} edited polygons with landmark details.`);
      });
    }

    // Import polygons functionality
    const importPolygonsBtn = document.getElementById('importPolygons');
    if (importPolygonsBtn) {
      importPolygonsBtn.addEventListener('click', () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = (e) => {
          const file = e.target.files[0];
          if (!file) return;

          const reader = new FileReader();
          reader.onload = (event) => {
            try {
              const importData = JSON.parse(event.target.result);

              if (!importData.editedPolygons) {
                throw new Error('Invalid file format - missing editedPolygons');
              }

              // Extract polygon coordinates from the grouped format
              const importedPolygons = {};
              for (const [city, cityLandmarks] of Object.entries(importData.editedPolygons)) {
                for (const [landmarkName, landmarkData] of Object.entries(cityLandmarks)) {
                  if (landmarkData.polygon) {
                    const key = `${city}_${landmarkName}`;
                    importedPolygons[key] = landmarkData.polygon;
                  }
                }
              }

              // Merge imported data with existing data
              lmEditedPolygons = { ...lmEditedPolygons, ...importedPolygons };
              lmSaveEditedPolygons();
              try { window.LM_EDITED_POLYGONS = lmEditedPolygons; } catch(_) {}

              // Merge seedPanos from import if present (same grouped format)
              try {
                const seedsMerged = JSON.parse(localStorage.getItem('landmarkSeedPanos') || '{}');
                for (const [city, cityLandmarks] of Object.entries(importData.editedPolygons)) {
                  for (const [landmarkName, landmarkData] of Object.entries(cityLandmarks)) {
                    const key = `${city}_${landmarkName}`;
                    if (Array.isArray(landmarkData.seedPanos) && landmarkData.seedPanos.length) {
                      seedsMerged[key] = Array.isArray(seedsMerged[key]) ? seedsMerged[key] : [];
                      // Merge without duplicating panoIds
                      const existingIds = new Set(seedsMerged[key].map(s => s.panoId));
                      landmarkData.seedPanos.forEach(s => {
                        if (!existingIds.has(s.panoId)) seedsMerged[key].push(s);
                      });
                    }
                  }
                }
                localStorage.setItem('landmarkSeedPanos', JSON.stringify(seedsMerged));
                Object.assign(seedPanosByLandmark, seedsMerged);
                try { window.SEED_PANOS_BY_LANDMARK = seedPanosByLandmark; } catch(_) {}
              } catch(_) {}

              // Refresh display
              if (lmCurrentCity) {
                lmShowCity(lmCurrentCity);
              }

              const importedCount = Object.keys(importedPolygons).length;
              showNotification(`Imported ${importedCount} edited polygons successfully.`);

            } catch (error) {
              showNotification('Error importing file: ' + error.message);
            }
          };
          reader.readAsText(file);
        };
        input.click();
      });
    }

    if (filterInput) filterInput.addEventListener('input', lmRenderList);

    // Reset edited polygons: clear localStorage and refresh
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        try {
          localStorage.removeItem('landmarkEditedPolygons');
          lmEditedPolygons = {};
          try { window.LM_EDITED_POLYGONS = lmEditedPolygons; } catch(_) {}
          if (lmCurrentCity) lmShowCity(lmCurrentCity);
          showNotification('Cleared edited polygons.');
        } catch (e) {
          showNotification('Failed to reset: ' + e.message);
        }
      });
    }

    if (citySel.options.length) lmShowCity(citySel.options[0].value);
  })();

  // -------------------------------
  // Auto Route controls and logic
  // -------------------------------
  const autoStartBtn = document.getElementById('autoStart');
  const autoStopBtn = document.getElementById('autoStop');
  const downloadRoutesBtn = document.getElementById('downloadRoutes');
  const backendStartBtn = document.getElementById('backendStart');
  const backendStopBtn = document.getElementById('backendStop');
  const backendStartBatchBtn = document.getElementById('backendStartBatch');
  const backendStopBatchBtn = document.getElementById('backendStopBatch');
  const batchNumRunsInput = document.getElementById('batchNumRuns');
  const autoRadiusInput = document.getElementById('autoDiscoveryRadius');
  const autoTargetDistanceInput = document.getElementById('autoTargetDistance');

  if (autoStopBtn) autoStopBtn.disabled = true;
  if (backendStopBtn) backendStopBtn.disabled = true;
  if (backendStopBatchBtn) backendStopBatchBtn.disabled = true;

  if (autoStartBtn) {
    autoStartBtn.addEventListener('click', async () => {
      if (autoActive) {
        showNotification('Auto Route is already running.');
        return;
      }
      if (!selectedLandmark) {
        showNotification('Pick a landmark first (click from the list or map).');
        return;
      }
      const radiusMeters = Math.max(5, Number(autoRadiusInput ? autoRadiusInput.value : 80) || 80);
      const targetMeters = Math.max(10, Number(autoTargetDistanceInput ? autoTargetDistanceInput.value : 2000) || 2000);
      const livePlot = true; // Always enable live plotting

      autoActive = true;
      autoStopRequested = false;
      if (autoStartBtn) autoStartBtn.disabled = true;
      if (autoStopBtn) autoStopBtn.disabled = false;

      // Clear previous polyline
      if (autoPolyline) { autoPolyline.setMap(null); autoPolyline = null; }
      autoPolyline = new google.maps.Polyline({
        map,
        path: [],
        strokeColor: '#673AB7',
        strokeWeight: 4,
        strokeOpacity: 0.9
      });

      // 1) Use the currently selected panorama (via map clicks) as the start
      if (!currentPanoData) {
        showNotification('Click the map to select a starting panorama first.');
        autoActive = false;
        if (autoStartBtn) autoStartBtn.disabled = false;
        if (autoStopBtn) autoStopBtn.disabled = true;
        return;
      }
      showNotification('Starting from the selected panorama...');
      const seedPano = {
        panoId: currentPanoData.panoId,
        position: currentPanoData.position,
        description: currentPanoData.description || ''
      };

      // Move Street View to the found pano
      panorama.setPano(seedPano.panoId);
      if (panoMarker) panoMarker.setMap(null);
      panoMarker = new google.maps.Marker({
        position: seedPano.position,
        map,
        icon: { path: google.maps.SymbolPath.CIRCLE, scale: 6, fillColor: '#4285F4', fillOpacity: 1, strokeWeight: 0 },
        title: 'Auto Start Pano'
      });
      map.panTo(seedPano.position);

      // 2) Stochastic walk away until target distance
      showNotification('Walking away stochastically...');
      const walkResult = await stochasticWalkAway(seedPano, targetMeters, {
        livePlot,
        polyline: autoPolyline
      });

      if (!walkResult || walkResult.points.length < 2) {
        showNotification('Walk failed or stopped early.');
        autoActive = false;
        if (autoStartBtn) autoStartBtn.disabled = false;
        if (autoStopBtn) autoStopBtn.disabled = true;
        return;
      }

      // 3) Record to paths and routes
      const startPoint = walkResult.points[0];
      const endPoint = walkResult.points[walkResult.points.length - 1];
      const pathEntry = {
        start: {
          panoId: startPoint.panoId,
          position: startPoint.position,
          panoPosition: startPoint.position,
          description: selectedLandmark.name
        },
        end: {
          panoId: endPoint.panoId,
          position: endPoint.position,
          panoPosition: endPoint.position,
          description: 'AutoRoute end'
        },
        distance: (walkResult.totalDistanceMeters / 1000).toFixed(2),
        duration: undefined
      };
      paths.push(pathEntry);
      if (typeof updateJsonPreview === 'function') updateJsonPreview();
      saveLocalPaths();

      const routeEntry = {
        id: 'route-' + Date.now(),
        start: pathEntry.start,
        end: pathEntry.end,
        totalDistanceMeters: walkResult.totalDistanceMeters,
        points: walkResult.points
      };
      routes.push(routeEntry);

      // 4) Show markers for start/end and fit map
      if (currentStartMarker) currentStartMarker.setMap(null);
      if (currentEndMarker) currentEndMarker.setMap(null);
      currentStartMarker = new google.maps.Marker({
        position: new google.maps.LatLng(pathEntry.start.position.lat, pathEntry.start.position.lng),
        map,
        icon: { path: google.maps.SymbolPath.CIRCLE, scale: 8, fillColor: '#00C853', fillOpacity: 1, strokeWeight: 0 },
        title: 'Auto Start'
      });
      currentEndMarker = new google.maps.Marker({
        position: new google.maps.LatLng(pathEntry.end.position.lat, pathEntry.end.position.lng),
        map,
        icon: { path: google.maps.SymbolPath.CIRCLE, scale: 8, fillColor: '#D500F9', fillOpacity: 1, strokeWeight: 0 },
        title: 'Auto End'
      });
      const b = new google.maps.LatLngBounds();
      b.extend(currentStartMarker.getPosition());
      b.extend(currentEndMarker.getPosition());
      map.fitBounds(b, 80);

      showNotification('Auto route complete. Distance ≈ ' + Math.round(walkResult.totalDistanceMeters) + ' m');
      autoActive = false;
      if (autoStartBtn) autoStartBtn.disabled = false;
      if (autoStopBtn) autoStopBtn.disabled = true;
    });
  }

  if (autoStopBtn) {
    autoStopBtn.addEventListener('click', () => {
      if (!autoActive) return;
      autoStopRequested = true;
      showNotification('Stopping auto route...');
    });
  }

  if (downloadRoutesBtn) {
    downloadRoutesBtn.addEventListener('click', () => {
      if (!routes.length) {
        showNotification('No routes to download.');
        return;
      }
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify({ routes }, null, 2));
      const a = document.createElement('a');
      a.setAttribute('href', dataStr);
      a.setAttribute('download', 'routes.json');
      document.body.appendChild(a);
      a.click();
      a.remove();
    });
  }

  // -------------------------------
  // Backend compute controls
  // -------------------------------
  const SERVICE_BASE = 'http://127.0.0.1:8765';  // API server port
  let backendPollTimer = null;
  let lastJobId = null;

  // Polygon file source UI wiring
  const polygonFileInput = document.getElementById('polygonFilePath');
  const polygonSourceIndicator = document.getElementById('polygonSourceIndicator');
  function updatePolygonIndicator() {
    let txt = 'No polygon file loaded';
    if (POLYGON_FILE) {
      txt = `Using: ${POLYGON_FILE}`;
    } else {
      txt = 'Awaiting polygon file detection...';
    }
    if (polygonSourceIndicator) polygonSourceIndicator.textContent = txt;
  }
  if (polygonFileInput) {
    polygonFileInput.addEventListener('input', () => {
      const v = (polygonFileInput.value || '').trim();
      POLYGON_FILE = v || null;
      updatePolygonIndicator();
      try { window.dispatchEvent(new CustomEvent('polygonFileChanged', { detail: { file: POLYGON_FILE } })); } catch(_) {}
    });
  }
  updatePolygonIndicator();

  if (backendStartBtn) {
    backendStartBtn.addEventListener('click', async () => {
      if (!currentPanoData || !currentPanoData.panoId) {
        showNotification('Select a starting panorama first (we need panoId).');
        return;
      }
      const targetMeters = Math.max(10, Number(autoTargetDistanceInput ? autoTargetDistanceInput.value : 2000) || 400);
      const randomForward = true; // Always use exploratory mode
      try {
        const res = await fetch(SERVICE_BASE + '/api/route/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            seedPanoId: currentPanoData.panoId,
            targetMeters,
            randomForward,
            regressWindow: 25,
            minFinalDegree: 3,
            polygonFile: POLYGON_FILE || undefined
          })
        });
        const json = await res.json();
        if (!res.ok) {
          showNotification('Backend start error: ' + (json.error || res.status));
          return;
        }
        showNotification('Backend route started.');
        // capture jobId (new API returns it)
        if (json.jobId) lastJobId = json.jobId; else lastJobId = null;
        if (backendStartBtn) backendStartBtn.disabled = true;
        if (backendStopBtn) backendStopBtn.disabled = false;
        // reset live polyline
        if (autoPolyline) { autoPolyline.setMap(null); autoPolyline = null; }
        autoPolyline = new google.maps.Polyline({ map, path: [], strokeColor: '#009688', strokeWeight: 0, strokeOpacity: 0 });
        // Manual polling only - no automatic timer
        if (backendPollTimer) clearInterval(backendPollTimer);
      } catch (e) {
        showNotification('Backend start failed: ' + e.message);
      }
    });
  }

  if (backendStopBtn) {
    backendStopBtn.addEventListener('click', async () => {
      try {
        const body = lastJobId ? { jobId: lastJobId } : {};
        await fetch(SERVICE_BASE + '/api/route/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        showNotification('Backend stopping...');
      } catch (e) {
        showNotification('Backend stop failed: ' + e.message);
      }
    });
  }

  // Batch controls
  let batchPollTimer = null;
  let lastBatchId = null;
  let lastBatchJobIds = [];
  const batchPolylines = {}; // jobId -> Polyline
  const batchColors = {}; // jobId -> color string

  function hsvToRgb(h, s, v) {
    let f = (n, k = (n + h / 60) % 6) => v - v * s * Math.max(Math.min(k, 4 - k, 1), 0);
    const r = Math.round(f(5) * 255), g = Math.round(f(3) * 255), b = Math.round(f(1) * 255);
    return `rgb(${r},${g},${b})`;
  }
  function assignBatchColors(jobIds) {
    const n = jobIds.length || 1;
    for (let i = 0; i < jobIds.length; i++) {
      const hue = Math.round((360 * i) / n);
      batchColors[jobIds[i]] = hsvToRgb(hue, 0.9, 0.95);
    }
  }

  // Helper function to get current destination key for seed selection
  function getCurrentDestinationKey() {
    if (!currentPanoData || !currentPanoData.position) return null;

    // Try to find which polygon/destination this pano belongs to
    if (typeof window !== 'undefined' && window.SEED_PANOS_BY_LANDMARK) {
      for (const [key, seeds] of Object.entries(window.SEED_PANOS_BY_LANDMARK)) {
        // Check if current pano is in the seed list for this destination
        if (Array.isArray(seeds)) {
          const found = seeds.find(s => s && s.panoId === currentPanoData.panoId);
          if (found) return key;
        }
      }
    }

    return null;
  }

  if (backendStartBatchBtn) {
    backendStartBatchBtn.addEventListener('click', async () => {
      if (!currentPanoData || !currentPanoData.panoId) {
        showNotification('Select a starting panorama first (we need panoId).');
        return;
      }
      const targetMeters = Math.max(10, Number(autoTargetDistanceInput ? autoTargetDistanceInput.value : 2000) || 400);
      const randomForward = true; // Always use exploratory mode
      const numRuns = Math.max(1, Number(batchNumRunsInput ? batchNumRunsInput.value : 3) || 3);
      try {
        // Prepare available seeds for batch API
        let availableSeeds = [];
        if (typeof window !== 'undefined' && window.SEED_PANOS_BY_LANDMARK) {
          // Get seeds for current destination if available
          const currentKey = getCurrentDestinationKey();
          if (currentKey && window.SEED_PANOS_BY_LANDMARK[currentKey]) {
            availableSeeds = window.SEED_PANOS_BY_LANDMARK[currentKey];
          }
        }

        const res = await fetch(SERVICE_BASE + '/api/route/start_batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            seedPanoId: currentPanoData.panoId,
            availableSeeds: availableSeeds,
            numRuns,
            targetMeters,
            randomForward,
            regressWindow: 25,
            minFinalDegree: 3,
            polygonFile: POLYGON_FILE || undefined
          })
        });
        const json = await res.json();
        if (!res.ok) { showNotification('Batch start error: ' + (json.error || res.status)); return; }
        lastBatchId = json.batchId || null;
        lastBatchJobIds = Array.isArray(json.jobIds) ? json.jobIds : [];
        // clear old polylines
        Object.values(batchPolylines).forEach(pl => { try { pl.setMap(null); } catch(e){} });
        Object.keys(batchPolylines).forEach(k => delete batchPolylines[k]);
        Object.keys(batchColors).forEach(k => delete batchColors[k]);
        assignBatchColors(lastBatchJobIds);
        showNotification('Batch started with ' + (json.jobIds ? json.jobIds.length : numRuns) + ' jobs.');
        if (backendStopBatchBtn) backendStopBatchBtn.disabled = false;
        if (batchPollTimer) clearInterval(batchPollTimer);
      } catch (e) {
        showNotification('Batch start failed: ' + e.message);
      }
    });
  }

  if (backendStopBatchBtn) {
    backendStopBatchBtn.addEventListener('click', async () => {
      if (!lastBatchId) { showNotification('No batch to stop.'); return; }
      try {
        await fetch(SERVICE_BASE + '/api/route/stop_batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ batchId: lastBatchId }) });
        showNotification('Batch stopping...');
      } catch (e) {
        showNotification('Batch stop failed: ' + e.message);
      }
    });
  }

  async function pollBatchStatus() {
    try {
      if (!lastBatchId) return;
      const res = await fetch(SERVICE_BASE + '/api/route/batch_status?batchId=' + encodeURIComponent(lastBatchId));
      const json = await res.json();
      if (!json || !json.jobs) return;
      // Update/create polylines for all jobs
      for (const j of json.jobs) {
        if (!j || !Array.isArray(j.points)) continue;
        if (!batchPolylines[j.jobId]) {
          const color = batchColors[j.jobId] || '#006064';
          batchPolylines[j.jobId] = new google.maps.Polyline({ map, path: [], strokeColor: color, strokeWeight: 3, strokeOpacity: 0.8 });
        }
        const path = batchPolylines[j.jobId].getPath();
        path.clear();
        j.points.forEach(p => path.push(new google.maps.LatLng(p.position.lat, p.position.lng)));
      }
      if (json.status !== 'running') {
        if (batchPollTimer) { clearInterval(batchPollTimer); batchPollTimer = null; }
        showNotification('Batch done. Rendered ' + json.jobs.length + ' routes.');
      }
    } catch (e) {
      showNotification('Batch poll failed: ' + e.message);
    }
  }

  async function pollBackendStatus() {
    try {
      const url = lastJobId ? (SERVICE_BASE + '/api/route/status?jobId=' + encodeURIComponent(lastJobId)) : (SERVICE_BASE + '/api/route/status');
      const res = await fetch(url);
      const json = await res.json();
      const status = json.status;
      const pts = Array.isArray(json.points) ? json.points : [];
      const msg = json.message || json.error || '';
      // live update trail with per-segment gradient (recent = brighter)
      // Clear old segment overlays stored on polyline instance
      if (!autoPolyline.__segments) autoPolyline.__segments = [];
      autoPolyline.__segments.forEach(seg => { try { seg.setMap(null); } catch(e){} });
      autoPolyline.__segments = [];
      const n = pts.length;
      if (n >= 2) {
        const maxSeg = Math.min(200, n - 1);
        for (let i = n - maxSeg - 1; i < n - 1; i++) {
          if (i < 0) continue;
          const a = pts[i], b = pts[i+1];
          const frac = (i - (n - maxSeg - 1)) / Math.max(1, maxSeg);
          const opacity = 0.25 + 0.6 * frac; // older faint, newer brighter
          const hue = 200 + Math.round(140 * frac); // teal→purple sweep
          const color = `hsl(${hue}, 85%, 50%)`;
          const seg = new google.maps.Polyline({
            map,
            path: [new google.maps.LatLng(a.position.lat, a.position.lng), new google.maps.LatLng(b.position.lat, b.position.lng)],
            strokeColor: color,
            strokeWeight: 5,
            strokeOpacity: opacity
          });
          autoPolyline.__segments.push(seg);
        }
      }
      // Also log backend walk events (if provided)
      if (Array.isArray(json.events) && json.events.length) {
        console.log('[backend]', json.events.join('\n'));
      }
      if (status === 'running') return;
      // finalize
      if (backendPollTimer) { clearInterval(backendPollTimer); backendPollTimer = null; }
      if (backendStartBtn) backendStartBtn.disabled = false;
      if (backendStopBtn) backendStopBtn.disabled = true;
      if (!pts.length) { showNotification('Backend: no points. Reason: ' + msg); return; }
      const startPoint = pts[0];
      const endPoint = pts[pts.length - 1];
      const routeEntry = {
        id: 'route-' + Date.now(),
        start: { panoId: startPoint.panoId, position: startPoint.position, panoPosition: startPoint.position, description: 'Backend start' },
        end: { panoId: endPoint.panoId, position: endPoint.position, panoPosition: endPoint.position, description: 'Backend end' },
        totalDistanceMeters: json.totalDistanceMeters || 0,
        points: pts
      };
      routes.push(routeEntry);
      // Also add to paths for convenience
      paths.push({ start: routeEntry.start, end: routeEntry.end, distance: ((json.totalDistanceMeters || 0)/1000).toFixed(2) });
      if (typeof updateJsonPreview === 'function') updateJsonPreview();
      saveLocalPaths();
      // Markers and fit
      if (currentStartMarker) currentStartMarker.setMap(null);
      if (currentEndMarker) currentEndMarker.setMap(null);
      currentStartMarker = new google.maps.Marker({ position: new google.maps.LatLng(startPoint.position.lat, startPoint.position.lng), map, icon: { path: google.maps.SymbolPath.CIRCLE, scale: 8, fillColor: '#00C853', fillOpacity: 1, strokeWeight: 0 }, title: 'Backend Start' });
      currentEndMarker = new google.maps.Marker({ position: new google.maps.LatLng(endPoint.position.lat, endPoint.position.lng), map, icon: { path: google.maps.SymbolPath.CIRCLE, scale: 8, fillColor: '#D500F9', fillOpacity: 1, strokeWeight: 0 }, title: 'Backend End' });
      const b = new google.maps.LatLngBounds();
      b.extend(currentStartMarker.getPosition());
      b.extend(currentEndMarker.getPosition());
      map.fitBounds(b, 80);
      showNotification('Backend route ' + status + '. Distance ≈ ' + Math.round(json.totalDistanceMeters || 0) + ' m. Reason: ' + msg);
    } catch (e) {
      showNotification('Backend poll failed: ' + e.message);
    }
  }

  // Manual refresh functions (no automatic polling)
  async function manualRefreshBackendStatus() {
    try {
      if (lastJobId) {
        await pollBackendStatus();
        showNotification('Backend status refreshed');
      } else {
        showNotification('No active job to refresh');
      }
    } catch (e) {
      showNotification('Manual refresh failed: ' + e.message);
    }
  }

  async function manualRefreshBatchStatus() {
    try {
      if (lastBatchId) {
        await pollBatchStatus();
        showNotification('Batch status refreshed');
      } else {
        showNotification('No active batch to refresh');
      }
    } catch (e) {
      showNotification('Batch refresh failed: ' + e.message);
    }
  }

  // Keyboard shortcuts for manual refresh
  document.addEventListener('keydown', function(e) {
    // Ctrl+R or Cmd+R for backend refresh
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
      e.preventDefault();
      manualRefreshBackendStatus();
    }
    // Ctrl+B or Cmd+B for batch refresh
    if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
      e.preventDefault();
      manualRefreshBatchStatus();
    }
  });

  // Find a random navigable pano around a center point
  async function findRandomNavigablePanoAround(centerLatLng, radiusMeters, attempts) {
    const maxAttempts = Math.max(1, attempts || 40);
    for (let i = 0; i < maxAttempts; i++) {
      const candidate = randomPointInRadius(centerLatLng, radiusMeters);
      // Use a small search radius to snap to nearest pano
      const result = await getPanoramaAsync({ location: candidate, radius: 50, preference: google.maps.StreetViewPreference.NEAREST });
      if (result && result.links && result.links.length) {
        return {
          panoId: result.location.pano,
          position: result.location.latLng,
          description: result.location.description || '',
          links: result.links
        };
      }
    }
    return null;
  }

  // Stochastic walk away from the start pano
  async function stochasticWalkAway(seedPano, targetMeters, opts) {
    const options = opts || {};
    const pathPoints = [{ panoId: seedPano.panoId, position: seedPano.position.toJSON() }];
    const visitedPanos = new Set([seedPano.panoId]);
    let current = { panoId: seedPano.panoId, position: seedPano.position };
    let prevPanoId = null;
    const startLatLng = seedPano.position;
    const stepLimit = 200;

    if (options.polyline && options.livePlot) {
      const path = options.polyline.getPath();
      path.push(seedPano.position);
    }

    for (let step = 0; step < stepLimit; step++) {
      if (autoStopRequested) break;

      const curData = await getPanoramaAsync({ pano: current.panoId });
      if (!curData || !curData.links || !curData.links.length) break;

      const currentStraight = computeDistanceMeters(startLatLng, curData.location.latLng);

      // Do not backtrack to the immediate previous pano
      const candidates = curData.links.filter(l => l.pano !== prevPanoId);
      // Prefer unvisited; if none unvisited remain, stop to avoid loops
      const unvisited = candidates.filter(l => !visitedPanos.has(l.pano));
      if (!unvisited.length) break;

      // Evaluate all unvisited next steps
      const evaluated = [];
      for (let i = 0; i < unvisited.length; i++) {
        const link = unvisited[i];
        const nextData = await getPanoramaAsync({ pano: link.pano });
        if (!nextData || !nextData.location || !nextData.location.latLng) continue;
        const nextLatLng = nextData.location.latLng;
        const newStraight = computeDistanceMeters(startLatLng, nextLatLng);
        evaluated.push({ id: link.pano, latLng: nextLatLng, dist: newStraight });
      }

      if (!evaluated.length) break;

      // Prefer the option that maximizes distance from start; if none improve, pick a random unvisited
      const improving = evaluated.filter(e => e.dist > currentStraight);
      let chosen = null;
      if (improving.length) {
        chosen = improving.reduce((a, b) => (a.dist >= b.dist ? a : b));
      } else {
        chosen = evaluated[Math.floor(Math.random() * evaluated.length)];
      }

      // Commit step
      panorama.setPano(chosen.id);
      pathPoints.push({ panoId: chosen.id, position: chosen.latLng.toJSON() });
      visitedPanos.add(chosen.id);

      if (options.polyline && options.livePlot) {
        const path = options.polyline.getPath();
        path.push(chosen.latLng);
      }

      if (chosen.dist >= targetMeters) {
        return { points: pathPoints, totalDistanceMeters: chosen.dist };
      }

      prevPanoId = current.panoId;
      current = { panoId: chosen.id, position: chosen.latLng };
      await delay(120);
    }

    const endLatLng = new google.maps.LatLng(pathPoints[pathPoints.length - 1].position.lat, pathPoints[pathPoints.length - 1].position.lng);
    const straightDist = computeDistanceMeters(startLatLng, endLatLng);
    return { points: pathPoints, totalDistanceMeters: straightDist };
  }

  // ---- Small utilities ----
  function randomPointInRadius(centerLatLng, radiusMeters) {
    const r = radiusMeters * Math.sqrt(Math.random());
    const theta = Math.random() * 2 * Math.PI;
    const dx = r * Math.cos(theta);
    const dy = r * Math.sin(theta);
    const dLat = dy / 111111; // meters to degrees
    const dLng = dx / (111111 * Math.cos(centerLatLng.lat() * Math.PI / 180));
    return new google.maps.LatLng(centerLatLng.lat() + dLat, centerLatLng.lng() + dLng);
  }

  function computeDistanceMeters(a, b) {
    const R = 6371000;
    const lat1 = a.lat() * Math.PI / 180;
    const lat2 = b.lat() * Math.PI / 180;
    const dLat = (b.lat() - a.lat()) * Math.PI / 180;
    const dLng = (b.lng() - a.lng()) * Math.PI / 180;
    const sa = Math.sin(dLat/2) * Math.sin(dLat/2) + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng/2) * Math.sin(dLng/2);
    const c = 2 * Math.atan2(Math.sqrt(sa), Math.sqrt(1-sa));
    return R * c;
  }

  function computeBearing(fromLatLng, toLatLng) {
    const lat1 = fromLatLng.lat() * Math.PI / 180;
    const lat2 = toLatLng.lat() * Math.PI / 180;
    const dLng = (toLatLng.lng() - fromLatLng.lng()) * Math.PI / 180;
    const y = Math.sin(dLng) * Math.cos(lat2);
    const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng);
    let brng = Math.atan2(y, x) * 180 / Math.PI;
    return (brng + 360) % 360;
  }

  function headingDiff(a, b) {
    const d = Math.abs(a - b) % 360;
    return d > 180 ? 360 - d : d;
  }

  function delay(ms) { return new Promise(res => setTimeout(res, ms)); }

  function getPanoramaAsync(request) {
    return new Promise((resolve) => {
      streetViewService.getPanorama(request, (data, status) => {
        if (status === 'OK' && data) resolve(data); else resolve(null);
      });
    });
  }
}

function showNotification(message) {
  const notif = document.getElementById('notification');
  if (!notif) {
    console.error("Notification element not found.");
    return;
  }
  notif.textContent = message;
  notif.style.display = 'block';
  if (!document.fullscreenElement) {
    setTimeout(() => { notif.style.display = 'none'; }, 3000);
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  const notif = document.getElementById('notification');
  if (notif) {
    notif.addEventListener('click', () => { notif.style.display = 'none'; });
  }

  // Deterministic polygon file detection
  try {
    const detectedFile = await autoDetectLatestPolygonFile();
    if (detectedFile) {
      POLYGON_FILE = detectedFile;
      // Update the input field and trigger the change event
      const polygonFileInput = document.getElementById('polygonFilePath');
      if (polygonFileInput) {
        polygonFileInput.value = detectedFile;
        // Trigger the input event to update the system
        polygonFileInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
    } else {
      console.error('❌ No valid polygon file found - manual file selection required');
    }
  } catch (e) {
    console.error('❌ Polygon file auto-detection failed:', e);
  }
});

window.initMap = initMap;
