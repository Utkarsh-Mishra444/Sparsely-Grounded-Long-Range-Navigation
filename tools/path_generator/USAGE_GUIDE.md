# How to Use the Path Generator

This guide walks you through generating Street View paths using the path generator tool, from setup to export.

---

## Prerequisites

1. **Install dependencies** (if not already done):
   ```bash
   cd tools/path_generator
   pip install -r requirements.txt
   playwright install chromium
   ```

2. **Set your Google Maps API key:**
   ```bash
   export GOOGLE_MAPS_API_KEY="your_api_key_here"
   ```
   You need these APIs enabled in Google Cloud Console:
   - Maps JavaScript API
   - Street View Static API

3. **Start the server:**
   ```bash
   python -m src.server
   ```

4. **Open:** `http://localhost:8766`

---

## Understanding Seed Panoramas

Seed panoramas ("seed panos") are the starting points for path generation. The pathfinder begins at a seed pano and walks outward through the Street View graph until it reaches the target distance.

Before generating paths, you need to add seed panos for the landmarks you want to use. Select seed panos on well-connected roads near your target landmarks.

When you add seeds via the UI, they are stored in your **browser's localStorage**. To share seeds with collaborators, see [Exporting & Sharing Seeds](#exporting--sharing-seeds).

---

## Step-by-Step Workflow

### Step 1: Add Seed Panos

Before generating paths, add seed panos for the landmarks you want to use:

1. Open the **Landmarks** tab (first tab in the sidebar).
2. Click a landmark name in the sidebar to zoom to it on the map.
3. Click on a road within the landmark's polygon boundary on the map. A red circle appears at your click location, and a blue circle shows the nearest Street View panorama found.
4. The Street View viewer (right panel) loads that panorama.
5. Click **"Add Seed Pano"** to save this panorama as a seed for the landmark.

> **Tip:** Choose seed panos on well-connected roads (intersections with multiple links work best). Avoid dead-end streets, alleys, or pedestrian-only areas.

### Step 2: Validate Seeds (Optional)

1. Switch to the **Bulk Operations** tab (third tab).
2. Select your destinations, then click **"Validate Seeds"**.
3. This checks that each seed pano's position falls inside a landmark polygon — useful if you want to confirm your seeds are correctly associated with their landmarks.

### Step 3: Configure and Start Bulk Path Generation

1. In the **Bulk Operations** tab, select your target destinations.
2. Set parameters:
   - **Target Distance (km):** How far each path should walk (e.g., 3 km).
   - **Runs per Destination:** How many different paths to generate per landmark.
   - **Min Existing:** Skip landmarks that already have this many paths.
   - **Bias Direction:** Click the compass to set a preferred walking direction (optional).
   - **Bias Decisions:** Number of initial decisions that use the direction bias.
3. Click **"Start"** to launch all jobs.

Jobs run in parallel on the backend. The pathfinder uses DFS with backtracking, annealing, and divergence control to produce varied paths.

### Step 4: Monitor Progress

**Paths do not auto-update on the map.** You must manually refresh by pressing **Cmd+U** (Mac) / **Ctrl+U** (Windows).

This polls the backend for job status and draws paths on the map as they progress. You'll see:
- Running/finished counts
- Total and average displacement
- Live polylines on the map (color-coded per run)

### Step 5: Commit and Export Paths

Once jobs finish:

1. Click **"Commit Live"** in the Bulk Operations tab to save completed paths to the database.
2. Switch to the **Paths** tab (second tab) to view all saved paths.
3. Use filters (city, destination) and sorting (displacement, points, decisions) to find paths.
4. Select paths using checkboxes, then click **"Download JSON"** to export.

---

## Keyboard Shortcuts

| Action | Mac | Windows/Linux |
|--------|-----|---------------|
| Start single pathfinding job | Cmd+R | Ctrl+R |
| Start batch jobs | Cmd+B | Ctrl+B |
| Refresh / view paths (Bulk Ops) | Cmd+U | Ctrl+U |

In the Street View / Landmarks tab:

| Action | Key |
|--------|-----|
| Set start pano | S |
| Set end pano | E |
| Add path | A |
| Toggle Street View coverage | T |
| Calculate distance | D |

---

## The Three Tabs

### Landmarks (Tab 1)
The main navigation view. Browse landmarks, explore Street View, add seed panos, and manually create individual paths (start-to-end pairs).

### Paths (Tab 2)
View and manage all generated paths stored in the database. Filter by city/destination, sort by various metrics, multi-select for export or deletion.

### Bulk Operations (Tab 3)
Generate multiple paths at once. Configure parameters, launch parallel jobs, monitor progress, and commit results to the database.

---

## Exporting & Sharing Seeds

Seeds added via the UI are stored in your browser's localStorage and are **not** shared with the repo. To share seeds with collaborators:

1. In the Landmarks tab, click **"Export Polygons"** — this downloads a JSON file containing all your edited polygons and seed panos.
2. Replace `data/polygons.json` with the exported file (or merge the seed panos into it).
3. Commit and push the updated `polygons.json`.

This ensures other users get your seed panos on a fresh clone.

---

## Creating Individual Paths (Manual Mode)

Instead of bulk generation, you can manually trace paths:

1. In the **Landmarks** tab, navigate to your desired start location in Street View.
2. Click **"Set Start"** (or press **S**) to save the current panorama as your path's start point.
3. Navigate through Street View to your desired end location.
4. Click **"Set End"** (or press **E**) to save the end point.
5. Click **"Add Path"** (or press **A**) to finalize the path.
6. Repeat for additional paths.
7. Click **"Done"** to export all manually created paths as `paths.json`.
8. Optionally click **"Commit Local"** to save them to the backend database.

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Paths not showing on the map | UI requires manual refresh | Press Cmd+U (Mac) / Ctrl+U (Windows) |
| Bulk jobs won't start | No seed panos for selected landmarks | Add seed panos first (see Step 1) |
| Jobs seem stuck | Backend still processing (DFS with backtracking can be slow) | Check the terminal running the server for log output |
| Map/Street View not loading | Invalid or missing API key | Verify `GOOGLE_MAPS_API_KEY` is set and has the required APIs enabled |
