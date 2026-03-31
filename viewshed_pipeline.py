#%% Imports and Configuration
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
import rasterio
from rasterio.features import shapes, rasterize
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Use script directory so the path is portable
base_dir = os.path.dirname(os.path.abspath(__file__))
dem_file = os.path.join(base_dir, "DEM_Subset-WithBuildings.tif")
marks_file = os.path.join(base_dir, "Marks_Brief2.shp")
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)


#%% Bresenham Line Algorithm
def get_line(r0, c0, r1, c1):
    """Generates pixel coordinates along a line using Bresenham's algorithm."""
    points = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        points.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc

    if not points:
        return np.array([], dtype=int), np.array([], dtype=int)
    rr, cc = zip(*points)
    return np.array(rr), np.array(cc)


#%% Viewshed Computation Function (Correct Border-Ray Algorithm)
def compute_viewshed(dem, transform, obs_row, obs_col, observer_height=1.6):
    """
    Compute a Boolean viewshed raster using the standard border-ray approach.

    For each border pixel, cast a ray from the observer. Walk along the ray
    and track the maximum elevation angle seen so far. Any pixel whose
    elevation angle exceeds the running maximum is visible.

    Parameters
    ----------
    dem : 2-D ndarray  – surface elevation grid (with buildings baked in)
    transform : Affine – rasterio geo-transform
    obs_row, obs_col : int – observer pixel indices
    observer_height : float – height above ground (metres)

    Returns
    -------
    viewshed : 2-D uint8 ndarray (1 = visible, 0 = not visible)
    """
    rows, cols = dem.shape
    res_x = transform[0]  # pixel width in metres

    obs_elev = dem[obs_row, obs_col] + observer_height
    viewshed = np.zeros((rows, cols), dtype=np.uint8)
    viewshed[obs_row, obs_col] = 1  # observer can always see itself

    # Handle NaN / nodata in the DEM by treating them as -inf (transparent)
    safe_dem = np.where(np.isnan(dem), -np.inf, dem)

    # Collect unique border pixels
    border = set()
    for c in range(cols):
        border.add((0, c))
        border.add((rows - 1, c))
    for r in range(1, rows - 1):
        border.add((r, 0))
        border.add((r, cols - 1))

    for (br, bc) in border:
        rr, cc = get_line(obs_row, obs_col, br, bc)
        if len(rr) < 2:
            continue

        max_angle = -np.inf  # running maximum elevation angle

        # Start from index 1 (skip the observer cell itself)
        for i in range(1, len(rr)):
            r, c = int(rr[i]), int(cc[i])
            dist = np.sqrt((r - obs_row) ** 2 + (c - obs_col) ** 2) * res_x
            if dist == 0:
                continue

            elev_angle = (safe_dem[r, c] - obs_elev) / dist

            if elev_angle >= max_angle:
                viewshed[r, c] = 1
                max_angle = elev_angle
            # If elev_angle < max_angle the pixel is hidden; leave it 0

    return viewshed


#%% Load DEM
with rasterio.open(dem_file) as src:
    dem_data = src.read(1)
    transform = src.transform
    dem_crs = src.crs
    rows, cols = dem_data.shape

print(f"Loaded DEM — shape: {dem_data.shape}, CRS: {dem_crs}")
print(f"  Elevation range: {np.nanmin(dem_data):.2f} – {np.nanmax(dem_data):.2f} m")


#%% Load and Reproject Viewpoint Marks
marks = gpd.read_file(marks_file).to_crs(dem_crs)

viewpoints = []
for _, row in marks.iterrows():
    geom = row.geometry
    if geom.geom_type == "MultiPoint":
        for pt in geom.geoms:
            viewpoints.append((row["id"], pt))
    elif geom.geom_type == "Point":
        viewpoints.append((row["id"], geom))

print(f"Extracted {len(viewpoints)} viewpoint(s):")
for vp_id, pt in viewpoints:
    r, c = rowcol(transform, pt.x, pt.y)
    print(f"  ID {vp_id}: ({pt.x:.2f}, {pt.y:.2f}) → pixel ({r}, {c})")


#%% Compute Viewsheds for Each Observer
observer_height = 1.6
all_viewshed_masks = {}   # id → 2-D uint8 array
out_polygons = []

for vp_id, pt in viewpoints:
    r0, c0 = rowcol(transform, pt.x, pt.y)

    if not (0 <= r0 < rows and 0 <= c0 < cols):
        print(f"⚠ Skipping observer {vp_id} — outside DEM bounds")
        continue

    print(f"Computing viewshed for observer {vp_id} at pixel ({r0}, {c0}) …")
    vs = compute_viewshed(dem_data, transform, r0, c0, observer_height)
    all_viewshed_masks[vp_id] = vs

    visible_count = int(vs.sum())
    total = rows * cols
    print(f"  → {visible_count}/{total} pixels visible ({100*visible_count/total:.1f}%)")

    # Save per-observer raster for debugging / further analysis
    raster_out = os.path.join(output_dir, f"viewshed_observer_{vp_id}.tif")
    with rasterio.open(
        raster_out, "w", driver="GTiff",
        height=rows, width=cols, count=1,
        dtype="uint8", crs=dem_crs, transform=transform,
    ) as dst:
        dst.write(vs, 1)
    print(f"  Saved raster → {raster_out}")

    # Polygonize (preserving holes from building occlusion)
    for geom_dict, value in shapes(vs, mask=(vs == 1), transform=transform):
        poly = shape(geom_dict)  # correctly handles exterior + interior rings
        out_polygons.append({"observer": int(vp_id), "geometry": poly})


#%% Save Combined Vector Layer (Shapefile)
if out_polygons:
    out_gdf = gpd.GeoDataFrame(out_polygons, crs=dem_crs)
    out_shp = os.path.join(output_dir, "Viewsheds_Output.shp")
    out_gdf.to_file(out_shp)
    print(f"\nSaved viewshed vector layer → {out_shp}")
else:
    print("No viewsheds generated.")


#%% Visualization
fig, axes = plt.subplots(1, len(all_viewshed_masks) + 1,
                         figsize=(5 * (len(all_viewshed_masks) + 1), 5))

# Base DEM
axes[0].imshow(dem_data, cmap="terrain")
axes[0].set_title("DEM (with buildings)")

cmap_vs = ListedColormap(["none", "red"])
for i, (vp_id, vs) in enumerate(all_viewshed_masks.items()):
    ax = axes[i + 1]
    ax.imshow(dem_data, cmap="terrain")
    ax.imshow(vs, cmap=cmap_vs, alpha=0.45)
    # Mark observer location
    r0, c0 = rowcol(transform, viewpoints[i][1].x, viewpoints[i][1].y)
    ax.plot(c0, r0, "k*", markersize=12)
    ax.set_title(f"Viewshed – Observer {vp_id}")

plt.tight_layout()
vis_path = os.path.join(output_dir, "viewshed_overview.png")
plt.savefig(vis_path, dpi=150)
plt.close()
print(f"Saved visualization → {vis_path}")


#%% 3D Volume Export (.obj) — Shared-Vertex Mesh
obj_path = os.path.join(output_dir, "Viewshed_Volume.obj")
print("Exporting 3D volume mesh …")

# Build combined viewshed mask from all observers
combined = np.zeros((rows, cols), dtype=np.uint8)
for vs in all_viewshed_masks.values():
    combined = np.maximum(combined, vs)

# Build a shared vertex grid and emit only visible quads
vertex_index = np.full((rows, cols), -1, dtype=int)
vertices = []
idx = 1  # OBJ is 1-indexed

for r in range(rows):
    for c in range(cols):
        # A vertex is needed if any of the 4 quads touching it is visible
        needed = False
        for dr, dc in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
            qr, qc = r + dr, c + dc
            if 0 <= qr < rows - 1 and 0 <= qc < cols - 1 and combined[qr, qc]:
                needed = True
                break
        if needed:
            x, y = transform * (c, r)
            z = float(dem_data[r, c])
            vertices.append((x, y, z))
            vertex_index[r, c] = idx
            idx += 1

faces = []
for r in range(rows - 1):
    for c in range(cols - 1):
        if combined[r, c]:
            v1 = vertex_index[r, c]
            v2 = vertex_index[r, c + 1]
            v3 = vertex_index[r + 1, c + 1]
            v4 = vertex_index[r + 1, c]
            if v1 > 0 and v2 > 0 and v3 > 0 and v4 > 0:
                faces.append((v1, v2, v3, v4))

with open(obj_path, "w") as f:
    f.write("# 3D Viewshed Volume – El-Bagawat\n")
    f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n")
    f.write("o Viewshed_Volume\n\n")
    for x, y, z in vertices:
        f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
    f.write("\n")
    for v1, v2, v3, v4 in faces:
        f.write(f"f {v1} {v2} {v3} {v4}\n")

print(f"Exported 3D volume → {obj_path}  ({len(vertices)} verts, {len(faces)} faces)")

# %%
