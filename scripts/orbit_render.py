"""
Orbit renders of the building: a camera 50 ft from the building, stepped
around the Z axis in 45-degree increments, with a high-resolution still
rendered at every stop.

Distance is measured from the building's outside face (its plan footprint),
not from its centre, so the camera keeps 50 ft of clearance at every angle.
At a corner view it sits 50 ft from the corner, and at a straight-on view
it sits 50 ft from the facade. Use --from-center to measure from the
centre point instead.

Angle 0 looks at the front (the -Y side, where the entry is). Angles increase
counter-clockwise seen from above: 0 = front, 90 = east (+X) side, 180 = rear,
270 = west side.

Run headless (recommended):
    blender -b t6_outlook.blend -P scripts/orbit_render.py -- --out ./renders/orbit

Or open the .blend in Blender, load this file in the Text Editor, and press
Run Script. Renders then go to "renders/orbit" next to the .blend file.

Options (everything after "--"):
    --out DIR           output folder (default: //renders/orbit, next to the .blend)
    --distance-ft N     clearance from the building in feet (default 50)
    --step-deg N        rotation step around Z in degrees (default 45)
    --height-ft N       camera height above grade in feet
                        (default: half the building height, keeps verticals straight)
    --lens-mm N         fixed focal length; by default the lens is fitted so the
                        whole building is in frame
    --res WxH           output resolution (default 3840x2400)
    --samples N         override render samples (EEVEE and Cycles)
    --engine NAME       override render engine, e.g. CYCLES (default: keep the file's)
    --from-center       measure the distance from the building centre, not its facade
    --keep-camera       leave the orbit camera in the scene afterwards

The script does not save the .blend file. The scene's active camera and
render settings are restored when it finishes.
"""

import argparse
import math
import os
import sys

import bpy
from mathutils import Vector

FT = 0.3048  # metres per foot

# Collections whose names contain any of these are not part of "the building"
# (site paving, boundary walls, setting-out grid). They still render; they are
# only left out when working out the building's footprint and height.
NOT_BUILDING_KEYWORDS = ("site", "grid")

CAMERA_NAME = "Orbit Camera"
FRAME_MARGIN = 1.06  # 6% breathing room around the building when fitting the lens


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="orbit_render.py")
    p.add_argument("--out", default="//renders/orbit")
    p.add_argument("--distance-ft", type=float, default=50.0)
    p.add_argument("--step-deg", type=float, default=45.0)
    p.add_argument("--height-ft", type=float, default=None)
    p.add_argument("--lens-mm", type=float, default=None)
    p.add_argument("--res", default="3840x2400")
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--engine", default=None)
    p.add_argument("--from-center", action="store_true")
    p.add_argument("--keep-camera", action="store_true")
    return p.parse_args(argv)


def feet_to_bu(scene, feet):
    """Feet to Blender units, honouring the scene's unit scale."""
    return feet * FT / scene.unit_settings.scale_length


def is_building_object(obj):
    if obj.type != "MESH" or obj.hide_render or not obj.visible_get():
        return False
    for coll in obj.users_collection:
        name = coll.name.lower()
        if coll.hide_render or any(k in name for k in NOT_BUILDING_KEYWORDS):
            return False
    return True


def building_bounds(scene, ground_z):
    """World-space bounding box of the building, clipped at grade (below-grade
    floors can't be seen from outside, so they shouldn't shrink the frame)."""
    pts = [
        obj.matrix_world @ Vector(corner)
        for obj in scene.objects
        if is_building_object(obj)
        for corner in obj.bound_box
    ]
    if not pts:
        raise RuntimeError("No visible mesh objects found to treat as the building.")
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), max(min(p.z for p in pts), ground_z)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def distance_to_footprint(x, y, lo, hi):
    dx = max(lo.x - x, 0.0, x - hi.x)
    dy = max(lo.y - y, 0.0, y - hi.y)
    return math.hypot(dx, dy)


def orbit_position(angle_deg, center, lo, hi, distance, from_center):
    """Point in plan at the given angle whose distance from the building
    (footprint edge, or centre point) is exactly `distance`."""
    a = math.radians(angle_deg)
    # Angle 0 is the front (-Y); counter-clockwise from above.
    dx, dy = math.sin(a), -math.cos(a)
    if from_center:
        return center.x + dx * distance, center.y + dy * distance

    # Walk outward from the centre until the clearance to the footprint equals
    # `distance`. Clearance only grows along the ray, so bisection is exact.
    near, far = 0.0, (hi - lo).length + distance * 2
    for _ in range(60):
        mid = (near + far) / 2
        if distance_to_footprint(center.x + dx * mid, center.y + dy * mid, lo, hi) < distance:
            near = mid
        else:
            far = mid
    return center.x + dx * far, center.y + dy * far


def fit_lens(cam_obj, corners, aspect):
    """Focal length that just fits every bounding-box corner in frame."""
    bpy.context.view_layer.update()
    inv = cam_obj.matrix_world.inverted()
    need = 0.0
    for c in corners:
        p = inv @ c  # camera looks down its local -Z
        depth = -p.z
        if depth <= 1e-6:
            continue
        need = max(need, abs(p.x) / depth, abs(p.y) / depth * aspect)
    need *= FRAME_MARGIN
    cam = cam_obj.data
    return cam.sensor_width / (2.0 * need) if need > 0 else cam.lens


def main():
    args = parse_args()
    scene = bpy.context.scene
    render = scene.render

    res_x, res_y = (int(v) for v in args.res.lower().split("x"))
    distance = feet_to_bu(scene, args.distance_ft)
    out_dir = bpy.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    lo, hi = building_bounds(scene, ground_z=0.0)
    center = (lo + hi) / 2
    cam_z = feet_to_bu(scene, args.height_ft) if args.height_ft is not None else center.z
    target = Vector((center.x, center.y, center.z))
    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]

    print(
        "Building: {:.1f} x {:.1f} ft footprint, {:.1f} ft tall; camera {:.1f} ft above grade".format(
            (hi.x - lo.x) / feet_to_bu(scene, 1),
            (hi.y - lo.y) / feet_to_bu(scene, 1),
            (hi.z - lo.z) / feet_to_bu(scene, 1),
            cam_z / feet_to_bu(scene, 1),
        )
    )

    # Remember what we change so the file is left as we found it.
    saved = {
        "camera": scene.camera,
        "res": (render.resolution_x, render.resolution_y, render.resolution_percentage),
        "filepath": render.filepath,
        "format": (render.image_settings.file_format, render.image_settings.color_mode,
                   render.image_settings.color_depth),
        "engine": render.engine,
        "eevee_samples": getattr(scene.eevee, "taa_render_samples", None),
        "cycles_samples": getattr(getattr(scene, "cycles", None), "samples", None),
    }

    cam_data = bpy.data.cameras.new(CAMERA_NAME)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.1
    cam_data.clip_end = max(1000.0, (hi - lo).length * 4)
    cam_obj = bpy.data.objects.new(CAMERA_NAME, cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    try:
        render.resolution_x, render.resolution_y = res_x, res_y
        render.resolution_percentage = 100
        render.image_settings.file_format = "PNG"
        render.image_settings.color_mode = "RGBA"
        render.image_settings.color_depth = "16"
        if args.engine:
            render.engine = args.engine
        if args.samples:
            if hasattr(scene.eevee, "taa_render_samples"):
                scene.eevee.taa_render_samples = args.samples
            if hasattr(scene, "cycles"):
                scene.cycles.samples = args.samples

        aspect = res_x / res_y
        steps = max(1, round(360.0 / args.step_deg))
        for i in range(steps):
            angle = (i * args.step_deg) % 360.0
            x, y = orbit_position(angle, center, lo, hi, distance, args.from_center)
            if distance_to_footprint(x, y, lo, hi) == 0.0:
                print(f"Warning: at {angle:.0f} deg the camera is inside the building footprint.")
            cam_obj.location = (x, y, cam_z)
            look = target - cam_obj.location
            cam_obj.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
            cam_data.lens = args.lens_mm or fit_lens(cam_obj, corners, aspect)

            path = os.path.join(out_dir, f"orbit_{i:02d}_{int(round(angle)):03d}deg.png")
            render.filepath = path
            print(f"[{i + 1}/{steps}] {angle:5.1f} deg  lens {cam_data.lens:.1f} mm  ->  {path}")
            bpy.ops.render.render(write_still=True)
    finally:
        scene.camera = saved["camera"]
        render.resolution_x, render.resolution_y, render.resolution_percentage = saved["res"]
        render.filepath = saved["filepath"]
        (render.image_settings.file_format, render.image_settings.color_mode,
         render.image_settings.color_depth) = saved["format"]
        render.engine = saved["engine"]
        if saved["eevee_samples"] is not None:
            scene.eevee.taa_render_samples = saved["eevee_samples"]
        if saved["cycles_samples"] is not None:
            scene.cycles.samples = saved["cycles_samples"]
        if not args.keep_camera:
            bpy.data.objects.remove(cam_obj)
            bpy.data.cameras.remove(cam_data)

    print(f"Done: {steps} renders in {out_dir}")


if __name__ == "__main__":
    main()
