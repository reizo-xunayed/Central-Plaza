"""
Turntable video of the building: a camera 50 ft from the building orbits a
full 360 degrees around the Z axis over 240 frames (10 seconds at 24 fps),
rendered straight to an H.264 MP4.

Distance is measured from the building's outside face (its plan footprint),
not from its centre, so the camera keeps 50 ft of clearance at every angle.
At a corner view it sits 50 ft from the corner, and at a straight-on view
it sits 50 ft from the facade. Use --from-center to measure from the
centre point instead.

The orbit starts at the front (the -Y side, where the entry is) and turns
counter-clockwise seen from above. The camera is keyframed on every frame and
the last frame stops one step short of 360, so the video loops seamlessly.
The lens is fixed for the whole orbit (wide enough for the widest view), so
the building never appears to zoom in and out.

Run headless (recommended):
    blender -b t6_outlook.blend -P scripts/orbit_render.py -- --out ./renders/turntable.mp4

Or open the .blend in Blender, load this file in the Text Editor, and press
Run Script. The video then goes to "renders/turntable.mp4" next to the .blend
file. Blender is busy until all frames are rendered.

Options (everything after "--"):
    --out FILE          output video (default: //renders/turntable.mp4, next to the .blend)
    --distance-ft N     clearance from the building in feet (default 50)
    --frames N          frames for one full turn (default 240)
    --fps N             frames per second (default 24)
    --height-ft N       camera height above grade in feet
                        (default: half the building height, keeps verticals straight)
    --lens-mm N         fixed focal length; by default the lens is fitted so the
                        whole building is in frame
    --res WxH           output resolution, even numbers (default 3840x2160, 4K UHD)
    --samples N         override render samples (EEVEE and Cycles)
    --engine NAME       override render engine, e.g. CYCLES (default: keep the file's)
    --from-center       measure the distance from the building centre, not its facade
    --keep-camera       leave the animated orbit camera in the scene afterwards

The script does not save the .blend file. The scene's active camera, frame
range and render settings are restored when it finishes.
"""

import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

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
    p.add_argument("--out", default="//renders/turntable.mp4")
    p.add_argument("--distance-ft", type=float, default=50.0)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--height-ft", type=float, default=None)
    p.add_argument("--lens-mm", type=float, default=None)
    p.add_argument("--res", default="3840x2160")
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


def frame_extent(cam_matrix, corners, aspect):
    """How wide the view must be (tan of half the horizontal field of view)
    to fit every bounding-box corner in frame."""
    inv = cam_matrix.inverted()
    need = 0.0
    for c in corners:
        p = inv @ c  # camera looks down its local -Z
        depth = -p.z
        if depth <= 1e-6:
            continue
        need = max(need, abs(p.x) / depth, abs(p.y) / depth * aspect)
    return need * FRAME_MARGIN


def set_video_output(render, path, fps):
    """H.264 in an MP4 container."""
    fmt = render.image_settings
    if hasattr(fmt, "media_type"):  # Blender 5.0+: video formats sit behind media_type
        fmt.media_type = "VIDEO"
    fmt.file_format = "FFMPEG"
    fmt.color_mode = "RGB"
    ff = render.ffmpeg
    ff.format = "MPEG4"
    ff.codec = "H264"
    ff.constant_rate_factor = "HIGH"
    ff.ffmpeg_preset = "GOOD"
    ff.gopsize = fps  # a keyframe every second keeps scrubbing responsive
    ff.audio_codec = "NONE"
    render.fps = fps
    render.fps_base = 1.0
    render.use_file_extension = False  # write exactly the --out path
    render.filepath = path


def main():
    args = parse_args()
    scene = bpy.context.scene
    render = scene.render

    res_x, res_y = (int(v) for v in args.res.lower().split("x"))
    if res_x % 2 or res_y % 2:
        raise ValueError("H.264 needs an even width and height, got " + args.res)
    distance = feet_to_bu(scene, args.distance_ft)
    out_path = bpy.path.abspath(args.out)
    if not out_path.lower().endswith(".mp4"):
        out_path += ".mp4"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

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
    fmt = render.image_settings
    ff = render.ffmpeg
    prefs = bpy.context.preferences.edit
    saved = {
        "camera": scene.camera,
        "res": (render.resolution_x, render.resolution_y, render.resolution_percentage),
        "frames": (scene.frame_start, scene.frame_end, scene.frame_step, scene.frame_current),
        "fps": (render.fps, render.fps_base),
        "filepath": (render.filepath, render.use_file_extension),
        "media_type": getattr(fmt, "media_type", None),
        "format": (fmt.file_format, fmt.color_mode, fmt.color_depth),
        "ffmpeg": (ff.format, ff.codec, ff.constant_rate_factor, ff.ffmpeg_preset,
                   ff.gopsize, ff.audio_codec),
        "interpolation": prefs.keyframe_new_interpolation_type,
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
        set_video_output(render, out_path, args.fps)
        scene.frame_start, scene.frame_end, scene.frame_step = 1, args.frames, 1
        if args.engine:
            render.engine = args.engine
        if args.samples:
            if hasattr(scene.eevee, "taa_render_samples"):
                scene.eevee.taa_render_samples = args.samples
            if hasattr(scene, "cycles"):
                scene.cycles.samples = args.samples

        # Work out the camera for every frame first: position on the 50 ft
        # path, aimed at the building's centre.
        aspect = res_x / res_y
        poses = []
        prev_euler = None
        for i in range(args.frames):
            angle = 360.0 * i / args.frames  # last frame stops one step short: seamless loop
            x, y = orbit_position(angle, center, lo, hi, distance, args.from_center)
            if distance_to_footprint(x, y, lo, hi) == 0.0:
                print(f"Warning: at {angle:.0f} deg the camera is inside the building footprint.")
            loc = Vector((x, y, cam_z))
            quat = (target - loc).to_track_quat("-Z", "Y")
            # Keep each rotation continuous with the last, so the camera
            # doesn't spin the long way round when the angle wraps past 180.
            euler = quat.to_euler("XYZ", prev_euler) if prev_euler else quat.to_euler("XYZ")
            prev_euler = euler
            poses.append((loc, euler, Matrix.LocRotScale(loc, quat, None)))

        # One lens for the whole orbit, wide enough for the widest view.
        if args.lens_mm:
            cam_data.lens = args.lens_mm
        else:
            need = max(frame_extent(m, corners, aspect) for _, _, m in poses)
            cam_data.lens = cam_data.sensor_width / (2.0 * need)

        # A key on every frame, linear in between: exact distance at every frame.
        prefs.keyframe_new_interpolation_type = "LINEAR"
        for frame, (loc, euler, _) in enumerate(poses, start=scene.frame_start):
            cam_obj.location = loc
            cam_obj.rotation_euler = euler
            cam_obj.keyframe_insert("location", frame=frame)
            cam_obj.keyframe_insert("rotation_euler", frame=frame)

        print(f"Rendering {args.frames} frames at {args.fps} fps "
              f"({args.frames / args.fps:.1f} s), lens {cam_data.lens:.1f} mm -> {out_path}")
        bpy.ops.render.render(animation=True)
    finally:
        scene.camera = saved["camera"]
        render.resolution_x, render.resolution_y, render.resolution_percentage = saved["res"]
        scene.frame_start, scene.frame_end, scene.frame_step, frame_current = saved["frames"]
        scene.frame_set(frame_current)
        render.fps, render.fps_base = saved["fps"]
        render.filepath, render.use_file_extension = saved["filepath"]
        if saved["media_type"] is not None:
            fmt.media_type = saved["media_type"]
        fmt.file_format, fmt.color_mode, fmt.color_depth = saved["format"]
        (ff.format, ff.codec, ff.constant_rate_factor, ff.ffmpeg_preset,
         ff.gopsize, ff.audio_codec) = saved["ffmpeg"]
        prefs.keyframe_new_interpolation_type = saved["interpolation"]
        render.engine = saved["engine"]
        if saved["eevee_samples"] is not None:
            scene.eevee.taa_render_samples = saved["eevee_samples"]
        if saved["cycles_samples"] is not None:
            scene.cycles.samples = saved["cycles_samples"]
        if not args.keep_camera:
            action = cam_obj.animation_data.action if cam_obj.animation_data else None
            bpy.data.objects.remove(cam_obj)
            bpy.data.cameras.remove(cam_data)
            if action is not None and action.users == 0:
                bpy.data.actions.remove(action)

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
