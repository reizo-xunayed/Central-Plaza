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

For the render the building gets extra light: a soft area light facing each
facade, so whichever side is towards the camera is lit (the shaded side no
longer goes dark), plus a sun from the front-left if the scene has none
(neither a sun lamp nor a Sky Texture with a sun disc). The facade lights are
hidden from the camera and from reflections. Their brightness follows the
sun's, so they lift the shadows without washing out the sunlit sides.

Glass is made opaque for the render, so the inside of the building can't be
seen through windows and curtain walls. A material counts as glass if its
name contains "glass", "glazing" or "glazed", or if its shader lets light
through (Glass or Refraction BSDF, Principled BSDF with transmission, or a
fixed alpha below 1). Each one is swapped for a tinted, reflective glass that
shows the sky and surroundings instead of the interior. The list of swapped
materials is printed at the start of the render.

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
    --light-strength N  multiplier for the added lights (default 1.0; 2 is twice as bright)
    --no-lights         don't add any lights, render with the scene's own lighting
    --keep-glass        leave the glass as modelled (the interior shows through)

The script does not save the .blend file. The scene's active camera, frame
range and render settings are restored when it finishes, the added lights are
removed and the original glass materials are put back.
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

# Materials whose names contain any of these are treated as glass.
GLASS_KEYWORDS = ("glass", "glazing", "glazed")

LIGHT_NAME = "Orbit Light"
SUN_STRENGTH = 3.0  # W/m^2, a clear-day sun when the scene has none of its own
SUN_ELEVATION = 45.0  # degrees above the horizon
SUN_AZIMUTH = -35.0  # degrees from the front, negative is towards the left side
FILL_RATIO = 0.35  # light from each facade light, as a fraction of the sun's


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
    p.add_argument("--light-strength", type=float, default=1.0)
    p.add_argument("--no-lights", action="store_true")
    p.add_argument("--keep-glass", action="store_true")
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


def is_glass_material(mat):
    """Glass by name, or by a shader that lets light (and the view) through."""
    if any(k in mat.name.lower() for k in GLASS_KEYWORDS):
        return True
    if mat.node_tree is None:
        return False
    for node in mat.node_tree.nodes:
        if node.type in ("BSDF_GLASS", "BSDF_REFRACTION"):
            return True
        if node.type == "BSDF_PRINCIPLED":
            # "Transmission Weight" since Blender 4.0, "Transmission" before.
            trans = node.inputs.get("Transmission Weight") or node.inputs.get("Transmission")
            if trans and not trans.is_linked and trans.default_value >= 0.5:
                return True
            # A fixed alpha below 1 is see-through glazing; a linked alpha is a
            # cut-out (leaves, mesh screens) and is left alone.
            alpha = node.inputs.get("Alpha")
            if alpha and not alpha.is_linked and alpha.default_value < 0.95:
                return True
    return False


def make_opaque_glass(name):
    """Tinted, reflective glass that hides whatever is behind it."""
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:  # before Blender 5.0, new materials have no nodes
        mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    settings = {
        "Base Color": (0.10, 0.14, 0.18, 1.0),  # dark blue-grey tint
        "Metallic": 0.6,  # strong reflections of the sky and surroundings
        "Roughness": 0.05,
        "Alpha": 1.0,
        "Transmission Weight": 0.0,
        "Transmission": 0.0,
    }
    for key, value in settings.items():
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = value
    mat.diffuse_color = settings["Base Color"]  # viewport colour
    return mat


def make_glass_opaque(scene, swaps):
    """Swap every glass material used in the scene for an opaque copy.
    Appends (original, replacement) to `swaps` so restore_glass can undo it."""
    glass = {
        slot.material
        for obj in scene.objects
        for slot in getattr(obj, "material_slots", ())
        if slot.material is not None and is_glass_material(slot.material)
    }
    for mat in sorted(glass, key=lambda m: m.name):
        replacement = make_opaque_glass(mat.name + " (opaque)")
        swaps.append((mat, replacement))
        mat.user_remap(replacement)
    if glass:
        print("Glass made opaque: " + ", ".join(m.name for m, _ in swaps))
    else:
        print("Warning: no glass materials found (looked for names containing {} and "
              "see-through shaders); the inside may still show.".format(" / ".join(GLASS_KEYWORDS)))


def restore_glass(swaps):
    for mat, replacement in swaps:
        replacement.user_remap(mat)
        bpy.data.materials.remove(replacement)


def view_factor(width, height, d):
    """Share of a width x height panel's light (per unit of its exitance) that
    reaches a point d in front of its centre: E = (P / area) * view_factor."""
    def quarter(a, b):
        x, y = a / d, b / d
        sx, sy = math.sqrt(1 + x * x), math.sqrt(1 + y * y)
        return (x / sx * math.atan(y / sx) + y / sy * math.atan(x / sy)) / (2 * math.pi)
    return 4 * quarter(width / 2, height / 2)


def add_light(scene, name, kind, added):
    data = bpy.data.lights.new(name, kind)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    added.append(obj)
    return obj


def add_lights(scene, lo, hi, strength, added):
    """A sun (only if the scene has none) and a soft area light facing each
    facade, aimed at the building. Appends the new light objects to `added`."""
    suns = [
        o.data.energy for o in scene.objects
        if o.type == "LIGHT" and o.data.type == "SUN" and not o.hide_render
    ]
    world = scene.world
    sky_has_sun = world is not None and world.node_tree is not None and any(
        n.type == "TEX_SKY" and n.sky_type not in ("PREETHAM", "HOSEK_WILKIE") and n.sun_disc
        for n in world.node_tree.nodes
    )
    if suns:
        sun_strength = max(suns)
        print(f"Using the scene's sun ({sun_strength:g} W/m^2) as the key light")
    elif sky_has_sun:
        # A second sun would cast a second set of shadows.
        sun_strength = SUN_STRENGTH
        print("Using the sun in the world's Sky Texture as the key light")
    else:
        sun_strength = SUN_STRENGTH
        sun = add_light(scene, LIGHT_NAME + " Sun", "SUN", added)
        sun.data.energy = sun_strength * strength
        sun.data.angle = math.radians(2.0)  # slightly soft shadow edges
        el, az = math.radians(SUN_ELEVATION), math.radians(SUN_AZIMUTH)
        # Same angle convention as the orbit: 0 is the front (-Y).
        towards_sun = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
        sun.rotation_euler = (-towards_sun).to_track_quat("-Z", "Y").to_euler()
        sun.location = (lo + hi) / 2 + towards_sun * (hi - lo).length

    # One panel per side, as wide and tall as that facade, set back half the
    # building's diagonal so the light is soft and even across the face.
    size = hi - lo
    gap = size.length / 2
    mid_z = (lo.z + hi.z) / 2
    fill_irradiance = FILL_RATIO * sun_strength * strength
    sides = (
        ("Front", Vector((0, -1, 0)), Vector(((lo.x + hi.x) / 2, lo.y, mid_z)), size.x),
        ("Right", Vector((1, 0, 0)), Vector((hi.x, (lo.y + hi.y) / 2, mid_z)), size.y),
        ("Back", Vector((0, 1, 0)), Vector(((lo.x + hi.x) / 2, hi.y, mid_z)), size.x),
        ("Left", Vector((-1, 0, 0)), Vector((lo.x, (lo.y + hi.y) / 2, mid_z)), size.y),
    )
    for side, normal, face_centre, width in sides:
        if width <= 0 or size.z <= 0:
            continue  # flat bounding box: no facade on this side to light
        fill = add_light(scene, f"{LIGHT_NAME} {side}", "AREA", added)
        fill.data.shape = "RECTANGLE"
        fill.data.size, fill.data.size_y = width, size.z
        area = width * size.z
        fill.data.energy = fill_irradiance * area / view_factor(width, size.z, gap)
        fill.data.specular_factor = 0.0  # no bright panels mirrored in the glass
        fill.location = face_centre + normal * gap
        fill.rotation_euler = (-normal).to_track_quat("-Z", "Y").to_euler()
        fill.visible_camera = False
        fill.visible_glossy = False


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

    glass_swaps = []
    added_lights = []
    try:
        if not args.keep_glass:
            make_glass_opaque(scene, glass_swaps)
        if not args.no_lights:
            add_lights(scene, lo, hi, args.light_strength, added_lights)

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
        restore_glass(glass_swaps)
        for light in added_lights:
            data = light.data
            bpy.data.objects.remove(light)
            bpy.data.lights.remove(data)
        if not args.keep_camera:
            action = cam_obj.animation_data.action if cam_obj.animation_data else None
            bpy.data.objects.remove(cam_obj)
            bpy.data.cameras.remove(cam_data)
            if action is not None and action.users == 0:
                bpy.data.actions.remove(action)

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
