# Doctor Strange Hand-Tracking Portal

A learning project that builds a real-time, Doctor Strange-inspired webcam
effect in small, testable stages. Stages 13 and 14 add smooth portal growth and
a one-shot magical energy explosion when two active hands merge.

## Current progress

- [x] Stage 1: Python environment, webcam capture, mirrored display, FPS counter
- [x] Stage 2: MediaPipe hand tracking and landmark visualization
- [x] Stage 3: Palm-center calculation and temporary verification markers
- [x] Stage 4: Simple palm-following magical rings
- [x] Stage 5: Time-based rotating ring detail
- [x] Stage 6: Subtle Gaussian-blurred magical glow
- [x] Stage 7: Procedural golden dust and spark particles
- [x] Stage 8: Open-palm gesture activation
- [x] Stage 9: Razor-light arcs and enhanced magical particles
- [x] Stage 10: Gesture charging and activation state machine
- [x] Stage 11: Screenshot and MP4 recording controls
- [x] Stage 12: Two-hand distance-controlled portal
- [x] Stage 13: Time-based portal growth animation
- [x] Stage 14: Re-armable two-hand merge energy burst

## Requirements

- Windows with a working webcam
- Python 3.12
- Camera permission enabled for desktop applications

The tested environment uses Python 3.12.10, NumPy 2.5.4, OpenCV Contrib
5.0.0.93, and MediaPipe 1.2.0. OpenCV Contrib is a superset of the base OpenCV
package and satisfies MediaPipe's dependency metadata. Do not install
`opencv-python` beside it because both packages provide the same `cv2` module.

The pretrained model is stored at `models/hand_landmarker.task`. It contains
Google's palm detector and 21-point hand-landmark model.

## Setup

From PowerShell in this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, activation is optional:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

The image is mirrored by default. MediaPipe detects and draws up to two hands,
with points at 21 anatomical landmarks and lines showing their connections. A
cyan-and-white bullseye marks each estimated palm center. Thin broken
golden-orange energy arcs and a soft halo follow that point.
Eight inner ticks rotate using elapsed time rather than frame count. Short-lived
golden dust and occasional faster sparks drift outward from each ring. Open at
least three fingers and hold for 0.8 seconds to activate an effect; close the
hand for 0.3 seconds to deactivate it.
Thin incomplete arcs rotate in opposite directions to break up the former
heavy circular outline.

Controls and options:

- Press `Q` or `Esc` to close the application.
- Press `S` to save the current annotated frame as a PNG screenshot.
- Press `R` to start recording; press `R` again to finish the MP4 file.
- Use `--camera-index 1` if the desired webcam is not camera `0`.
- Use `--width 640 --height 480` for a lower-resolution, faster feed.
- Use `--no-mirror` to display the camera's original orientation.
- Use `--model-path PATH` to select another compatible model.
- Use `--captures-dir PATH` to change where screenshots and recordings go.

By default, captures are stored in the project `captures` directory with
timestamped names such as `portal_20261029_123456_123456.png`. A red `REC`
indicator is included while recording. Closing the application safely releases
an active video writer.

See every command-line option:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

## Checks

Run checks that do not require a webcam:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q main.py src tests
.\.venv\Scripts\python.exe -m pip check
```

For a short camera and model check without a window:

```powershell
.\.venv\Scripts\python.exe main.py --headless --max-frames 30
```

## Stage 2 coordinates and hand layout

MediaPipe reports image landmarks with **normalized coordinates**. Instead of
returning a specific pixel, it returns decimal fractions. `x = 0` means the
left edge, `x = 1` means the right edge, `y = 0` means the top, and `y = 1`
means the bottom. This lets the same model work with any image resolution.

Conversion to image pixels is simple multiplication:

```text
pixel_x = normalized_x * (image_width - 1)
pixel_y = normalized_y * (image_height - 1)
```

For a 640-by-480 image, `(0.5, 0.5)` is approximately the center at
`(320, 240)`. Coordinates are clamped to the image bounds before drawing.

Each detected hand has 21 ordered points:

- `0`: wrist
- `1-4`: thumb, from its base to its tip
- `5-8`: index finger, from knuckle to tip
- `9-12`: middle finger
- `13-16`: ring finger
- `17-20`: little finger

Lines connect neighboring points to form a simple hand skeleton. MediaPipe also
returns `z`, an estimated depth value, but Stage 2 only draws `x` and `y`.

## Stage 3 palm-center mathematics

The palm estimate uses landmark `0` at the wrist and landmarks `5`, `9`, `13`,
and `17` at the four finger-base knuckles. These five points outline the broad
palm area and are less affected by finger movement than the fingertips.

The normalized center is the arithmetic mean of their coordinates:

```text
center_x = (x0 + x5 + x9 + x13 + x17) / 5
center_y = (y0 + y5 + y9 + y13 + y17) / 5
```

Each hand is calculated independently. A hand is skipped if any required
landmark or coordinate is missing or invalid. The center then passes through
the same safe normalized-to-pixel conversion used for the landmarks. Because
the camera frame is mirrored before detection, no second mirroring step is
needed.

## Stage 4 ring mathematics

Each ring reuses the normalized palm center from Stage 3 and converts it with
the existing safe pixel helper. Its default radius is 13% of the frame's shorter
dimension:

```text
radius = round(min(frame_width, frame_height) * 0.13)
```

At 1280-by-720 this is about 94 pixels; at 640-by-480 it is about 62 pixels.
Near an image edge, the radius shrinks to the available space instead of moving
the ring away from the real palm center. A non-positive radius is skipped.

OpenCV uses blue-green-red (BGR) color order. The main ring color
`(0, 180, 255)` therefore has no blue, medium green, and full red, producing a
golden orange. A wider dark-orange outline underneath improves contrast. A
second circle at 72% of the outer radius adds layered geometry.

For two hands, the same calculation and drawing steps run once for each valid
palm center. Ring size currently follows frame size, not physical hand size.

## Stage 5 rotation mathematics

The camera pipeline already supplies elapsed milliseconds from Python's
monotonic performance clock. A default speed of 90 degrees per second converts
that elapsed time into a repeating angle:

```text
elapsed_seconds = elapsed_ms / 1000
angle = (elapsed_seconds * 90) modulo 360
```

For example, after 0.5 seconds the angle is 45 degrees, and after 1 second it
is 90 degrees. After 4 seconds it wraps from 360 back to 0. Because the angle
comes from time, a slower computer may draw fewer intermediate frames but the
rotation speed remains approximately 90 degrees per real second.

Eight short ticks are separated by `360 / 8 = 45` degrees. For each tick, its
angle is converted from degrees to radians, then its endpoints are calculated:

```text
x = center_x + radius * cos(angle)
y = center_y + radius * sin(angle)
```

Cosine controls horizontal displacement and sine controls vertical
displacement. Image `y` coordinates increase downward, so increasing angles
appear clockwise. Both hands share the same time angle but use their own palm
center and boundary-aware radius; no per-hand animation state is required.

## Stage 6 glow and image blending

The glow uses one black overlay with the same width, height, channels, and
`uint8` data type as the webcam frame. Thick orange copies of both rings and
rotating ticks are drawn on that overlay.

Gaussian blur replaces each pixel with a weighted average of nearby pixels.
Bright strokes therefore spread outward into a soft halo instead of remaining
hard lines. The blurred overlay is blended into the camera image with:

```text
output = 1.00 * camera_frame + 0.45 * blurred_glow
```

The 0.45 glow weight keeps the effect visible without strongly washing out the
webcam image. The sharp landmarks, rings, rotating ticks, and cyan center marker
are drawn after this blend, so blur never softens those foreground details.

Both hands draw onto the same overlay, which is blurred only once per frame.
`HandTracker` also clears and reuses that overlay while frame dimensions and
data type remain unchanged. Empty or invalid hand results leave the camera
frame untouched.

## Stage 7 particle motion and lifetime

Each particle is a small Python object that keeps its own pixel position,
horizontal and vertical velocity, remaining and initial lifetime, size,
brightness, and whether it is a dust point or a faster spark. The same objects
are updated over many frames; the collection is not rebuilt from scratch.

Movement uses elapsed real time:

```text
new_x = old_x + velocity_x * elapsed_seconds
new_y = old_y + velocity_y * elapsed_seconds
opacity = remaining_lifetime / initial_lifetime
```

For example, a particle at `x = 100` with `velocity_x = 20 pixels/second`
moves to `x = 101` after `0.05` seconds. Its movement step is capped at 0.05
seconds after a delayed frame, preventing a large visual jump. Its lifetime
still uses the full real delay, so stale particles expire normally.

New particles start near the ring circumference. A random angle selects a
direction, then cosine and sine turn it into horizontal and vertical offsets:

```text
x = center_x + ring_radius * cos(angle)
y = center_y + ring_radius * sin(angle)
```

A radial velocity moves the particle outward. A perpendicular direction
`(-sin(angle), cos(angle))` adds a small sideways component, which makes the
dust drift around the ring instead of looking like a rigid explosion. Stage 9
makes roughly one particle in three a brighter, faster spark; the rest are
restrained dust points.

Particles add light to the existing reusable glow overlay before its single
Gaussian blur. Crisp dots and short streaks are then drawn on the camera frame,
followed by the sharp landmarks, rings, ticks, and cyan marker. A maximum of
160 live particles and a per-frame spawn limit bound the work. When a hand
vanishes, no new particles are created there, while its existing particles
fade naturally; the other hand continues independently.

## Stage 8 gesture activation

The gesture check ignores the thumb and examines the index, middle, ring, and
little fingers. For each finger, it compares two distances from landmark `0`
at the wrist:

```text
joint distance = distance(wrist, middle finger joint)
tip distance = distance(wrist, fingertip)
```

A finger counts as extended when its fingertip is at least 1.18 times farther
from the wrist than its middle joint. At least three extended fingers activate
that hand's effect. Squared distances are compared, so the code does not need
to calculate a square root for every finger on every frame.

Each detected hand is checked independently. Hand landmarks remain visible
even when its effect is inactive. Closing a hand stops its rings, ticks, glow,
and new particle emission; particles that already exist fade naturally.

### Using the gesture

1. Hold a hand where the webcam can see the wrist and fingers clearly.
2. Face the palm roughly toward the camera and spread at least three fingers.
3. Keep it open while `CHARGING` advances from 0% to 100% for 0.8 seconds.
4. The status changes to `ACTIVE`, and the magical effect appears.
5. Curl the fingers into a fist or leave fewer than three extended for 0.3
   seconds to turn the effect off.
6. Reopen during `HOLD OPEN` to cancel cooldown without charging again.
7. Repeat independently with the other hand.

## Stage 10 gesture state machine

Instant frame-by-frame activation can flicker when one landmark is briefly
misdetected. Stage 10 stores a small state for each Left/Right handedness label:

```text
INACTIVE -> CHARGING -> ACTIVE -> COOLDOWN -> INACTIVE
```

An open palm moves from `INACTIVE` to `CHARGING`. Continuous elapsed time must
reach 800 milliseconds before activation. Closing an active hand starts a
300-millisecond cooldown; reopening during that grace period returns directly
to `ACTIVE`. A hand that disappears for more than 700 milliseconds loses its
stored track and must charge again when it returns.

Charging progress is based on time, not processed-frame count:

```text
progress = charging_elapsed_ms / 800
```

At 400 milliseconds, progress is 0.5 or 50%, regardless of whether the camera
processed 8 frames or 24 frames. MediaPipe's handedness label associates state
with each hand instead of treating result-list positions as permanent IDs.

## Stage 9 energy polish

Complete sharp circles were removed. Six incomplete outer arcs rotate
clockwise while five inner arcs rotate in the opposite direction, all with
narrow white-gold energy cores. Tiny stationary accents provide structure
without rebuilding a thick round tube.

Particle emission increases from 28 to 60 particles per second for each active
palm, with a bounded maximum of 260 live particles. Dust lives longer and uses
brighter amber colors. Roughly 36% become fast white-gold sparks with a wide
orange trail, thin bright core, and glowing head.

Each particle also receives a small changing sideways force:

```text
force_x = cos(time * speed + random_phase) * turbulence_strength
force_y = sin(time * speed + random_phase) * turbulence_strength
velocity = velocity + force * elapsed_seconds
```

The random phase gives different particles different paths, while elapsed time
keeps the motion independent of frame count. Ring, arc, and particle light all
share the existing reusable glow overlay, so the application still performs
only one full-frame Gaussian blur per processed frame.

## Stage 11 screenshots and recordings

`CaptureManager` owns screenshot and video output separately from the camera
and hand tracker. `S` calls OpenCV's PNG writer for one fully annotated frame.
The first `R` creates an MP4 writer using the displayed frame dimensions; the
second `R` releases it. The camera loop also closes the writer in its `finally`
block, so `Q`, `Esc`, or an error cannot intentionally leave it open.

Recording uses a nominal 30 FPS because webcam backends do not always report a
stable capture rate. A slower computer may therefore produce video whose
playback timing differs slightly from the live session.

## Stage 12 two-hand portal mathematics

Both hands must first reach the Stage 10 `ACTIVE` state. Their pixel centers
define the portal midpoint:

```text
portal_x = (left_x + right_x) / 2
portal_y = (left_y + right_y) / 2
```

The distance between palms controls portal size:

```text
distance = sqrt((right_x - left_x)^2 + (right_y - left_y)^2)
portal_radius = distance * 0.48
```

The radius is limited to 30% of the frame's shorter dimension and reduced near
image edges. Hands closer than 80 pixels do not create a portal. Moving active
hands apart enlarges it; moving them together shrinks and eventually removes
it. Three broken arc layers counter-rotate at the midpoint, use the existing
shared glow overlay, and add one bounded portal particle emitter.

## Stage 13 portal growth

When two active hands first form a valid portal, it grows from zero to its
distance-controlled target radius over 600 milliseconds. A cubic ease-out
function makes growth fast initially and gentle near full size:

```text
linear = elapsed_ms / 600
growth_scale = 1 - (1 - linear)^3
displayed_radius = target_radius * growth_scale
```

At halfway in time, `linear = 0.5`, but the eased scale is `0.875`. The portal
therefore responds quickly rather than feeling delayed, while still finishing
smoothly. Losing either active hand resets the growth animation.

## Stage 14 two-hand merge burst

The explosion is a fictional procedural energy burst. To use it:

1. Charge both hands until both are `ACTIVE`.
2. Separate their palm centers by at least 140 pixels. This arms the merge.
3. Move the palms together until their centers are within 70 pixels.
4. A 750-millisecond shockwave appears at the midpoint with a bright core,
   sixteen radial rays, an expanding ring, and up to 72 radial sparks.
5. Separate the hands to 140 pixels again before attempting another burst.

The armed/re-arm rule prevents a burst from firing continuously while hands
remain together. Explosion particles share the existing maximum particle
budget: dimmer old particles may be replaced so the burst remains visible,
but the total collection never grows without limit.

## FPS mathematics

If two frames arrive `dt` seconds apart, the instantaneous rate is:

```text
FPS = 1 / dt
```

For example, `dt = 0.05` seconds produces `20 FPS`. The displayed value uses an
exponentially weighted average to avoid distracting jumps:

```text
smoothed = 0.9 * previous + 0.1 * newest
```

Stage 5 uses angles, sine, and cosine for its rotating tick positions.

## Known limitations

- Requested camera resolution is a preference; webcam drivers may select a
  nearby supported resolution.
- Detection can become less stable with poor lighting, motion blur, occlusion,
  or hands that are very small in the image.
- MediaPipe's first startup can take longer while native components initialize.
- The palm center is a simple geometric estimate and can move slightly as
  landmark predictions change; temporal smoothing is not implemented yet.
- The bullseye is a temporary verification marker, not a portal effect.
- The larger ring radius scales with frame size, so it does not grow when a hand moves
  closer to the camera or shrink when a hand moves farther away.
- Rings shrink near frame edges to remain inside the image where practical.
- The tick rotation is time-based, but its perceived smoothness still depends
  on the achieved FPS and stability of the detected palm center.
- Both hands currently share the same rotation phase and speed while remaining
  independently positioned.
- Particle positions use camera pixels, so a very fast palm can briefly leave
  a short dust trail; short lifetimes keep that trail restrained.
- Particle emission is deliberately subtle and its visual balance depends on
  camera resolution, lighting, and achieved FPS.
- Gesture classification uses simple landmark geometry rather than a trained
  gesture model. Sideways hands, partial occlusion, and motion blur can cause
  brief activation changes.
- Handedness labels are normally stable, but crossing hands or severe
  occlusion can briefly confuse which stored gesture state belongs to a hand.
- Video uses a fixed nominal 30 FPS; actual camera FPS may differ.
- The two-hand portal is procedural arc energy, not a replacement background
  or composited view into another scene.
- Merge thresholds use screen pixels, so the physical hand spacing required
  changes with camera resolution and distance from the webcam.
- No elaborate runes or portal opening exists yet.
