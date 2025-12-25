using Pkg
Pkg.activate(@__DIR__)

using Luxor
using Colors

# --- Configuration ---
WIDTH, HEIGHT = 800, 800
R_TRIANGLE = 180      # Distance from center to the 3 main nodes
R_INNER_DOT = 45      # Radius of the solid center dots
R_MID_THICKNESS = 35  # Thickness of the surrounding rings
GAP = 15              # The whitespace gap between rings

# --- Helper Function: Create a "Pinched" Rounded Triangle ---
# This creates the flowing curves connecting the corners
function path_pinched_triangle(radius_at_corner, pinch_factor)
    # 3 points of the triangle (shifted -90 deg so point 1 is Top)
    pts = [polar(R_TRIANGLE, -π / 2 + (i - 1) * 2π / 3) for i in 1:3]

    # Move to the start of the first arc
    # We start slightly "past" the top to ensure the curve connects smoothly
    start_angle = -π / 2

    move(pts[1] + polar(radius_at_corner, start_angle + deg2rad(60)))

    for i in 1:3
        current_p = pts[i]
        next_p = pts[mod1(i + 1, 3)]

        # Calculate angles relative to the center of the current corner
        # The logic ensures the arc covers the outer corner
        angle_start = -π / 2 + (i - 1) * 2π / 3 + deg2rad(60)
        angle_end = -π / 2 + (i - 1) * 2π / 3 + deg2rad(240) # Wrap around the corner

        # Draw the rounded corner
        arc(current_p, radius_at_corner, angle_start, angle_end)

        # Draw the connecting curve that "pinches" inward
        # We use the midpoint between corners, pulled towards center (0,0)
        mid_p = midpoint(current_p, next_p)
        pinch_p = midpoint(mid_p, Point(0, 0)) * pinch_factor

        # Bezier curve to the next corner's start point
        # Next start point calculation
        next_start =
            next_p +
            polar(radius_at_corner, -π / 2 + (mod1(i + 1, 3) - 1) * 2π / 3 + deg2rad(60))

        curve(
            current_p + polar(radius_at_corner, angle_end) + (next_p - current_p) * 0.2, # Control 1
            next_start - (next_p - current_p) * 0.2 + (Point(0, 0) - mid_p) * 0.3,             # Control 2
            next_start,
        )
    end
    return closepath()
end

# --- Main Drawing Function ---
function draw_logo(filename)
    Drawing(WIDTH, HEIGHT, filename)
    origin()
    background("white")

    # 1. Define the Gradient Mesh
    # Create a triangular mesh matching the logo's triangular shape
    triangle_pts = [
        Point(0, -300),      # Top - Red
        Point(300, 200),     # Bottom Right - Purple
        Point(-300, 200)     # Bottom Left - Green
    ]
    g_mesh = mesh(
        triangle_pts,
        [colorant"#D94646", colorant"#8E44AD", colorant"#4DAF7C"]
    )

    # --- Construct the Mask ---
    # We will add all shapes to the current path, then Clip, then Paint.

    # Layer 1: The Inner Solid Circles
    for i in 1:3
        p = polar(R_TRIANGLE, -π / 2 + (i - 1) * 2π / 3)
        circle(p, R_INNER_DOT, :path)
        newsubpath() # Important to separate shapes
    end

    # Layer 2: The Middle Ring (Donut)
    # To make a ring, we define the Outer path clockwise and Inner path counter-clockwise
    # But here, we can just treat them as separate positive shapes for the mask

    # We construct this by creating a "thick" pinched triangle shape
    # Actually, simpler method: Draw the shape, stroke it?
    # No, to use a mesh gradient, we need a fillable path (region).

    # Let's draw the Middle Ring using our helper function
    # Inner edge of middle ring
    path_pinched_triangle(R_INNER_DOT + GAP, 0.75)
    # Outer edge of middle ring (we reverse it to create a hole if we were filling,
    # but for clipping a union works differently. Let's make a custom "Ring Path")

    # Alternative Strategy:
    # Since complex path boolean ops are hard, let's just add ALL positive areas
    # to the clipping path.

    # 2. Add Middle Ring Path (The "Tube")
    # We simulate the tube by drawing a very thick line converted to a path?
    # Better: explicitly define the outer and inner boundaries of the tube.

    # Outer Boundary of Middle Ring
    path_pinched_triangle(R_INNER_DOT + GAP + R_MID_THICKNESS, 0.65)
    newsubpath()

    # 3. Add Outer Shell Path
    # Outer Boundary
    path_pinched_triangle(R_INNER_DOT + GAP + R_MID_THICKNESS + GAP + 20, 0.55)
    newsubpath()

    # --- The Arrow (Top Right) ---
    # The arrow breaks out of the purple ring.
    # We'll add a polygonal arrow shape to the mask.
    # Approximate location
    gsave()
    translate(180, -100)
    rotate(-π / 4)
    arrow_head = [Point(0, 0), Point(-40, 20), Point(-30, 0), Point(-40, -20)]
    poly(arrow_head, :path; close=true)
    grestore()

    # --- Rendering ---
    # Now that we have defined all "Solid" areas in the path...
    clip() # Restrict all future drawing to these shapes

    # Paint the gradient mesh over the whole canvas
    # The clip ensures it only appears on our logo shapes
    setmesh(g_mesh)
    box(BoundingBox(), :fill)

    finish()
    return preview()
end

draw_logo("luxor_triskelion.svg")
