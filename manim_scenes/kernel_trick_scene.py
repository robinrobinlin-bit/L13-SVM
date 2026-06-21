from manim import *

class KernelTrickScene(ThreeDScene):
    def construct(self):
        # 2D original data (non-linear, e.g., two moons)
        moon1 = VGroup(*[Dot(np.array([np.cos(t), np.sin(t)+0.2, 0]), color=BLUE, radius=0.08) for t in np.linspace(0, PI, 12)])
        moon2 = VGroup(*[Dot(np.array([1.5+np.cos(t), -np.sin(t)-0.2, 0]), color=RED, radius=0.08) for t in np.linspace(0, PI, 12)])
        # Show 2D axes
        axes2d = Axes(x_range=[-1, 3, 1], y_range=[-1, 2, 1], x_length=6, y_length=4)
        self.play(Create(axes2d), FadeIn(moon1), FadeIn(moon2))
        self.wait(1)

        # Transform to 3D (feature mapping)
        self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES)
        axes3d = ThreeDAxes(x_range=[-1, 3, 1], y_range=[-1, 2, 1], z_range=[-1, 3, 1])
        self.play(Transform(axes2d, axes3d))
        # Lift points into 3D (simple quadratic mapping: (x, y) -> (x, y, x**2 + y**2))
        mapped_points = []
        for dot in moon1 + moon2:
            x, y, _ = dot.get_center()
            z = x**2 + y**2
            new_dot = Dot3D(np.array([x, y, z]), color=dot.get_color(), radius=0.08)
            mapped_points.append(new_dot)
        mapped = VGroup(*mapped_points)
        self.play(FadeOut(moon1), FadeOut(moon2), FadeIn(mapped))
        self.wait(1)

        # Show separating plane in 3D
        plane = Surface(
            lambda u, v: np.array([u, v, 0]),
            u_range=[-1, 3],
            v_range=[-1, 2],
            fill_opacity=0.2,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        self.play(Create(plane))
        self.wait(1)

        # Rotate back to 2D view to illustrate non-linear boundary
        self.move_camera(phi=0 * DEGREES, theta=0 * DEGREES)
        self.play(FadeOut(axes3d), FadeOut(mapped), FadeOut(plane))
        self.wait(1)
        # Show original 2D moons with a curved decision boundary (approximated by a line of dots)
        boundary = VGroup(*[Dot(np.array([np.cos(t), np.sin(t), 0]), color=YELLOW, radius=0.06) for t in np.linspace(0, 2*PI, 30)])
        self.play(FadeIn(boundary))
        self.wait(2)
        # Add explanatory text
        txt = Text("Kernel trick maps data to higher‑dimensional space where it becomes linearly separable", font_size=24).to_edge(DOWN)
        self.play(FadeIn(txt))
        self.wait(2)
