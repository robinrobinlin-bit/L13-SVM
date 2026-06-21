from manim import *

class SVMMarginScene(Scene):
    def construct(self):
        # Axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=6,
            y_length=6,
        )
        self.play(Create(axes))

        # Data points: red class and blue class
        red_points = [(-3, -2), (-2, -3), (-4, -1), (-3, -4)]
        blue_points = [(3, 2), (2, 3), (4, 1), (3, 4)]
        red_dots = VGroup(*[Dot(axes.coords_to_point(*p), color=RED) for p in red_points])
        blue_dots = VGroup(*[Dot(axes.coords_to_point(*p), color=BLUE) for p in blue_points])
        self.play(FadeIn(red_dots), FadeIn(blue_dots))

        # Candidate separating lines (different slopes)
        cand_lines = []
        for m in [-1, -0.5, 0, 0.5, 1]:
            line = axes.plot(lambda x, m=m: m * x, color=GRAY, x_range=[-5, 5])
            cand_lines.append(line)
        candidates = VGroup(*cand_lines)
        self.play(FadeIn(candidates), run_time=2)

        # Highlight the maximum‑margin line (example slope 0.5)
        margin_line = axes.plot(lambda x: 0.5 * x, color=YELLOW)
        self.play(Transform(candidates, margin_line), run_time=2)

        # Support vectors (highlight the nearest points)
        sv_red = Dot(axes.coords_to_point(-3, -2), color=YELLOW, radius=0.15)
        sv_blue = Dot(axes.coords_to_point(3, 2), color=YELLOW, radius=0.15)
        self.play(FadeIn(sv_red), FadeIn(sv_blue))

        self.wait(2)
