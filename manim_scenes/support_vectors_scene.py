from manim import *

class SupportVectorsScene(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-5,5,1],
            y_range=[-5,5,1],
            x_length=6,
            y_length=6,
        )
        self.play(Create(axes))

        # Points
        red_points = [(-3, -2), (-2, -3), (-4, -1), (-3, -4)]
        blue_points = [(3, 2), (2, 3), (4, 1), (3, 4)]
        reds = VGroup(*[Dot(axes.coords_to_point(*p), color=RED) for p in red_points])
        blues = VGroup(*[Dot(axes.coords_to_point(*p), color=BLUE) for p in blue_points])
        self.play(FadeIn(reds), FadeIn(blues))

        # Hard margin line (tight)
        hard_line = axes.plot(lambda x: 0.5 * x, color=YELLOW)
        self.play(Create(hard_line))
        self.wait(0.5)

        # Highlight support vectors
        sv_red = Dot(axes.coords_to_point(-3, -2), color=YELLOW, radius=0.15)
        sv_blue = Dot(axes.coords_to_point(3, 2), color=YELLOW, radius=0.15)
        self.play(FadeIn(sv_red), FadeIn(sv_blue))

        # Soft margin: show slack with a dashed line
        soft_line = DashedLine(
            start=axes.coords_to_point(-5, -2.5),
            end=axes.coords_to_point(5, 2.5),
            color=ORANGE,
        )
        self.play(Create(soft_line))

        # Text explaining C
        txt = Text("Large C => Hard margin, Small C => Soft margin", font_size=24).to_edge(DOWN)
        self.play(FadeIn(txt))
        self.wait(2)
