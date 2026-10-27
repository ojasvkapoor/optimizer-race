from manim import *
import numpy as np
from optimizers import f, run, B, START, STEPS as N, CONFIG

# Render with:  manim -qm scene.py OptimizerRace   (720p30; use -ql for a fast preview)

Text.set_default(font="DejaVu Sans")

COLORS = {"sgd": "#E8E8E8", "mom": "#FFB000", "rms": "#2EC4B6", "adam": "#FF4D6D"}
CFG = [  # label, key, learning rate, colour
    (label, key, lr, COLORS[key]) for key, (label, lr) in CONFIG.items()
]
PATHS = [run(k, lr, N, START) for _, k, lr, _ in CFG]
LOSS = [np.array([f(p) for p in P]) for P in PATHS]
LOGL = [np.clip(np.log10(L), -6, 1) for L in LOSS]
HIT = [int(np.argmax(L < 1e-3)) if (L < 1e-3).any() else None for L in LOSS]

H = np.array([[2 + 8 * B, -4 * B], [-4 * B, 2 * B]])
EVAL, EVEC = np.linalg.eigh(H)
LMIN, LMAX = EVAL
KAPPA = LMAX / LMIN


def at(arr, t):
    i = int(np.floor(t))
    if i >= len(arr) - 1:
        return arr[-1]
    fr = t - i
    return arr[i] * (1 - fr) + arr[i + 1] * fr


class OptimizerRace(Scene):
    def construct(self):
        # ---------- title ----------
        title = Text("Optimizer Race", font_size=64, weight=BOLD)
        sub = Text("How AI learns: four algorithms descend the same loss landscape",
                   font_size=26, color=GRAY_B).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(title, shift=UP * 0.3), run_time=1.2)
        self.play(FadeIn(sub), run_time=0.8)
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(sub))

        # ---------- landscape ----------
        ax = Axes(
            x_range=[-2, 2, 1], y_range=[-1, 3, 1], x_length=6.0, y_length=6.0, tips=False,
            x_axis_config={"numbers_to_include": [-2, -1, 0, 1, 2], "font_size": 20},
            y_axis_config={"numbers_to_include": [-1, 0, 1, 2, 3], "font_size": 20},
        ).move_to(LEFT * 3.75 + DOWN * 0.1)
        levels = [0.1, 0.4, 1, 2, 4, 8, 16, 32, 64]
        c_hi, c_lo = ManimColor("#9D8CFF"), ManimColor("#2A2F6B")
        contours = VGroup(*[
            ax.plot_implicit_curve(lambda x, y, lv=lv: f((x, y)) - lv,
                                   color=interpolate_color(c_hi, c_lo, i / (len(levels) - 1)),
                                   stroke_width=2)
            for i, lv in enumerate(levels)
        ])
        self.play(Create(ax), run_time=1.0)
        self.play(LaggedStart(*[Create(c) for c in contours], lag_ratio=0.15), run_time=3.0)

        formula = MathTex(r"f(x,y)=(1-x)^2+5\,(y-x^2)^2", font_size=34).move_to(RIGHT * 3.9 + UP * 3.45)
        self.play(Write(formula))

        minimum = Dot(ax.c2p(1, 1), radius=0.09, color=YELLOW)
        min_lbl = Text("minimum (1, 1)", font_size=18, color=YELLOW).next_to(minimum, RIGHT, buff=0.3).shift(UP * 0.18)
        start_dot = Dot(ax.c2p(*START), radius=0.09, color=WHITE)
        start_lbl = Text("start (-1.5, 2)", font_size=18).next_to(start_dot, UP, buff=0.1)
        self.play(FadeIn(minimum), FadeIn(min_lbl), FadeIn(start_dot), FadeIn(start_lbl))

        # ---------- Hessian insight ----------
        d_flat = EVEC[:, 0]
        d_steep = EVEC[:, 1]
        def dir_line(d, col):
            a = np.array([1, 1]) - 0.55 * d
            b = np.array([1, 1]) + 0.55 * d
            return DashedLine(ax.c2p(*a), ax.c2p(*b), color=col, stroke_width=3, dash_length=0.1)
        l_flat = dir_line(d_flat, GREEN_B)
        l_steep = dir_line(d_steep, RED_B)
        def end_label(line, txt, col):
            p0, p1 = line.get_start(), line.get_end()
            top = p0 if p0[1] > p1[1] else p1
            other = p1 if top is p0 else p0
            direction = (top - other) / np.linalg.norm(top - other)
            return Text(txt, font_size=18, color=col).next_to(top, direction, buff=0.1)
        t_flat = end_label(l_flat, "flat", GREEN_B)
        t_steep = end_label(l_steep, "steep", RED_B)
        hess = MathTex(r"H(1,1)=\begin{pmatrix}42&-20\\-20&10\end{pmatrix}", font_size=34)
        eig = MathTex(rf"\lambda_{{\max}}\approx {LMAX:.1f},\quad \lambda_{{\min}}\approx {LMIN:.2f}", font_size=32)
        kap = MathTex(rf"\kappa=\lambda_{{\max}}/\lambda_{{\min}}\approx {KAPPA:.0f}", font_size=36, color=YELLOW)
        cap = MathTex(rf"\text{{SGD is stable only for }}\eta<2/\lambda_{{\max}}\approx {2/LMAX:.3f}", font_size=28)
        cap2 = VGroup(
            Text("at η = 0.03 the flat direction shrinks by only", font_size=21, color=GRAY_B),
            MathTex(rf"1-\eta\lambda_{{\min}}\approx {1-0.03*LMIN:.3f}\ \text{{ per step}}", font_size=30),
        ).arrange(DOWN, buff=0.15)
        block = VGroup(hess, eig, kap, cap, cap2).arrange(DOWN, buff=0.38).move_to(RIGHT * 3.9 + DOWN * 0.1)
        self.play(Create(l_flat), Create(l_steep), FadeIn(t_flat), FadeIn(t_steep))
        for m in (hess, eig, kap):
            self.play(Write(m), run_time=1.2)
            self.wait(0.4)
        self.play(Write(cap), run_time=1.2)
        self.play(FadeIn(cap2))
        self.wait(2.0)
        self.play(FadeOut(block), FadeOut(l_flat), FadeOut(l_steep), FadeOut(t_flat), FadeOut(t_steep))

        # ---------- update rules ----------
        def row(name, color, lines):
            head = Text(name, font_size=24, color=color, weight=BOLD)
            body = VGroup(*[MathTex(s, font_size=26) for s in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            return VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        rules = VGroup(
            row("SGD", CFG[0][3], [r"\theta\leftarrow\theta-\eta\,g"]),
            row("Momentum", CFG[1][3], [r"v\leftarrow\beta v+g", r"\theta\leftarrow\theta-\eta\,v"]),
            row("RMSProp", CFG[2][3], [r"s\leftarrow\rho s+(1-\rho)g^2",
                                       r"\theta\leftarrow\theta-\eta\,g/(\sqrt{s}+\epsilon)"]),
            row("Adam", CFG[3][3], [r"m\leftarrow\beta_1 m+(1-\beta_1)g,\;\; s\leftarrow\beta_2 s+(1-\beta_2)g^2",
                                    r"\theta\leftarrow\theta-\eta\,\hat m/(\sqrt{\hat s}+\epsilon)"]),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        rules.scale_to_fit_width(min(rules.width, 6.3)).move_to(RIGHT * 3.9 + DOWN * 0.15)
        g_lbl = MathTex(r"g=\nabla f(\theta)", font_size=26, color=GRAY_B).next_to(formula, DOWN, buff=0.2)
        self.play(FadeIn(g_lbl))
        for r in rules:
            self.play(FadeIn(r, shift=LEFT * 0.3), run_time=0.8)
            self.wait(0.9)
        self.wait(0.8)
        self.play(FadeOut(rules), FadeOut(g_lbl))

        # ---------- chart + scoreboard ----------
        t = ValueTracker(0)
        ax2 = Axes(
            x_range=[0, N, 50], y_range=[0, 7, 1], x_length=5.6, y_length=2.4, tips=False,
            x_axis_config={"numbers_to_include": [0, 50, 100, 150], "font_size": 18},
            y_axis_config={"include_numbers": False},
        ).move_to(RIGHT * 4.1 + UP * 1.25)
        y_labs = VGroup(*[
            MathTex(rf"10^{{{k}}}", font_size=20).next_to(ax2.c2p(0, k + 6), LEFT, buff=0.12) for k in (-6, -4, -2, 0)
        ])
        thr = DashedLine(ax2.c2p(0, 3), ax2.c2p(N, 3), color=GRAY, stroke_width=1.5, dash_length=0.08)
        thr_lbl = Text("target 0.001", font_size=15, color=GRAY_B).next_to(ax2.c2p(N, 3), UP, buff=0.05).shift(LEFT * 0.7)
        ch_title = Text("loss (log scale) vs step", font_size=18, color=GRAY_B).next_to(ax2, UP, buff=0.12).align_to(ax2, LEFT)
        self.play(Create(ax2), FadeIn(y_labs), Create(thr), FadeIn(thr_lbl), FadeIn(ch_title))

        hdr = Text("steps to reach loss below 0.001", font_size=18, color=GRAY_B).move_to(RIGHT * 4.0 + DOWN * 0.45)
        rows = VGroup()
        for j, (name, _, lr, col) in enumerate(CFG):
            y = -1.05 - 0.55 * j
            nm = Text(name, font_size=21, color=col, weight=BOLD).move_to(RIGHT * 0.8 + UP * y, aligned_edge=LEFT)
            lrt = MathTex(rf"\eta={lr}", font_size=24, color=col).move_to(RIGHT * 3.0 + UP * y, aligned_edge=LEFT)
            rows.add(nm, lrt)
        self.play(FadeIn(hdr), FadeIn(rows))

        def stat(k):
            tt = t.get_value()
            if HIT[k] is not None and tt >= HIT[k]:
                s, col, w = f"{HIT[k]} steps", CFG[k][3], BOLD
            elif tt >= N - 1e-6:
                s, col, w = "not reached", "#9AA0A6", NORMAL
            else:
                s, col, w = f"loss {LOSS[k][min(int(tt), N)]:.1e}", "#9AA0A6", NORMAL
            return Text(s, font_size=21, color=col, weight=w).move_to(RIGHT * 4.6 + UP * (-1.05 - 0.55 * k), aligned_edge=LEFT)
        stats = [always_redraw(lambda k=k: stat(k)) for k in range(4)]

        # trails, curves, dots
        def trail(k, axes, getter):
            col = CFG[k][3]
            m = VMobject(stroke_color=col, stroke_width=3.2, stroke_opacity=0.95)
            def upd(mob):
                tt = t.get_value()
                n = int(np.floor(tt))
                pts = [getter(k, i) for i in range(min(n, N) + 1)] + [getter(k, tt)]
                mob.set_points_as_corners([axes.c2p(*p) for p in pts])
            m.add_updater(upd)
            return m
        get_path = lambda k, i: at(PATHS[k], i)
        def get_loss_pt(k, i):
            return (float(i), float(at(LOGL[k], i)) + 6.0)
        trails = [trail(k, ax, get_path) for k in range(4)]
        curves = [trail(k, ax2, get_loss_pt) for k in range(4)]
        dots = []
        for k in range(4):
            d = Dot(radius=0.1, color=CFG[k][3]).move_to(ax.c2p(*START))
            d.add_updater(lambda mob, k=k: mob.move_to(ax.c2p(*at(PATHS[k], t.get_value()))))
            dots.append(d)
        counter = always_redraw(lambda: Text(f"step {int(t.get_value())}", font_size=26, weight=BOLD)
                                .move_to(ax.get_corner(UL) + UP * 0.38 + RIGHT * 0.85))
        self.play(FadeOut(start_lbl))
        self.add(*trails, *curves, *stats, counter, *dots)
        go = Text("GO", font_size=72, weight=BOLD, color=YELLOW).move_to(ax.get_center())
        self.play(FadeIn(go, scale=1.4), run_time=0.4)
        self.play(FadeOut(go, scale=0.8), run_time=0.4)

        # ---------- race ----------
        self.play(t.animate.set_value(N), run_time=22, rate_func=linear)
        self.wait(1.0)

        # ---------- takeaway ----------
        foot = Text("Same start · same 150-step budget · default learning rates · nothing hand-tuned",
                    font_size=20, color=GRAY_B).move_to(DOWN * 3.72)
        self.play(FadeIn(foot))
        self.wait(2.5)
        take = Text("No universal winner: on this landscape, momentum and Adam reach 0.001 first.",
                    font_size=22, color=YELLOW).move_to(DOWN * 3.72)
        self.play(FadeOut(foot), FadeIn(take))
        self.wait(4.0)
