"""
NPU Synchronization Flow Animation (Manim Community Edition)
=============================================================
PPT 'com,hw) sync 설계 제안 검토' 기반 12개 Scene.

Usage:
    manim -pql npu_sync_animation.py Scene01_HardwareOverview
    manim -pql npu_sync_animation.py Scene02_Initialization
    ...
    manim -pql npu_sync_animation.py Scene12_SyncConfusion

Dependencies:
    pip install manim
"""

from manim import *

# ── Color Palette ──
SM_COLOR = BLUE
UB_COLOR = GREEN
UB_A_COLOR = "#4CAF50"
UB_B_COLOR = "#2E7D32"
COMPUTE_COLOR = RED
DMA_COLOR = YELLOW
FETCH_COLOR = PURPLE
QUEUE_COLOR = TEAL
CORE_COLOR = ORANGE
RUNTIME_COLOR = "#E91E63"
REG_COLOR = GOLD
SEM_COLOR = "#FF9800"


def make_block(label, width=1.8, height=0.8, color=WHITE, font_size=20):
    rect = RoundedRectangle(
        corner_radius=0.12, width=width, height=height,
        color=color, fill_opacity=0.2, stroke_width=2
    )
    txt = Text(label, font_size=font_size, color=color)
    txt.move_to(rect.get_center())
    return VGroup(rect, txt)


def phase_title(text, scene_num):
    t = Text(f"[{scene_num}/12] {text}", font_size=26, color=WHITE)
    t.to_edge(UP, buff=0.25)
    line = Line(t.get_left() + DOWN * 0.15, t.get_right() + DOWN * 0.15,
                color=GREY_B, stroke_width=1)
    return VGroup(t, line)


def desc_text(line1, line2=""):
    full = line1 if not line2 else f"{line1}\n{line2}"
    t = Text(full, font_size=15, color=GREY_A)
    t.to_edge(DOWN, buff=0.2)
    return t


def make_queue_block(x, y, label="queue", scale=1.0):
    """queue 바이너리 구조 시각화"""
    entries = ["stop", "layer_start", "layer_run", "layer_run",
               "layer_end", "...", "epoch_commit", "loop_jump"]
    grp = VGroup()
    for i, e in enumerate(entries):
        r = Rectangle(width=1.2 * scale, height=0.22 * scale,
                       color=QUEUE_COLOR, fill_opacity=0.1, stroke_width=1)
        r.move_to([x, y - i * 0.22 * scale, 0])
        t = Text(e, font_size=int(9 * scale), color=QUEUE_COLOR)
        t.move_to(r)
        grp.add(VGroup(r, t))
    lbl = Text(label, font_size=int(12 * scale), color=QUEUE_COLOR)
    lbl.next_to(grp, UP, buff=0.05)
    return VGroup(lbl, grp)


# ======================================================================
# Scene 1: Hardware Overview (Slide 1)
# ======================================================================
class Scene01_HardwareOverview(Scene):
    def construct(self):
        title = phase_title("하드웨어 전체 구조", 1)
        self.play(FadeIn(title))

        # Runtime
        runtime = make_block("Runtime", 1.5, 0.7, RUNTIME_COLOR, 18)
        runtime.move_to(LEFT * 5.5 + UP * 1.5)

        # SM
        sm_box = RoundedRectangle(corner_radius=0.15, width=3.5, height=4.0,
                                   color=SM_COLOR, fill_opacity=0.08, stroke_width=2)
        sm_box.move_to(LEFT * 2.5 + DOWN * 0.3)
        sm_lbl = Text("SM (Shared Memory)", font_size=16, color=SM_COLOR)
        sm_lbl.next_to(sm_box, UP, buff=0.05)

        # SM 내부 영역
        out_buf = make_block("Output\nBuffer", 1.4, 0.6, "#e74c3c", 12)
        out_buf.move_to(sm_box.get_center() + UP * 1.2)
        in_buf = make_block("Input\nBuffer", 1.4, 0.6, "#3498db", 12)
        in_buf.move_to(sm_box.get_center() + UP * 0.3)
        temp_buf = make_block("Temp Buffer", 1.4, 0.6, GREY_B, 12)
        temp_buf.move_to(sm_box.get_center() + DOWN * 0.6)
        meta = make_block("meta data\n(weight)", 1.4, 0.5, SEM_COLOR, 11)
        meta.move_to(sm_box.get_center() + DOWN * 1.4)

        sm_group = VGroup(sm_box, sm_lbl, out_buf, in_buf, temp_buf, meta)

        # Core pairs (p core + A core + UB + queue) x 2
        def make_core_unit(x_pos, label_p="p core", label_a="A core"):
            p = make_block(label_p, 1.0, 0.5, CORE_COLOR, 14)
            a = make_block(label_a, 1.0, 0.5, COMPUTE_COLOR, 14)
            ub = make_block("UB", 1.0, 0.5, UB_COLOR, 14)
            q = make_block("queue", 1.0, 0.4, QUEUE_COLOR, 12)
            p.move_to([x_pos, 1.5, 0])
            a.move_to([x_pos + 1.3, 1.5, 0])
            ub.move_to([x_pos + 0.65, 0.7, 0])
            q.move_to([x_pos + 0.65, 0.0, 0])
            return VGroup(p, a, ub, q)

        core1 = make_core_unit(1.5)
        core2 = make_core_unit(4.0)

        self.play(FadeIn(runtime))
        self.play(FadeIn(sm_group))
        self.play(FadeIn(core1), FadeIn(core2))

        # Arrows Runtime → SM
        arr_rt = Arrow(runtime.get_right(), sm_box.get_left() + UP * 1.5,
                       color=RUNTIME_COLOR, stroke_width=2, buff=0.1)
        self.play(GrowArrow(arr_rt))

        # Arrows SM ↔ Cores
        for cu in [core1, core2]:
            arr = Arrow(sm_box.get_right(), cu[2].get_left(),
                        color=SM_COLOR, stroke_width=2, buff=0.1)
            self.play(GrowArrow(arr), run_time=0.5)

        d = desc_text(
            "Runtime이 SM을 통해 데이터를 관리하고,",
            "각 코어 쌍(p core + A core)이 UB와 queue를 가집니다."
        )
        self.play(FadeIn(d))
        self.wait(3)


# ======================================================================
# Scene 2: Initialization (Slide 2)
# ======================================================================
class Scene02_Initialization(Scene):
    def construct(self):
        title = phase_title("1. 초기화", 2)
        self.play(FadeIn(title))

        # SM
        sm = make_block("SM", 3.0, 2.5, SM_COLOR, 20)
        sm.move_to(LEFT * 3.5 + UP * 0.3)
        meta = make_block("meta data\n(weight)", 2.0, 0.6, SEM_COLOR, 14)
        meta.move_to(sm.get_center() + DOWN * 0.5)

        # Acore
        acore = make_block("A core", 1.5, 0.8, COMPUTE_COLOR, 18)
        acore.move_to(RIGHT * 0.5 + UP * 1.8)

        # Queue structure
        queue = make_queue_block(3.5, 1.5, "queue", 1.2)

        self.play(FadeIn(sm), FadeIn(meta))
        self.play(FadeIn(acore), FadeIn(queue))

        # Step 1: weight → A core
        step1 = Text("① A Core에 weight eFlash", font_size=16, color=CORE_COLOR)
        step1.move_to(LEFT * 1 + DOWN * 1.5)
        arr1 = Arrow(sm.get_right() + UP * 0.3, acore.get_left(),
                     color=CORE_COLOR, stroke_width=2, buff=0.1)
        self.play(GrowArrow(arr1), FadeIn(step1))

        # Step 2: meta data → SM
        step2 = Text("② SM에 meta data load (weight 포함)", font_size=16, color=SM_COLOR)
        step2.move_to(LEFT * 1 + DOWN * 2.0)
        self.play(FadeIn(step2))

        # Step 3: queue bin load
        step3 = Text("③ queue에 queue bin load", font_size=16, color=QUEUE_COLOR)
        step3.move_to(LEFT * 1 + DOWN * 2.5)
        self.play(FadeIn(step3))

        # Step 4: pc → stop
        step4 = Text("④ pc는 stop을 가리킴 (첫 명령어 = stop)", font_size=16, color=RED)
        step4.move_to(LEFT * 1 + DOWN * 3.0)
        # Highlight stop in queue
        stop_highlight = SurroundingRectangle(queue[1][0], color=RED, buff=0.03)
        self.play(FadeIn(step4), Create(stop_highlight))

        d = desc_text(
            "weight 주소/크기, meta data 주소/크기는 컴파일러가 미리 결정.",
            "binary 첫 명령어가 stop임을 컴파일러가 보장."
        )
        self.play(FadeIn(d))
        self.wait(3)


# ======================================================================
# Scene 3: NPU Start (Slide 3)
# ======================================================================
class Scene03_NPUStart(Scene):
    def construct(self):
        title = phase_title("2. NPU 시작", 3)
        self.play(FadeIn(title))

        # Thread 1: Start thread
        t1_box = RoundedRectangle(corner_radius=0.1, width=3.5, height=2.0,
                                   color=RUNTIME_COLOR, fill_opacity=0.1, stroke_width=2)
        t1_box.move_to(LEFT * 3.5 + UP * 0.5)
        t1_title = Text("Thread 1: Start", font_size=16, color=RUNTIME_COLOR)
        t1_title.next_to(t1_box, UP, buff=0.05)
        t1_code = Text("while {\n  pre-process\n  register check\n  Input Write\n}",
                        font_size=12, color=WHITE)
        t1_code.move_to(t1_box)

        # Thread 2: Complete thread
        t2_box = RoundedRectangle(corner_radius=0.1, width=3.5, height=2.0,
                                   color="#2196F3", fill_opacity=0.1, stroke_width=2)
        t2_box.move_to(RIGHT * 3.5 + UP * 0.5)
        t2_title = Text("Thread 2: Complete", font_size=16, color="#2196F3")
        t2_title.next_to(t2_box, UP, buff=0.05)
        t2_code = Text("while {\n  register check\n  post-process\n}",
                        font_size=12, color=WHITE)
        t2_code.move_to(t2_box)

        self.play(
            FadeIn(t1_box), FadeIn(t1_title), FadeIn(t1_code),
            FadeIn(t2_box), FadeIn(t2_title), FadeIn(t2_code),
        )

        # Registers
        regs = VGroup()
        reg_names = ["Input Buffer Enable", "Output Buffer Enable",
                     "Stop", "Loop Counter", "Queue Complete Counter"]
        for i, name in enumerate(reg_names):
            r = make_block(f"Reg: {name}", 3.0, 0.35, REG_COLOR, 11)
            r.move_to(DOWN * (1.2 + i * 0.45))
            regs.add(r)

        self.play(FadeIn(regs), run_time=1)

        # Steps
        steps = VGroup(
            Text("① Input Buffer Enable Reg = 1 → Input 쓰기", font_size=13, color=RUNTIME_COLOR),
            Text("② Stop Reg = 1 → 0으로 만들고 pc를 loop_start_pc로 이동", font_size=13, color=RED),
            Text("③ Output Buffer Enable Reg = 1 → 후처리 진행", font_size=13, color="#2196F3"),
        )
        steps.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        steps.move_to(RIGHT * 0 + DOWN * 3.5)
        self.play(FadeIn(steps))

        # SemaPhore Entry label
        sem = make_block("SemaPhore Entry", 2.5, 0.4, SEM_COLOR, 13)
        sem.move_to(LEFT * 0 + UP * 2.2)
        self.play(FadeIn(sem))

        s_detail = Text(
            "Input SM load 시 SemaPhore Entry 세팅,\n"
            "read_reference_counter는 컴파일타임 결정",
            font_size=12, color=SEM_COLOR
        )
        s_detail.next_to(sem, RIGHT, buff=0.2)
        self.play(FadeIn(s_detail))

        self.wait(3)


# ======================================================================
# Scene 4: Layer Start + 제안1 (Slides 4, 5)
# ======================================================================
class Scene04_LayerStart(Scene):
    def construct(self):
        title = phase_title("3. Layer Start + SM Semaphore Entry 제안", 4)
        self.play(FadeIn(title))

        # ── 제안1: Semaphore Entry 구조 ──
        struct_title = Text("SM Semaphore Entry 구조 (제안)", font_size=20, color=SEM_COLOR)
        struct_title.move_to(UP * 2.0)
        self.play(FadeIn(struct_title))

        # 기존 필드 (회색, 취소선 효과)
        old_fields = VGroup()
        old_names = ["write_ptr\n(32b)", "read_ptr\n(32b)", "write_valid\n_size(16b)", "read_valid\n_size(16b)"]
        for i, name in enumerate(old_names):
            r = Rectangle(width=1.8, height=0.7, color=GREY_B, fill_opacity=0.05, stroke_width=1)
            r.shift(LEFT * 3.6 + RIGHT * i * 1.85 + UP * 1.0)
            t = Text(name, font_size=10, color=GREY_B)
            t.move_to(r)
            cross = Line(r.get_corner(UL), r.get_corner(DR), color=RED, stroke_width=1.5)
            old_fields.add(VGroup(r, t, cross))

        self.play(FadeIn(old_fields))
        old_label = Text("기존 (96bit) → 불필요 필드 제거", font_size=13, color=GREY_A)
        old_label.next_to(old_fields, RIGHT, buff=0.2)
        self.play(FadeIn(old_label))

        # 새 필드
        new_fields = VGroup()
        new_data = [
            ("data_position\n(32bit)", SM_COLOR, 2.5),
            ("write_ref_cnt\n(4bit)", "#2ecc71", 1.8),
            ("read_ref_cnt\n(4bit)", "#e74c3c", 1.8),
            ("semaphore_id\n(8bit)", SEM_COLOR, 1.8),
        ]
        cumx = -3.8
        for name, col, w in new_data:
            r = Rectangle(width=w, height=0.7, color=col, fill_opacity=0.15, stroke_width=2)
            r.move_to([cumx + w / 2, -0.2, 0])
            t = Text(name, font_size=11, color=col)
            t.move_to(r)
            new_fields.add(VGroup(r, t))
            cumx += w + 0.05

        self.play(FadeIn(new_fields))

        # fan-out / fan-in 설명
        fanout = Text(
            "write_ref_cnt → fan-in 해결 (여러 코어가 하나의 데이터 제작)",
            font_size=13, color="#2ecc71"
        )
        fanin = Text(
            "read_ref_cnt → fan-out 해결 (하나의 데이터를 여러 코어가 참조)",
            font_size=13, color="#e74c3c"
        )
        fanout.move_to(DOWN * 1.0)
        fanin.move_to(DOWN * 1.4)
        self.play(FadeIn(fanout), FadeIn(fanin))
        self.wait(2)

        # ── Layer Start 동작 ──
        self.play(*[FadeOut(m) for m in self.mobjects])
        title2 = phase_title("3. Layer Start - 데이터 가져오기", 4)
        self.play(FadeIn(title2))

        sm = make_block("SM", 2.0, 1.5, SM_COLOR, 20)
        sm.move_to(LEFT * 4 + UP * 0.5)
        sem_entry = make_block("SemaPhore\nEntry", 1.5, 0.6, SEM_COLOR, 13)
        sem_entry.move_to(LEFT * 4 + DOWN * 1.0)

        core = make_block("Core\n(Layer Start)", 2.0, 1.0, COMPUTE_COLOR, 16)
        core.move_to(RIGHT * 0 + UP * 0.5)

        ub = make_block("UB", 1.5, 1.0, UB_COLOR, 20)
        ub.move_to(RIGHT * 3.5 + UP * 0.5)

        self.play(FadeIn(sm), FadeIn(sem_entry), FadeIn(core), FadeIn(ub))

        # Step 1: SM 읽기 → SemaPhore 확인
        s1 = Text("① SM 읽기 시 SemaPhore Entry 확인", font_size=14, color=SEM_COLOR)
        s1.move_to(DOWN * 1.8)
        arr_check = Arrow(core.get_left(), sem_entry.get_right(),
                          color=SEM_COLOR, stroke_width=2, buff=0.1)
        self.play(GrowArrow(arr_check), FadeIn(s1))

        s2 = Text("② write_ref_cnt = 0일 때만 읽기 가능", font_size=14, color="#2ecc71")
        s2.move_to(DOWN * 2.2)
        self.play(FadeIn(s2))

        s3 = Text("③ 읽기 완료 후 read_ref_cnt 1 감소", font_size=14, color="#e74c3c")
        s3.move_to(DOWN * 2.6)

        # Data flow
        arr_data = Arrow(sm.get_right(), core.get_left(),
                         color=SM_COLOR, stroke_width=3, buff=0.1)
        self.play(GrowArrow(arr_data), FadeIn(s3))

        s4 = Text("④ UB 전송 시 handshake (stream tx/rx)", font_size=14, color=UB_COLOR)
        s4.move_to(DOWN * 3.0)
        arr_ub = Arrow(core.get_right(), ub.get_left(),
                       color=UB_COLOR, stroke_width=3, buff=0.1)
        self.play(GrowArrow(arr_ub), FadeIn(s4))

        d = desc_text("SyncOp의 stream tx/rx로 전송 완료를 보장하여 RAW, WAR hazard 해결")
        self.play(FadeIn(d))
        self.wait(3)


# ======================================================================
# Scene 5: Layer Run + 제안2 (Slides 6, 7)
# ======================================================================
class Scene05_LayerRun(Scene):
    def construct(self):
        title = phase_title("4. Layer Run + Sync 제안", 5)
        self.play(FadeIn(title))

        # ── 제안2 설명 ──
        prop_title = Text("제안2: SM_WAIT_GE / SM_POST 제거", font_size=20, color=RED)
        prop_title.move_to(UP * 2.0)
        self.play(FadeIn(prop_title))

        old_box = make_block("기존: SM_WAIT_GE, SM_POST\n(1:1 semaphore)", 4.0, 0.8, GREY_B, 14)
        old_box.move_to(UP * 0.8)
        cross = Cross(old_box, stroke_color=RED, stroke_width=3)
        self.play(FadeIn(old_box), Create(cross))

        new_box = make_block("변경: reference_counter 기반\n상호 동작 (1:N 지원)", 4.0, 0.8, "#2ecc71", 14)
        new_box.move_to(DOWN * 0.3)
        self.play(FadeIn(new_box))

        # 규칙 목록
        rules = VGroup()
        rule_texts = [
            "읽기: semaphore Entry 없으면 오류",
            "쓰기: 반드시 semaphore Entry 설정",
            "읽기: write_ref_cnt = 0일 때만 가능",
            "읽기 완료: read_ref_cnt 1 감소",
            "쓰기 완료: write_ref_cnt 1 증가 (설정값 도달 시 0)",
            "쓰기 완료: read_ref_cnt를 ISA 값으로 세팅",
            "읽기: semaphore id 일치 필수",
            "쓰기 완료: semaphore id 세팅",
            "주소할당: 컴파일타임 정적할당 고정",
        ]
        for i, rt in enumerate(rule_texts):
            t = Text(f"{i+1}. {rt}", font_size=11, color=WHITE)
            rules.add(t)
        rules.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        rules.move_to(DOWN * 2.0)
        self.play(FadeIn(rules), run_time=1.5)

        self.wait(2)

        # ── Layer Run 동작 ──
        self.play(*[FadeOut(m) for m in self.mobjects])
        title2 = phase_title("4. Layer Run - 연산 수행", 5)
        self.play(FadeIn(title2))

        core = make_block("Core\n(Layer Run)", 2.5, 1.2, COMPUTE_COLOR, 18)
        core.move_to(ORIGIN)

        ub = make_block("UB", 2.0, 1.5, UB_COLOR, 22)
        ub.move_to(RIGHT * 4)

        sm = make_block("SM\n(meta + tensor)", 2.0, 1.0, SM_COLOR, 14)
        sm.move_to(LEFT * 4)

        self.play(FadeIn(core), FadeIn(ub), FadeIn(sm))

        # 연산
        glow = core[0].copy().set_fill(COMPUTE_COLOR, opacity=0.4)
        arr_in = Arrow(ub.get_left(), core.get_right(), color=UB_COLOR, stroke_width=3, buff=0.1)
        arr_out = Arrow(core.get_right(), ub.get_left(), color=DMA_COLOR, stroke_width=3, buff=0.1)
        arr_out.shift(DOWN * 0.3)
        arr_in.shift(UP * 0.3)

        self.play(GrowArrow(arr_in), GrowArrow(arr_out))

        for _ in range(3):
            self.play(FadeIn(glow), run_time=0.25)
            self.play(FadeOut(glow), run_time=0.25)

        d = desc_text(
            "Layer Run에서는 UB 밖으로 데이터 이동을 최소화.",
            "자세한 내부 동작은 하드웨어 설계에 따름."
        )
        self.play(FadeIn(d))
        self.wait(2)


# ======================================================================
# Scene 6: Layer End + Double Buffering (Slide 8)
# ======================================================================
class Scene06_LayerEnd(Scene):
    def construct(self):
        title = phase_title("5. Layer End + Double Buffering", 6)
        self.play(FadeIn(title))

        # Core
        core = make_block("Core\n(Layer End)", 2.0, 0.8, COMPUTE_COLOR, 16)
        core.move_to(LEFT * 1 + UP * 1.5)

        # SM
        sm = make_block("SM\n(Output 전송)", 2.0, 1.0, SM_COLOR, 14)
        sm.move_to(LEFT * 4.5 + UP * 1.5)

        # UB-A / UB-B
        ub_a = RoundedRectangle(corner_radius=0.1, width=2.0, height=1.2,
                                 color=UB_A_COLOR, fill_opacity=0.15, stroke_width=2)
        ub_a.move_to(RIGHT * 2.5 + UP * 1.8)
        ub_a_lbl = Text("UB-A", font_size=16, color=UB_A_COLOR)
        ub_a_lbl.move_to(ub_a)

        ub_b = RoundedRectangle(corner_radius=0.1, width=2.0, height=1.2,
                                 color=UB_B_COLOR, fill_opacity=0.15, stroke_width=2)
        ub_b.move_to(RIGHT * 2.5 + UP * 0.3)
        ub_b_lbl = Text("UB-B", font_size=16, color=UB_B_COLOR)
        ub_b_lbl.move_to(ub_b)

        self.play(FadeIn(core), FadeIn(sm), FadeIn(ub_a), FadeIn(ub_a_lbl),
                  FadeIn(ub_b), FadeIn(ub_b_lbl))

        # Step 1: broadcasting
        s1 = Text("① 먼저 끝난 코어부터 결과 데이터 broadcasting", font_size=14, color=DMA_COLOR)
        s1.move_to(DOWN * 0.5)
        arr_bc = Arrow(core.get_left(), sm.get_right(),
                       color=DMA_COLOR, stroke_width=3, buff=0.1)
        self.play(GrowArrow(arr_bc), FadeIn(s1))

        for _ in range(4):
            dot = Dot(color=DMA_COLOR, radius=0.06).move_to(core.get_left())
            self.play(dot.animate.move_to(sm.get_right()), run_time=0.3)
            self.remove(dot)

        # Step 2: double buffering
        s2 = Text("② Double Buffering: 다른 영역으로 다음 layer_start 진입",
                   font_size=14, color=UB_COLOR)
        s2.move_to(DOWN * 1.0)
        self.play(FadeIn(s2))

        # UB-A → 연산 완료 (lock)
        lock_a = Text("쓰기 완료\n(읽어가기 전 사용 불가)", font_size=10, color=RED)
        lock_a.next_to(ub_a, RIGHT, buff=0.1)

        # UB-B → 다음 작업 가능
        next_b = Text("다음 layer\n데이터 로드", font_size=10, color=UB_B_COLOR)
        next_b.next_to(ub_b, RIGHT, buff=0.1)

        self.play(FadeIn(lock_a))
        self.play(FadeIn(next_b))

        # 교대 표시
        swap_arrow = CurvedArrow(ub_a.get_bottom(), ub_b.get_top(),
                                  color=YELLOW, stroke_width=2)
        swap_txt = Text("번갈아 사용", font_size=12, color=YELLOW)
        swap_txt.next_to(swap_arrow, RIGHT, buff=0.05)
        self.play(Create(swap_arrow), FadeIn(swap_txt))

        d = desc_text(
            "UB 영역을 반으로 나누어 번갈아 사용하면,",
            "쓰기 완료 대기 없이 다음 레이어 작업을 진행할 수 있습니다."
        )
        self.play(FadeIn(d))
        self.wait(3)


# ======================================================================
# Scene 7: loop_jump or stop (Slide 9)
# ======================================================================
class Scene07_LoopJump(Scene):
    def construct(self):
        title = phase_title("6. loop_jump or stop", 7)
        self.play(FadeIn(title))

        # Queue
        queue = make_queue_block(0, 1.5, "queue", 1.5)
        self.play(FadeIn(queue))

        # PC indicator
        pc = Triangle(fill_color=RED, fill_opacity=0.8, color=RED)
        pc.scale(0.15)
        pc.next_to(queue[1][-2], LEFT, buff=0.1)  # epoch_commit
        pc_label = Text("PC", font_size=12, color=RED)
        pc_label.next_to(pc, LEFT, buff=0.05)
        self.play(FadeIn(pc), FadeIn(pc_label))

        # Registers
        reg_grp = VGroup()
        reg_data = [
            ("Queue Complete Counter", "0"),
            ("Loop Counter", "0"),
            ("Output Buffer Enable", "0"),
        ]
        for i, (name, val) in enumerate(reg_data):
            r = make_block(f"{name}: {val}", 4.0, 0.4, REG_COLOR, 13)
            r.move_to(RIGHT * 3.5 + UP * (1.5 - i * 0.6))
            reg_grp.add(r)
        self.play(FadeIn(reg_grp))

        # Step 1
        s1 = Text("① loop_end_pc 도착 → Queue Complete Counter + 1",
                   font_size=14, color=REG_COLOR)
        s1.move_to(DOWN * 0.8)
        self.play(FadeIn(s1))

        # Step 2
        s2 = Text("② layer_count와 비교:", font_size=14, color=WHITE)
        s2a = Text("  같거나 크면 → pc를 stop으로", font_size=13, color=RED)
        s2b = Text("  작으면 → pc를 loop_start_pc로", font_size=13, color=UB_COLOR)
        s2.move_to(DOWN * 1.3)
        s2a.move_to(DOWN * 1.7)
        s2b.move_to(DOWN * 2.1)
        self.play(FadeIn(s2), FadeIn(s2a), FadeIn(s2b))

        # Step 3
        s3 = Text(
            "③ Queue Complete Counter = 코어 수 →\n"
            "   0 초기화, Loop Counter + 1,\n"
            "   Output Buffer Enable = 1",
            font_size=13, color=RUNTIME_COLOR
        )
        s3.move_to(DOWN * 3.0)
        self.play(FadeIn(s3))

        # loop_jump 동작
        arr_loop = CurvedArrow(
            queue[1][-1].get_right(),  # loop_jump
            queue[1][0].get_right(),   # stop (= loop_start_pc)
            color=QUEUE_COLOR, stroke_width=2
        )
        loop_lbl = Text("loop_jump → loop_start_pc", font_size=11, color=QUEUE_COLOR)
        loop_lbl.next_to(arr_loop, RIGHT, buff=0.05)
        self.play(Create(arr_loop), FadeIn(loop_lbl))

        s4 = Text("다른 코어가 이전 추론 중이더라도 먼저 다음 추론 시작",
                   font_size=13, color=YELLOW)
        s4.move_to(RIGHT * 3.5 + DOWN * 0.5)
        self.play(FadeIn(s4))

        self.wait(3)


# ======================================================================
# Scene 8: epoch_commit (Slide 10)
# ======================================================================
class Scene08_EpochCommit(Scene):
    def construct(self):
        title = phase_title("7. epoch_commit (코어 간 동기화)", 8)
        self.play(FadeIn(title))

        # 문제 상황 (왼쪽)
        prob_title = Text("문제: 추론이 겹침", font_size=18, color=RED)
        prob_title.move_to(LEFT * 3.5 + UP * 1.8)
        self.play(FadeIn(prob_title))

        # 타임라인 - 문제
        bars_prob = VGroup()
        colors_p = [BLUE, GREEN, RED, YELLOW]
        labels_p = ["1", "2", "3", "4"]
        widths_p = [
            [(0, 2.5)],  # core fast
            [(0.5, 3.5)],  # core slow
        ]
        for i in range(2):
            bg = Rectangle(width=4.0, height=0.35, color=GREY_D, fill_opacity=0.15)
            bg.move_to(LEFT * 3.5 + UP * (0.8 - i * 0.6))
            lbl = Text(f"Core {i}", font_size=12, color=WHITE)
            lbl.next_to(bg, LEFT, buff=0.1)
            bars_prob.add(VGroup(bg, lbl))

        self.play(FadeIn(bars_prob))

        # 겹치는 영역 표시
        overlap = Rectangle(width=1.0, height=0.9, color=RED, fill_opacity=0.2, stroke_width=2)
        overlap.move_to(LEFT * 2.5 + UP * 0.5)
        overlap_lbl = Text("겹침!", font_size=12, color=RED)
        overlap_lbl.next_to(overlap, DOWN, buff=0.05)
        self.play(FadeIn(overlap), FadeIn(overlap_lbl))

        # 해결 (오른쪽)
        sol_title = Text("해결: epoch_commit", font_size=18, color="#2ecc71")
        sol_title.move_to(RIGHT * 3.5 + UP * 1.8)
        self.play(FadeIn(sol_title))

        # 타임라인 - 해결
        for i in range(2):
            bg = Rectangle(width=4.0, height=0.35, color=GREY_D, fill_opacity=0.15)
            bg.move_to(RIGHT * 3.5 + UP * (0.8 - i * 0.6))
            lbl = Text(f"Core {i}", font_size=12, color=WHITE)
            lbl.next_to(bg, LEFT, buff=0.1)
            self.play(FadeIn(VGroup(bg, lbl)), run_time=0.3)

        # epoch barrier
        barrier = DashedLine(
            RIGHT * 4.5 + UP * 1.2,
            RIGHT * 4.5 + DOWN * 0.1,
            color="#2ecc71", dash_length=0.1
        )
        b_lbl = Text("epoch\ncommit", font_size=10, color="#2ecc71")
        b_lbl.next_to(barrier, RIGHT, buff=0.05)
        self.play(Create(barrier), FadeIn(b_lbl))

        idle_lbl = Text("idle!", font_size=11, color=YELLOW)
        idle_lbl.move_to(RIGHT * 4 + UP * 0.8)
        self.play(FadeIn(idle_lbl))

        # 설명
        exp = VGroup(
            Text("• queue complete cnt = core 수 → 동기화 지점", font_size=13, color=WHITE),
            Text("• epoch_interval은 컴파일 옵션 (default = 1)", font_size=13, color=WHITE),
            Text("• 빠른 코어는 epoch_commit에서 대기", font_size=13, color=YELLOW),
        )
        exp.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        exp.move_to(DOWN * 1.5)
        self.play(FadeIn(exp))

        d = desc_text(
            "epoch_commit으로 일정 루프당 한 번씩 다른 코어의 연산을 기다려,",
            "추론 겹침을 방지합니다."
        )
        self.play(FadeIn(d))
        self.wait(3)


# ======================================================================
# Scene 9: Data Hazards (Slide 12)
# ======================================================================
class Scene09_DataHazards(Scene):
    def construct(self):
        title = phase_title("APPENDIX: Data Hazards", 9)
        self.play(FadeIn(title))

        hazards = [
            ("WAR", "다 쓰지 않았는데, 데이터를 읽는다", "write → read!", RED),
            ("RAW", "다 읽지 않았는데, 데이터를 쓴다", "read → write!", ORANGE),
            ("WAW", "다 쓰지 않았는데, 데이터를 쓴다", "write → write!", YELLOW),
        ]

        for i, (name, desc, flow, color) in enumerate(hazards):
            y = UP * (1.5 - i * 1.5)

            # Hazard name
            h_name = Text(name, font_size=28, color=color, weight=BOLD)
            h_name.move_to(LEFT * 4.5 + y)

            # Flow diagram
            box1 = make_block(flow.split("→")[0].strip(), 1.5, 0.5, WHITE, 14)
            box1.move_to(LEFT * 1.5 + y)
            arr = Arrow(LEFT * 0.5 + y, RIGHT * 0.5 + y, color=color, stroke_width=3)
            box2 = make_block(flow.split("→")[1].strip().rstrip("!"), 1.5, 0.5, color, 14)
            box2.move_to(RIGHT * 1.5 + y)
            danger = Text("!", font_size=24, color=color)
            danger.next_to(box2, RIGHT, buff=0.1)

            # Description
            h_desc = Text(desc, font_size=14, color=GREY_A)
            h_desc.move_to(RIGHT * 4 + y)

            self.play(
                FadeIn(h_name), FadeIn(box1), GrowArrow(arr),
                FadeIn(box2), FadeIn(danger), FadeIn(h_desc),
                run_time=0.8
            )

        # 해결
        sol = make_block("stream slot, sync slot으로 해결", 5.0, 0.5, "#2ecc71", 16)
        sol.move_to(DOWN * 2.5)
        self.play(FadeIn(sol))
        self.wait(3)


# ======================================================================
# Scene 10: Fan-out Problem (Slide 13)
# ======================================================================
class Scene10_FanOut(Scene):
    def construct(self):
        title = phase_title("APPENDIX: Fan-out 문제", 10)
        self.play(FadeIn(title))

        # SM
        sm = make_block("Shared Memory", 2.5, 1.0, SM_COLOR, 16)
        sm.move_to(LEFT * 3 + UP * 0.5)

        # 4 Cores
        cores = VGroup()
        for i in range(4):
            c = make_block(f"core {i+1}", 1.2, 0.5, COMPUTE_COLOR, 13)
            c.move_to(RIGHT * 2 + UP * (1.5 - i * 1.0))
            cores.add(c)

        self.play(FadeIn(sm), FadeIn(cores))

        # 문제 시나리오
        # Core 1 reads
        arr1 = Arrow(sm.get_right(), cores[0].get_left(), color=UB_COLOR, stroke_width=2, buff=0.1)
        r1 = Text("read", font_size=12, color=UB_COLOR)
        r1.next_to(arr1, UP, buff=0.02)
        self.play(GrowArrow(arr1), FadeIn(r1))

        # SM_WAIT_GE marks as done
        done = Text("SM_WAIT_GE:\n다 읽었다고 표시!", font_size=12, color=RED)
        done.move_to(LEFT * 0 + UP * 1.8)
        self.play(FadeIn(done))

        # Another core writes (thinking data is consumed)
        arr_w = Arrow(cores[2].get_left(), sm.get_right(), color=RED, stroke_width=3, buff=0.1)
        wt = Text("write\n(새 데이터 덮어씀!)", font_size=12, color=RED)
        wt.next_to(arr_w, DOWN, buff=0.02)
        self.play(GrowArrow(arr_w), FadeIn(wt))

        # Core 3 reads wrong data
        arr_wrong = Arrow(sm.get_right(), cores[2].get_left(), color=RED, stroke_width=2, buff=0.1)
        wrong = Text("전혀 다른 데이터를 읽음!", font_size=14, color=RED)
        wrong.move_to(DOWN * 1.8)
        self.play(GrowArrow(arr_wrong), FadeIn(wrong))

        # 해결
        sol_box = RoundedRectangle(corner_radius=0.1, width=6.0, height=0.7,
                                    color="#2ecc71", fill_opacity=0.1, stroke_width=2)
        sol_box.move_to(DOWN * 2.8)
        sol_txt = Text("해결: semaphore entry에 read_reference_count 추가",
                        font_size=15, color="#2ecc71")
        sol_txt.move_to(sol_box)
        self.play(FadeIn(sol_box), FadeIn(sol_txt))
        self.wait(3)


# ======================================================================
# Scene 11: Fan-in Problem (Slide 14)
# ======================================================================
class Scene11_FanIn(Scene):
    def construct(self):
        title = phase_title("APPENDIX: Fan-in 문제", 11)
        self.play(FadeIn(title))

        # SM
        sm = make_block("Shared Memory", 2.5, 1.0, SM_COLOR, 16)
        sm.move_to(RIGHT * 3 + UP * 0.5)

        # 4 Cores
        cores = VGroup()
        for i in range(4):
            c = make_block(f"core {i+1}", 1.2, 0.5, COMPUTE_COLOR, 13)
            c.move_to(LEFT * 2 + UP * (1.5 - i * 1.0))
            cores.add(c)

        self.play(FadeIn(sm), FadeIn(cores))

        # Core 1 writes
        arr1 = Arrow(cores[0].get_right(), sm.get_left(), color=DMA_COLOR, stroke_width=2, buff=0.1)
        w1 = Text("write", font_size=12, color=DMA_COLOR)
        w1.next_to(arr1, UP, buff=0.02)
        self.play(GrowArrow(arr1), FadeIn(w1))

        # SM_POST marks as ready
        ready = Text("SM_POST:\n읽을 준비 됐다고 표시!", font_size=12, color=RED)
        ready.move_to(RIGHT * 0 + UP * 1.8)
        self.play(FadeIn(ready))

        # Someone reads incomplete data
        arr_read = Arrow(sm.get_left(), cores[3].get_right(),
                         color=RED, stroke_width=3, buff=0.1)
        bad = Text("데이터가 다 써진 줄 알고\n읽어가게 됨!", font_size=14, color=RED)
        bad.move_to(DOWN * 1.8)
        self.play(GrowArrow(arr_read), FadeIn(bad))

        # 대표 예: concat
        concat = Text("대표적으로 concat 연산\n(여러 코어가 하나의 데이터를 나누어 작성)", font_size=13, color=YELLOW)
        concat.move_to(DOWN * 0.5 + LEFT * 2)
        self.play(FadeIn(concat))

        # 해결
        sol_box = RoundedRectangle(corner_radius=0.1, width=6.0, height=0.7,
                                    color="#2ecc71", fill_opacity=0.1, stroke_width=2)
        sol_box.move_to(DOWN * 2.8)
        sol_txt = Text("해결: semaphore entry에 write_reference_count 추가",
                        font_size=15, color="#2ecc71")
        sol_txt.move_to(sol_box)
        self.play(FadeIn(sol_box), FadeIn(sol_txt))
        self.wait(3)


# ======================================================================
# Scene 12: Sync Confusion (Slide 15)
# ======================================================================
class Scene12_SyncConfusion(Scene):
    def construct(self):
        title = phase_title("APPENDIX: 동기화 착각 + semaphore id", 12)
        self.play(FadeIn(title))

        # 타임라인
        # 3개 데이터 A, B, C의 쓰기/읽기 순서
        operations = [
            ("A 쓰기", DMA_COLOR, LEFT * 4.5, UP * 1.5),
            ("동기화", "#2ecc71", LEFT * 2.5, UP * 1.5),
            ("A 읽기", UB_COLOR, LEFT * 0.5, UP * 1.5),
            ("메모리\n비움", GREY_A, RIGHT * 1.5, UP * 1.5),
            ("B 쓰기", DMA_COLOR, LEFT * 4.5, UP * 0.3),
            ("B 읽기", UB_COLOR, LEFT * 0.5, UP * 0.3),
            ("C 쓰기", DMA_COLOR, LEFT * 4.5, DOWN * 0.9),
            ("C 읽기", UB_COLOR, LEFT * 0.5, DOWN * 0.9),
        ]

        boxes = VGroup()
        for label, color, xpos, ypos in operations:
            b = make_block(label, 1.5, 0.5, color, 12)
            b.move_to(xpos + ypos)
            boxes.add(b)

        self.play(FadeIn(boxes), run_time=1)

        # 문제 화살표
        problem_arrow = Arrow(
            RIGHT * 1.5 + DOWN * 0.9,
            LEFT * 0.5 + UP * 0.3,
            color=RED, stroke_width=3, buff=0.1
        )
        problem_txt = Text("예상치 못한 지연으로\nB가 아닌 A를 읽음!", font_size=13, color=RED)
        problem_txt.move_to(RIGHT * 3 + DOWN * 0.3)
        self.play(GrowArrow(problem_arrow), FadeIn(problem_txt))

        # 원인
        cause = Text(
            "semaphore는 \"누가\" 쓴 값이고,\n\"누가\" 소비하는지 표시가 없음",
            font_size=14, color=YELLOW
        )
        cause.move_to(DOWN * 2.0)
        self.play(FadeIn(cause))

        # 해결
        sol = RoundedRectangle(corner_radius=0.1, width=7.0, height=0.8,
                                color="#2ecc71", fill_opacity=0.1, stroke_width=2)
        sol.move_to(DOWN * 3.0)
        sol_txt = Text(
            "해결: semaphore id 추가\n"
            "→ id가 일치해야만 읽기 가능, 쓰기 완료 시 id 세팅",
            font_size=14, color="#2ecc71"
        )
        sol_txt.move_to(sol)
        self.play(FadeIn(sol), FadeIn(sol_txt))
        self.wait(3)
