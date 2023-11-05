from manim import *
import random

config.max_files_cached = 1_000

problem = {
    "DOTS_mobject": [],
    "DOTS_coord": [],
    "DOTS_visited": [],
    "EDGES": [],
    "EDGES_mobject": [],
    "DISTANCES_mobjects": []
}


LAST_VISITED_COLOR = RED
VISITED_COLOR = GREY
DEFAULT_COLOR = WHITE
RCL_COLOR = ORANGE


"""Initializes the dict with all the info for the problem"""
def build_problem(self):
    random.seed(2)
    N_POINTS = 20
    for _ in range(N_POINTS):
        x,y = random.uniform(-7, 7), random.uniform(-3, 3)
        problem["DOTS_mobject"].append(Dot(point=[x, y, 0.0], color=DEFAULT_COLOR))
        problem["DOTS_coord"].append((x,y))


"""Animation to draw the problem on the screen"""
def display_problem(self, title=None):
    # Remove the title
    if title is not None:
        self.play(FadeOut(title))

    # Create the Dots
    self.play(
        LaggedStart(
            *[Create(p) for p in problem["DOTS_mobject"]],
            *[Create(l) for l in problem["EDGES_mobject"]],
            lag_ratio = 0.1
        )
    )

    self.wait()

get_huge_black_rectangle = lambda : Rectangle(height=100, width=100, color=BLACK, fill_color=BLACK, fill_opacity=1)

LINE_HEIGHT = 0.5
get_rectangle_while = lambda : Rectangle(height=LINE_HEIGHT, width=4.8).move_to(2.15*LEFT + 0.875*UP)
get_rectangle_construct = lambda : Rectangle(height=LINE_HEIGHT, width=5.4).move_to(1.14*LEFT + 0.425*UP)
get_rectangle_whole_if_feasible = lambda : Rectangle(height=LINE_HEIGHT*2 - 0.1, width=5.45).move_to(0.175*DOWN + 1.15*LEFT)
get_rectangle_local_search = lambda : Rectangle(height=LINE_HEIGHT, width=5.8).move_to(0.8*DOWN + 0.95*LEFT)

def show_code_grasp_focus(self, rect, then=None):
    img, huge_rect = show_code_grasp(self, rect)
    self.wait()
    if then is None:
        self.play(FadeOut(huge_rect), FadeOut(img), FadeOut(rect))
    else:
        self.play(FadeOut(img), FadeOut(rect), FadeIn(then))
        self.wait()
        self.play(
            FadeOut(huge_rect),
            then.animate.scale(1/2).shift(UP*2 + LEFT*5)
        )
    self.wait()
    return then

def show_code_grasp_focus_construct(s):
    img = show_code_grasp_focus(
        s,
        get_rectangle_construct(),
        ImageMobject("my_media//construction_code_snippet_v2.png")
    )
    return img

show_code_grasp_focus_feasible = lambda s : show_code_grasp_focus(
    s,
    get_rectangle_whole_if_feasible(),
    None
)
show_code_grasp_focus_search = lambda s : show_code_grasp_focus(
    s, 
    get_rectangle_local_search(), 
    None
)


"""Given the shown GRASP code it uses rectangles to focus on specific parts of the code"""
def explain_code_grasp(self):
    image, rect = show_code_grasp(self)
    
    rectangle_while = get_rectangle_while()
    rectangle_construct = get_rectangle_construct()
    rectangle_whole_if_feasible = get_rectangle_whole_if_feasible()
    rectangle_local_search = get_rectangle_local_search()

    # Focus con construct
    self.play(Create(rectangle_construct))
    self.wait()

    # Focus on repair
    self.play(ReplacementTransform(rectangle_construct, rectangle_whole_if_feasible))
    self.wait()

    # Focur on local search
    self.play(ReplacementTransform(rectangle_whole_if_feasible, rectangle_local_search))
    self.wait()

    # Focus on loop
    self.play(ReplacementTransform(rectangle_local_search, rectangle_while))
    self.wait()

    # Remove image and rectangle
    self.play(FadeOut(image), FadeOut(rectangle_while), FadeOut(rect))


"""Shows the code for GRASP"""
def show_code_grasp(self, additional_mobjects=[]):
    rect = get_huge_black_rectangle()
    DOWN_SHIFT = 0
    image = ImageMobject("my_media//GRASP_code_snippet_v3.png").shift(DOWN_SHIFT * DOWN)

    self.play(
        FadeIn(rect),
        FadeIn(image),
        *[FadeIn(o) for o in additional_mobjects]    
    )

    self.wait()

    # Return the image so it can removed afterwards
    return image, rect


def explain_p(self):
    DISTANCES = get_distances()
    MIN_P, MAX_P = 1, len(DISTANCES.values())
    MIN_C, MAX_C = min(DISTANCES.values()), max(DISTANCES.values())

    p = ValueTracker(5)

    box = Rectangle(width=4, height=5, fill_color=BLACK, fill_opacity=1).move_to([-5, 1, 0])

    # Order the costs
    next_to_mobjects = {
        dot: i
        for (i,dot) in enumerate(problem["DOTS_mobject"])
        if i not in problem["DOTS_visited"]
    }

    for (i,key) in enumerate(next_to_mobjects.keys()):
        next_to_mobjects[key] = problem["DISTANCES_mobjects"][i]

    dist_text = {
        distance: problem["DISTANCES_mobjects"][i]
        for (i,distance) in enumerate(DISTANCES.values())
    }

    SHIFT_DOWN = 0.25 * DOWN
    ORIGINAL_POSITION = LEFT*5 + UP*3.25

    for t in problem["DISTANCES_mobjects"]:
        t.set_z_index(box.z_index + 2)

    x_start_line, x_end_line = LEFT*5.5, LEFT*4.5
    get_line = lambda : Line(
        x_start_line + UP*3.25 + SHIFT_DOWN*int(p.get_value()) - SHIFT_DOWN*0.5,
        x_end_line + UP*3.25 + SHIFT_DOWN*int(p.get_value()) - SHIFT_DOWN*0.5,
        z_index = box.z_index + 1
    )
    line = get_line()

    get_text = lambda : Text(f"{int(p.get_value())}").scale(1/3).next_to(line, RIGHT)
    text = get_text()

    self.play(
        FadeIn(box),
        *[
            dist_text[distance].animate.move_to(ORIGINAL_POSITION + SHIFT_DOWN*i).scale(3/4)
            for (i,distance) in enumerate(sorted(dist_text.keys()))
        ]
    )

    self.wait()

    self.play(
        Create(line),
        Create(text)
    )

    def line_updater(line):
        line.become(get_line())
        text.become(get_text())

    for (i,dist) in DISTANCES.items():
        def the_updater(x, dist=dist):
            x.set_color(DEFAULT_COLOR if (sorted(DISTANCES.values()).index(dist)+1 > int(p.get_value())) else RCL_COLOR)
        problem["DOTS_mobject"][i].add_updater(the_updater)
    
    line.add_updater(line_updater)
    text.add_updater(lambda z: z.become(get_text()))

    self.wait()

    # TODO: show different values of p
    # To min
    self.play(p.animate.set_value(MIN_P), run_time=2)
    self.wait()

    # To max
    self.play(p.animate.set_value(MAX_P), run_time=2)
    self.wait()

    # To medium
    self.play(p.animate.set_value((MAX_P+MIN_P)*0.5), run_time=2)
    self.wait()

    line.clear_updaters()
    text.clear_updaters()

    # Return distantes to original position
    self.play(
        *[distance.animate.next_to(dot, RIGHT).scale(4/3) for (dot,distance) in next_to_mobjects.items()],
        FadeOut(line),
        FadeOut(text)
    )

    # TODO Remove objects

    return box

"""Explains how to create the RCL"""
def explain_restricted_candidate_list(self):
    # Explain p
    box = explain_p(self)
    self.wait()

    # Explain alpha
    explain_alpha(self, box)


"""Tries all alpha posible combinations to explain how its value affects the
behaviour of the greedy-randomized approach"""
def explain_alpha(self, box):
    # TODO: add formula
    #text1 = Text(r"[c_min, c_min + α * (c_max - c_min)]")

    MIN_ALPHA, MAX_ALPHA = 0, 1
    DISTANCES = get_distances()
    MIN_C, MAX_C = min(DISTANCES.values()), max(DISTANCES.values())

    # Alpha value to be changed
    alpha = ValueTracker(0.5)

    # Rectangle to used as a box containing all the info
    new_box = Rectangle(width=4, height=3, fill_color=BLACK, fill_opacity=1).move_to([-5, 2, 0])

    get_text = lambda : Text(f"[{MIN_C:.2f}, {(MIN_C+alpha.get_value()*(MAX_C-MIN_C)):.2f}]").shift(5*LEFT+2.5*UP).scale(1/2)
    # The text showing the MIN and MAX values allowed within the range
    range_text = get_text()
    
    x_start_line, x_end_line, y_line = -6.5, -3.5, 1.5
    # The line that will be traveled by the Dot
    line = Line((x_start_line, y_line, 0), (x_end_line, y_line, 0))

    get_dot = lambda : Dot(point=((x_start_line + alpha.get_value()*(x_end_line-x_start_line)), y_line, 0))
    # Dot to visually see where the alpha is located
    dot = get_dot()
    
    get_lambda_text = lambda : Text(f"{(alpha.get_value()):.2f}").scale(1/2).next_to(dot, DOWN)
    # Text displaying the value of alpha
    alpha_text = get_lambda_text()

    # All the objects to be displayed
    objects = [range_text, alpha_text, line, dot]
    self.play(
        *[FadeIn(o) for o in objects],
        box.animate.become(new_box),
        *[
            FadeToColor(problem["DOTS_mobject"][i], color=RCL_COLOR)
            for i in DISTANCES.keys()
            if (DISTANCES[i] <= (MIN_C+alpha.get_value()*(MAX_C-MIN_C)))
        ]
    )

    self.wait()

    for (i,d) in DISTANCES.items():
        # Update the colors of the points if they can be included in RCL
        def the_updater(x, i=i):
            x.set_color(DEFAULT_COLOR if (DISTANCES[i] > (MIN_C+alpha.get_value()*(MAX_C-MIN_C))) else RCL_COLOR)
        problem["DOTS_mobject"][i].add_updater(the_updater)

    def update_lambda(point):
        point.become(get_dot())
        alpha_text.become(get_lambda_text())
    
    dot.add_updater(update_lambda)
    range_text.add_updater(lambda z: z.become(get_text()))

    # From center to MAX
    self.play(alpha.animate.set_value(MAX_ALPHA), run_time=2)
    self.wait()

    # From MAX to MIN
    self.play(alpha.animate.set_value(MIN_ALPHA), run_time=2)
    self.wait()

    # From MIN to center
    self.play(alpha.animate.set_value(0.5), run_time=2)
    self.wait()

    for i in DISTANCES.keys():
        problem["DOTS_mobject"][i].clear_updaters()
    
    range_text.clear_updaters()
    dot.clear_updaters()
    alpha_text.clear_updaters()

    # Restore view to match previous
    self.play(
        *[FadeOut(o) for o in objects],
        #*[FadeToColor(problem["DOTS_mobject"][i], color=DEFAULT_COLOR) for i in DISTANCES.keys()],
        FadeOut(box),
        FadeOut(new_box)
    )

    self.wait()


"""Calculates the distance from the last visited point to all non-visited points"""
def get_distances():
    last_x, last_y = problem["DOTS_coord"][ problem["DOTS_visited"][-1] ]

    return {
        i: (abs(p_x-last_x)**2 + abs(p_y - last_y)**2)**(1/2)
        for (i,(p_x, p_y)) in enumerate(problem["DOTS_coord"])
        if i not in problem["DOTS_visited"]
    }


"""Calculates the RCL given a list of distances and alpha value"""
def get_restricted_candidate_list(distances, alpha):
    max_distance, min_distance = max(distances.values()), min(distances.values())

    return {
        i: distance
        for (i,distance) in distances.items()
        if distance>=min_distance  and  distance<=(min_distance + alpha*(max_distance-min_distance))
    }


"""Shows the greedy-randomized construction of a solution"""
def construct_initial_solution(self):
    alpha = 0.25

    random.seed(0)

    # Choose initial point and modify list
    index = random.randint(0, len(problem["DOTS_mobject"]))
    problem["DOTS_visited"].append(index)

    pos_solution = LEFT*6.6 + UP*2.825
    pos_get_cand_1 = LEFT*6.6 + UP*2.67
    pos_while = LEFT*6.6 + UP*2.4
    pos_costs = LEFT*6.6 + UP*2.25
    pos_RCL = LEFT*6.6 + UP*1.8
    pos_random = LEFT*6.6 + UP*1.65
    pos_append = LEFT*6.6 + UP*1.5
    pos_get_cand_2 = LEFT*6.6 + UP*1.355
    pos_return = LEFT*6.6 + UP*1.09

    arrow = Arrow(start=pos_solution, end=pos_solution+RIGHT).scale(1/2).move_to(pos_solution)

    # Change color of the point
    self.play(
        FadeToColor(problem["DOTS_mobject"][index], color=LAST_VISITED_COLOR),
        Create(arrow)
    )
    self.wait()

    NUM_STEP_BY_STEP = 2

    # Repeat until all points have been visited
    while len(problem["DOTS_visited"]) != len(problem["DOTS_mobject"]):
        # Move arrow and color candidates
        if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP:
            self.play(
                arrow.animate.move_to(pos_get_cand_1 if len(problem["DOTS_visited"])==1 else pos_get_cand_2),
                *[FadeToColor(p, RCL_COLOR) for (i,p) in enumerate(problem["DOTS_mobject"]) if i not in problem["DOTS_visited"]]
            )
            self.wait()
        elif len(problem["DOTS_visited"]) == NUM_STEP_BY_STEP+1:
            self.play(arrow.animate.move_to(pos_while))
            self.wait()

        last_x, last_y = problem["DOTS_coord"][ problem["DOTS_visited"][-1] ]

        # Evaluate all remaining points
        distances = get_distances()
        if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP:
            distances_text = [
                Text(f"{distances[i]:.2f}").next_to(dot, RIGHT).scale(1/2.5).shift(LEFT*0.5)
                for (i,dot) in enumerate(problem["DOTS_mobject"])
                if i not in problem["DOTS_visited"]
            ]
            problem["DISTANCES_mobjects"].extend(distances_text)
            self.play(
                arrow.animate.move_to(pos_costs),
                *[FadeIn(t) for t in distances_text]
            )
            self.wait()

        if len(problem["DOTS_visited"]) == 1:
            self.play(
                arrow.animate.move_to(pos_RCL),
            )
            self.wait()
            # Pause to explain the RCL
            explain_restricted_candidate_list(self)

        # Filter out the worst ones
        restricted_candidate_list = get_restricted_candidate_list(distances, alpha)

        # Animation to show restricted candidate_list
        if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP:
            the_dots_1 = [d for (i,d) in enumerate(problem["DOTS_mobject"]) if i not in problem["DOTS_visited"] and i not in restricted_candidate_list.keys()]
            the_dots_2 = [d for (i,d) in enumerate(problem["DOTS_mobject"]) if i not in problem["DOTS_visited"] and i in restricted_candidate_list.keys()]
            self.play(
                *[FadeToColor(d, color=DEFAULT_COLOR) for d in the_dots_1],
                *[FadeToColor(d, color=RCL_COLOR) for d in the_dots_2],
                arrow.animate.move_to(pos_RCL),
                *[FadeOut(t) for t in distances_text]
            )
            self.wait()
        
        problem["DISTANCES_mobjects"].clear()

        # Choose next point randomly
        index = random.choice(list(restricted_candidate_list.keys()))
        if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP:
            the_dots = [d for (i,d) in enumerate(problem["DOTS_mobject"]) if i in restricted_candidate_list.keys()]
            self.play(
                *[FadeToColor(d, color=DEFAULT_COLOR) for d in the_dots if d is not problem["DOTS_mobject"][index]],
                arrow.animate.move_to(pos_random)
            )
            self.wait()

        # Animations for visiting
        new_x, new_y = problem["DOTS_coord"][index]
        line = Line([last_x, last_y, 0], [new_x, new_y, 0])
        if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP:
            self.play(
                FadeToColor(problem["DOTS_mobject"][problem["DOTS_visited"][-1]], color=VISITED_COLOR),
                FadeToColor(problem["DOTS_mobject"][index], color=LAST_VISITED_COLOR),
                Create(line),
                arrow.animate.move_to(pos_append),
                run_time = 1 if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP else 0.5
            )
            self.wait()
        else:
            self.play(
                FadeToColor(problem["DOTS_mobject"][problem["DOTS_visited"][-1]], color=VISITED_COLOR),
                FadeToColor(problem["DOTS_mobject"][index], color=LAST_VISITED_COLOR),
                Create(line),
                run_time = 1 if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP else 0.5
            )

        problem["EDGES_mobject"].append(line)
        problem["DOTS_visited"].append(index)



    # Go back to initial position
    last_x, last_y = problem["DOTS_coord"][ problem["DOTS_visited"][-1] ]
    new_x, new_y = problem["DOTS_coord"][ problem["DOTS_visited"][0] ]
    line = Line([last_x, last_y, 0], [new_x, new_y, 0])
    self.play(
        FadeToColor(problem["DOTS_mobject"][problem["DOTS_visited"][-1]], color=VISITED_COLOR),
        Create(line),
        arrow.animate.move_to(pos_return)
    )
    problem["EDGES_mobject"].append(line)

    self.wait()


"""Shows the title to do the introduction"""
def show_introduction(self):
    title_whole_name = Paragraph("Greedy", "Randomized", "Adaptive", "Search", "Procedure")
    self.add(title_whole_name)
    self.wait()

    title_GRASP = Text("GRASP")
    self.play(ReplacementTransform(title_whole_name, title_GRASP))
    self.wait()

    # TODO: pensar qué más poner

    self.play(FadeOut(title_GRASP))
    self.wait()


def introduce_problem(self):
    title = Text("Problema del viajante")
    self.play(Create(title))
    return title


class GRASP(Scene):
    def construct(self):
        # TODO: Show the name and meaning
        show_introduction(self)

        # Show the general code for GRASP
        explain_code_grasp(self)
        self.wait()

        # Introduce the problem to solve
        build_problem(self)
        title = introduce_problem(self)
        self.wait()
        display_problem(self, title)
        self.wait()

        # Show the general code and the code for construction
        code_img = show_code_grasp_focus_construct(self)
        # Visualize construction
        construct_initial_solution(self)

        # Show the general code and focus on repair
        show_code_grasp_focus_feasible(self)
        # TODO: Show the idea behind repair
        # TODO: Visualize repair

        # Show the general code and focus on local search
        show_code_grasp_focus_search(self)
        # TODO: Show the code for local search
        # TODO: Visualize local search

        # TODO: Redo explanation of code

        # TODO: Show results and modifications of parameters

        # TODO: Show extensions and improvements