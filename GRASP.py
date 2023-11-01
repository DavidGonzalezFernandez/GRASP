from manim import *
import random

problem = {
    "DOTS_mobject": [],
    "DOTS_coord": [],
    "DOTS_visited": [],
    "EDGES": [],
    "EDGES_mobject": []
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
def display_problem(self):
    # Create the Dots
    self.play(
        LaggedStart(
            *[Create(p) for p in problem["DOTS_mobject"]],
            *[Create(l) for l in problem["EDGES_mobject"]],
            lag_ratio = 0.1
        )
    )

    self.wait()

LINE_HEIGHT = 0.5
get_rectangle_while = lambda : Rectangle(height=LINE_HEIGHT, width=4.8).move_to(2.15*LEFT + 0.875*UP)
get_rectangle_construct = lambda : Rectangle(height=LINE_HEIGHT, width=5.4).move_to(1.14*LEFT + 0.425*UP)
get_rectangle_whole_if_feasible = lambda : Rectangle(height=LINE_HEIGHT*2 - 0.1, width=5.45).move_to(0.175*DOWN + 1.15*LEFT)
get_rectangle_local_search = lambda : Rectangle(height=LINE_HEIGHT, width=5.8).move_to(0.8*DOWN + 0.95*LEFT)

def show_code_grasp_focus_while(self):
    rect = get_rectangle_while()
    img = show_code_grasp(self, rect)
    self.wait()
    self.play(FadeOut(rect), FadeOut(img))

def show_code_grasp_focus_construct(self):
    rect = get_rectangle_construct()
    img = show_code_grasp(self, rect)
    self.wait()
    self.play(FadeOut(rect), FadeOut(img))

def show_code_grasp_focus_feasible(self):
    rect = get_rectangle_whole_if_feasible()
    img = show_code_grasp(self, rect)
    self.wait()
    self.play(FadeOut(rect), FadeOut(img))

def show_code_grasp_focus_search(self):
    rect = get_rectangle_local_search()
    img = show_code_grasp(self, rect)
    self.wait()
    self.play(FadeOut(rect), FadeOut(img))


"""Given the shown GRASP code it uses rectangles to focus on specific parts of the code"""
def explain_code_grasp(self):
    image = show_code_grasp(self)
    
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
    self.play(FadeOut(image), FadeOut(rectangle_while))


"""Shows the code for GRASP"""
def show_code_grasp(self, additional_mobjects=[]):
    DOWN_SHIFT = 0
    image = ImageMobject("my_media//GRASP_code_snippet_v3.png").shift(DOWN_SHIFT * DOWN)

    self.play(
        FadeIn(image),
        *[FadeIn(o) for o in additional_mobjects]    
    )

    self.wait()

    # Return the image so it can removed afterwards
    return image


""""""
# TODO: implement
def explain_restricted_candidate_list(self):

    # Explain different ways of filterin
    #text1 = Text(r"[c_min, c_min + α * (c_max - c_min)]")

    # Explain alpha
    explain_alpha(self)


"""Tries all alpha posible combinations to explain how its value affects the
behaviour of the greedy-randomized approach"""
def explain_alpha(self):
    MIN_ALPHA, MAX_ALPHA = 0, 1
    DISTANCES = get_distances()
    MIN_C, MAX_C = min(DISTANCES.values()), max(DISTANCES.values())

    # Alpha value to be changed
    alpha = ValueTracker(0.5)

    # Rectangle to used as a box containing all the info
    box = Rectangle(width=4, height=3)
    box.move_to([-5, 2, 0])

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
    objects = [box, range_text, alpha_text, line, dot]
    self.play(
        *[FadeIn(o) for o in objects],
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
        *[FadeToColor(problem["DOTS_mobject"][i], color=DEFAULT_COLOR) for i in DISTANCES.keys()]
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

    # Change color of the point
    self.play(FadeToColor(problem["DOTS_mobject"][index], color=LAST_VISITED_COLOR))
    self.wait()

    # Pause to explain the RCL
    explain_restricted_candidate_list(self)

    NUM_SHOW_RCL = 2
    NUM_CREATE_LINES_SLOW = NUM_SHOW_RCL + 0

    # Repeat until all points have been visited
    while len(problem["DOTS_visited"]) != len(problem["DOTS_mobject"]):
        last_x, last_y = problem["DOTS_coord"][ problem["DOTS_visited"][-1] ]

        # Evaluate all remaining points
        distances = get_distances()

        # Filter out the worst ones
        restricted_candidate_list = get_restricted_candidate_list(distances, alpha)

        # Animation to show candidate_list
        if len(problem["DOTS_visited"]) <= NUM_SHOW_RCL:
            the_dots = [d for (i,d) in enumerate(problem["DOTS_mobject"]) if i in restricted_candidate_list.keys()]
            self.play(*[FadeToColor(d, color=RCL_COLOR) for d in the_dots])
            self.wait()
            self.play(*[FadeToColor(d, color=DEFAULT_COLOR) for d in the_dots])

        # Choose next point randomly
        index = random.choice(list(restricted_candidate_list.keys()))

        # Animations for visiting
        new_x, new_y = problem["DOTS_coord"][index]
        line = Line([last_x, last_y, 0], [new_x, new_y, 0])
        self.play(
            FadeToColor(problem["DOTS_mobject"][problem["DOTS_visited"][-1]], color=VISITED_COLOR),
            FadeToColor(problem["DOTS_mobject"][index], color=LAST_VISITED_COLOR),
            Create(line),
            run_time = 1 if len(problem["DOTS_visited"]) <= NUM_CREATE_LINES_SLOW else 0.5
        )

        problem["EDGES_mobject"].append(line)
        problem["DOTS_visited"].append(index)

    # Go back to initial position
    last_x, last_y = problem["DOTS_coord"][ problem["DOTS_visited"][-1] ]
    new_x, new_y = problem["DOTS_coord"][ problem["DOTS_visited"][0] ]
    line = Line([last_x, last_y, 0], [new_x, new_y, 0])
    self.play(
        FadeToColor(problem["DOTS_mobject"][problem["DOTS_visited"][-1]], color=VISITED_COLOR),
        Create(line)
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




class GRASP(Scene):
    def construct(self):
        # TODO: Show the name and meaning
        show_introduction(self)

        # Show the general code for GRASP
        explain_code_grasp(self)
        self.wait()

        # Introduce the problem to solve
        # TODO: show problem's title
        build_problem(self)
        display_problem(self)
        self.wait()

        # Show the general code and focus on construction
        show_code_grasp_focus_construct(self)
        # TODO: Show the code for construction
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