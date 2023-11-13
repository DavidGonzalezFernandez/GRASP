from manim import *
import random

config.max_files_cached = 1_000

problem = {
    "DOTS_mobject": [],
    "DOTS_coord": [],
    "DOTS_visited": [],
    "DOTS_VISITED_search": [],
    "EDGES": [],
    "EDGES_main_mobject": [],
    "EDGES_tmp_mobject": [],
    "DISTANCES_mobjects": [],
}

VISITING_DOT_COLOR = RED
VISITED_DOT_COLOR = GREY
DEFAULT_DOT_COLOR = WHITE
SELECTED_DOT_COLOR = BLUE

MAIN_LINE_COLOR = GREY
EXPLORING_LINE_COLOR = BLUE


"""CODE: Initializes the dict with all the info for the problem"""
def initialize_problem():
    random.seed(2)
    N_POINTS = 20
    for _ in range(N_POINTS):
        x,y = random.uniform(-7, 7), random.uniform(-3, 3)
        problem["DOTS_mobject"].append(Dot(point=[x, y, 0.0], color=DEFAULT_DOT_COLOR))
        problem["DOTS_coord"].append((x,y))


"""ANIMATION: draws the problem on the screen"""
def display_problem(grasp_scene, title=None):
    # Remove the title
    if title is not None:
        grasp_scene.play(FadeOut(title))

    # Create the Dots
    grasp_scene.play(
        LaggedStart(
            *[Create(d) for d in problem["DOTS_mobject"]],
            *[Create(e) for e in problem["EDGES_main_mobject"]],
            *[Create(e) for e in problem["EDGES_tmp_mobject"]],
            lag_ratio = 0.1
        )
    )

    grasp_scene.wait()

get_huge_black_rectangle = lambda : Rectangle(height=100, width=100, color=BLACK, fill_color=BLACK, fill_opacity=1)

LINE_HEIGHT = 0.5
get_rectangle_while = lambda : Rectangle(height=LINE_HEIGHT, width=4.75).move_to(0.88*UP + 0.875*LEFT)
get_rectangle_construct = lambda : Rectangle(height=LINE_HEIGHT, width=5.55).move_to(0.465*UP + 0.225*RIGHT)
get_rectangle_local_search = lambda : Rectangle(height=LINE_HEIGHT, width=5.8).move_to(0.025*UP + 0.35*RIGHT)


"""ANIMATION: Shows the code and draws a rectangle around the construction invocation"""
def show_code_grasp_focus_construct(grasp_scene):
    rect = get_rectangle_construct()
    code_img = ImageMobject("my_media//construction_code_snippet_v2.png")

    # Show the code
    img, huge_rect = show_code_grasp(grasp_scene, rect)
    grasp_scene.wait()

    # Show specific implementation for helper function
    grasp_scene.play(
        FadeOut(img), 
        FadeOut(rect), 
        FadeIn(code_img)
    )
    grasp_scene.wait()

    # Scale and move the code
    grasp_scene.play(
        FadeOut(huge_rect),
        code_img.animate.become(ImageMobject("my_media//construction_code_snippet_v2.png").scale(1/2).shift(UP*2 + LEFT*5))
    )
    grasp_scene.wait()

    return code_img
    

"""ANIMATION: Shows the code and draws a rectangle around the local search invocation"""
def show_code_grasp_focus_search(grasp_scene):

    rect = get_rectangle_local_search()
    code_img = ImageMobject("my_media//search_code_snippet_v4.png")
    code_img_2 = ImageMobject("my_media//search_code_snippet_v5.png")

    # Show the code
    img, huge_rect = show_code_grasp(grasp_scene, rect)
    grasp_scene.wait()

    rect_break = Rectangle(height=0.45, width=1).shift(DOWN*1.425 + LEFT * 1.655)

    #Show specific implementation for helper function
    grasp_scene.play(
        FadeOut(img), 
        FadeOut(rect), 
        FadeIn(code_img)
    )
    grasp_scene.wait()

    # Compare first vs best approach
    grasp_scene.play(
        FadeOut(code_img),
        FadeIn(code_img_2),
        Create(rect_break)
    )
    grasp_scene.wait()

    # Scale and move the code
    grasp_scene.play(
        FadeOut(huge_rect),
        FadeOut(rect_break),
        code_img_2.animate.scale(1/2).shift(UP*2.1 + LEFT*4.5)
    )
    grasp_scene.wait()

    return code_img_2


"""ANIMATION: Given the shown GRASP code it uses rectangles to focus on specific parts of the code"""
def explain_code_grasp(grasp_scene):
    image, rect = show_code_grasp(grasp_scene)

    rectangle_while = get_rectangle_while()
    rectangle_construct = get_rectangle_construct()
    rectangle_local_search = get_rectangle_local_search()

    # Focus con construct
    grasp_scene.play(Create(rectangle_construct))
    grasp_scene.wait()

    # Focus on local search
    grasp_scene.play(ReplacementTransform(rectangle_construct, rectangle_local_search))
    grasp_scene.wait()

    # Focus on loop
    grasp_scene.play(ReplacementTransform(rectangle_local_search, rectangle_while))
    grasp_scene.wait()

    # Remove image and rectangle
    grasp_scene.play(FadeOut(image), FadeOut(rect), FadeOut(rectangle_while))
    grasp_scene.wait()


"""ANIMATION: Shows the code for GRASP"""
def show_code_grasp(grasp_scene, additional_mobjects=[]):
    rect = get_huge_black_rectangle()
    DOWN_SHIFT = 0
    image = ImageMobject("my_media//GRASP_code_snippet_v7.png").shift(DOWN_SHIFT * DOWN)

    grasp_scene.play(
        LaggedStart(
            Create(rect),
            FadeIn(image),
            *[Create(o) for o in additional_mobjects],
            lag_ratio = 0.1
        )
    )

    grasp_scene.wait()

    # Return the image so it can removed afterwards
    return image, rect


"""ANIMATION: Tries all p posible values to explain how its value affects the
behaviour of the greedy-randomized approach"""
def explain_p(grasp_scene):
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

    grasp_scene.play(
        FadeIn(box),
        *[
            dist_text[distance].animate.move_to(ORIGINAL_POSITION + SHIFT_DOWN*i).scale(3/4)
            for (i,distance) in enumerate(sorted(dist_text.keys()))
        ]
    )

    grasp_scene.wait()

    the_dots_1 = [problem["DOTS_mobject"][i] for (i,dist) in DISTANCES.items() if sorted(DISTANCES.values()).index(dist)+1 > int(p.get_value())]
    the_dots_2 = [problem["DOTS_mobject"][i] for (i,dist) in DISTANCES.items() if sorted(DISTANCES.values()).index(dist)+1 <=  int(p.get_value())]

    grasp_scene.play(
        Create(line),
        Create(text),
        *[FadeToColor(d, DEFAULT_DOT_COLOR) for d in the_dots_1],
        *[FadeToColor(d, SELECTED_DOT_COLOR) for d in the_dots_2]
    )

    def line_updater(line):
        line.become(get_line())
        text.become(get_text())

    for (i,dist) in DISTANCES.items():
        def the_updater(x, dist=dist):
            x.set_color(DEFAULT_DOT_COLOR if (sorted(DISTANCES.values()).index(dist)+1 > int(p.get_value())) else SELECTED_DOT_COLOR)
        problem["DOTS_mobject"][i].add_updater(the_updater)
    
    line.add_updater(line_updater)
    text.add_updater(lambda z: z.become(get_text()))

    grasp_scene.wait()

    # To min
    grasp_scene.play(p.animate.set_value(MIN_P), run_time=2)
    grasp_scene.wait()

    # To max
    grasp_scene.play(p.animate.set_value(MAX_P), run_time=2)
    grasp_scene.wait()

    # To medium
    grasp_scene.play(p.animate.set_value(5), run_time=2)
    grasp_scene.wait()

    line.clear_updaters()
    text.clear_updaters()

    for (i,dist) in DISTANCES.items():
        problem["DOTS_mobject"][i].clear_updaters()

    # Return distantes to original position and DOTS to original color
    grasp_scene.play(
        *[distance.animate.next_to(dot, RIGHT).scale(4/3) for (dot,distance) in next_to_mobjects.items()],
        FadeOut(line),
        FadeOut(text),
        *[FadeToColor(problem["DOTS_mobject"][i], color=SELECTED_DOT_COLOR) for i in DISTANCES.keys()],
    )

    return box


"""ANIMATION: Explains how to create the RCL"""
def explain_restricted_candidate_list(grasp_scene):
    # Explain alpha
    explain_alpha(grasp_scene)


"""ANIMATION: Tries all alpha posible p to explain how its value affects the
behaviour of the greedy-randomized approach"""
def explain_alpha(grasp_scene):
    MIN_ALPHA, MAX_ALPHA = 0, 1
    DISTANCES = get_distances()
    MIN_C, MAX_C = min(DISTANCES.values()), max(DISTANCES.values())

    # Alpha value to be changed
    alpha = ValueTracker(0.5)

    # Rectangle to used as a box containing all the info
    new_box = Rectangle(width=4, height=3, fill_color=BLACK, fill_opacity=1).move_to([-5, 2, 0])
    
    # Initial explanation for alpha
    raw_formula_text = Paragraph(r"[c_min ,", r"c_min + α*(c_max - c_min)]").shift(5*LEFT+2.75*UP).scale(2/5)
    alpha_range = Text(r"0 ≤ α ≤ 1").shift(5*LEFT+1.25*UP).scale(1/2)
    grasp_scene.play(
        FadeIn(new_box),
        FadeIn(raw_formula_text),
        FadeIn(alpha_range))
    grasp_scene.wait()

    get_text = lambda : Text(f"[{MIN_C:.2f}, {(MIN_C+alpha.get_value()*(MAX_C-MIN_C)):.2f}]").shift(5*LEFT+2.75*UP).scale(1/2)
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
    objects = [range_text, alpha_text, dot, line]
    grasp_scene.play(
        FadeIn(alpha_text),
        FadeIn(dot),
        FadeIn(line),
        ReplacementTransform(raw_formula_text, range_text),
        *[
            FadeToColor(problem["DOTS_mobject"][i], color=SELECTED_DOT_COLOR)
            for i in DISTANCES.keys()
            if (DISTANCES[i] <= (MIN_C+alpha.get_value()*(MAX_C-MIN_C)))
        ],
        *[
            FadeToColor(problem["DOTS_mobject"][i], color=DEFAULT_DOT_COLOR)
            for i in DISTANCES.keys()
            if (DISTANCES[i] > (MIN_C+alpha.get_value()*(MAX_C-MIN_C)))
        ],
        FadeOut(alpha_range)
    )

    grasp_scene.wait()

    for (i,d) in DISTANCES.items():
        # Update the colors of the points if they can be included in RCL
        def the_updater(x, i=i):
            x.set_color(DEFAULT_DOT_COLOR if (DISTANCES[i] > (MIN_C+alpha.get_value()*(MAX_C-MIN_C))) else SELECTED_DOT_COLOR)
        problem["DOTS_mobject"][i].add_updater(the_updater)

    def update_lambda(point):
        point.become(get_dot())
        alpha_text.become(get_lambda_text())
    
    dot.add_updater(update_lambda)
    range_text.add_updater(lambda z: z.become(get_text()))

    # From center to MAX
    grasp_scene.play(alpha.animate.set_value(MAX_ALPHA), run_time=2)
    grasp_scene.wait()

    # From MAX to MIN
    grasp_scene.play(alpha.animate.set_value(MIN_ALPHA), run_time=2)
    grasp_scene.wait()

    # From MIN to center
    grasp_scene.play(alpha.animate.set_value(0.2), run_time=2)
    grasp_scene.wait()

    for i in DISTANCES.keys():
        problem["DOTS_mobject"][i].clear_updaters()

    range_text.clear_updaters()
    alpha_text.clear_updaters()
    dot.clear_updaters()

    grasp_scene.play(
        FadeOut(range_text),
        FadeOut(alpha_text),
        FadeOut(dot),
        FadeOut(line),
        FadeOut(new_box),
        *[FadeToColor(problem["DOTS_mobject"][i], color=SELECTED_DOT_COLOR) for i in DISTANCES.keys()],
    )

    grasp_scene.wait()


"""CODE: Calculates the distance from the last visited point to all non-visited points"""
def get_distances():
    last_x, last_y = problem["DOTS_coord"][ problem["DOTS_visited"][-1] ]

    return {
        i: (abs(p_x-last_x)**2 + abs(p_y - last_y)**2)**(1/2)
        for (i,(p_x, p_y)) in enumerate(problem["DOTS_coord"])
        if i not in problem["DOTS_visited"]
    }


"""CODE: Calculates the RCL given a list of distances and alpha value"""
def get_restricted_candidate_list(distances, alpha):
    max_distance, min_distance = max(distances.values()), min(distances.values())

    return {
        i: distance
        for (i,distance) in distances.items()
        if distance>=min_distance  and  distance<=(min_distance + alpha*(max_distance-min_distance))
    }


"""ANIMATION & CODE: greedy-randomized construction of a solution.
Shows the process and updates the dict"""
def construct_initial_solution(grasp_scene, img_code):
    alpha = 0.2

    random.seed(1)

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
    pos_return = LEFT*6.6 + UP*1.08

    arrow = Arrow(start=pos_solution, end=pos_solution+RIGHT).scale(1/2).move_to(pos_solution)

    # Change color of the point
    grasp_scene.play(
        FadeToColor(problem["DOTS_mobject"][index], color=VISITING_DOT_COLOR),
        Create(arrow)
    )
    grasp_scene.wait()

    NUM_STEP_BY_STEP = 2

    # Repeat until all points have been visited
    while len(problem["DOTS_visited"]) != len(problem["DOTS_mobject"]):
        # Move arrow and color candidates
        if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP:
            grasp_scene.play(
                arrow.animate.move_to(pos_get_cand_1 if len(problem["DOTS_visited"])==1 else pos_get_cand_2),
                *[FadeToColor(p, SELECTED_DOT_COLOR) for (i,p) in enumerate(problem["DOTS_mobject"]) if i not in problem["DOTS_visited"]]
            )
            grasp_scene.wait()
        elif len(problem["DOTS_visited"]) == NUM_STEP_BY_STEP+1:
            grasp_scene.play(arrow.animate.move_to(pos_while))
            grasp_scene.wait()

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
            grasp_scene.play(
                arrow.animate.move_to(pos_costs),
                *[FadeIn(t) for t in distances_text]
            )
            grasp_scene.wait()

        if len(problem["DOTS_visited"]) == 1:
            grasp_scene.play(
                arrow.animate.move_to(pos_RCL),
            )
            grasp_scene.wait()
            # Pause to explain the RCL
            explain_restricted_candidate_list(grasp_scene)

        # Filter out the worst ones
        restricted_candidate_list = get_restricted_candidate_list(distances, alpha)

        # Animation to show restricted candidate_list
        if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP:
            the_dots_1 = [d for (i,d) in enumerate(problem["DOTS_mobject"]) if i not in problem["DOTS_visited"] and i not in restricted_candidate_list.keys()]
            the_dots_2 = [d for (i,d) in enumerate(problem["DOTS_mobject"]) if i not in problem["DOTS_visited"] and i in restricted_candidate_list.keys()]
            grasp_scene.play(
                *[FadeToColor(d, color=DEFAULT_DOT_COLOR) for d in the_dots_1],
                *[FadeToColor(d, color=SELECTED_DOT_COLOR) for d in the_dots_2],
                arrow.animate.move_to(pos_RCL),
                *[FadeOut(t) for t in distances_text]
            )
            grasp_scene.wait()
        
        problem["DISTANCES_mobjects"].clear()

        # Choose next point randomly
        index = random.choice(list(restricted_candidate_list.keys()))
        if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP:
            the_dots = [d for (i,d) in enumerate(problem["DOTS_mobject"]) if i in restricted_candidate_list.keys()]
            grasp_scene.play(
                *[FadeToColor(d, color=DEFAULT_DOT_COLOR) for d in the_dots if d is not problem["DOTS_mobject"][index]],
                arrow.animate.move_to(pos_random)
            )
            grasp_scene.wait()

        # Animations for visiting
        new_x, new_y = problem["DOTS_coord"][index]
        line = Line([last_x, last_y, 0], [new_x, new_y, 0], color=MAIN_LINE_COLOR)
        if len(problem["DOTS_visited"]) <= NUM_STEP_BY_STEP:
            grasp_scene.play(
                FadeToColor(problem["DOTS_mobject"][problem["DOTS_visited"][-1]], color=VISITED_DOT_COLOR),
                FadeToColor(problem["DOTS_mobject"][index], color=VISITING_DOT_COLOR),
                Create(line),
                arrow.animate.move_to(pos_append),
                run_time = 1
            )
            grasp_scene.wait()
        else:
            grasp_scene.play(
                FadeToColor(problem["DOTS_mobject"][problem["DOTS_visited"][-1]], color=VISITED_DOT_COLOR),
                FadeToColor(problem["DOTS_mobject"][index], color=VISITING_DOT_COLOR),
                Create(line),
                run_time = 1/3
            )

        problem["EDGES_main_mobject"].append(line)
        problem["DOTS_visited"].append(index)

    # Go back to initial position
    last_x, last_y = problem["DOTS_coord"][ problem["DOTS_visited"][-1] ]
    new_x, new_y = problem["DOTS_coord"][ problem["DOTS_visited"][0] ]
    line = Line([last_x, last_y, 0], [new_x, new_y, 0], color=MAIN_LINE_COLOR)
    grasp_scene.play(
        FadeToColor(problem["DOTS_mobject"][problem["DOTS_visited"][-1]], color=VISITED_DOT_COLOR),
        Create(line),
        arrow.animate.move_to(pos_return)
    )
    problem["EDGES_main_mobject"].append(line)

    grasp_scene.wait()

    # Remove code
    grasp_scene.play(FadeOut(arrow), FadeOut(img_code))
    grasp_scene.wait()


"""ANIMATION: Shows the title to do the introduction"""
def show_introduction(grasp_scene):
    group_name = Paragraph("Grupo 1", "\tDavid González Fernández", "\tSergio Arroni del Riego", "\tJosé Manuel Lamas Pérez").scale(2/5).shift(DOWN*3)

    title_whole_name = Paragraph("Greedy", "Randomized", "Adaptive", "Search", "Procedure").shift(UP*0.5)
    grasp_scene.add(title_whole_name, group_name)
    grasp_scene.wait()

    title_GRASP = Text("GRASP").shift(UP*2)

    details_text = Paragraph("• Multicomienzo", "\n• Optimización combinatoria").scale(2/3)

    grasp_scene.play(
        ReplacementTransform(title_whole_name, title_GRASP),
        FadeOut(group_name),
        FadeIn(details_text)
    )
    grasp_scene.wait()

    grasp_scene.play(FadeOut(title_GRASP), FadeOut(details_text))
    grasp_scene.wait()


"""CODE: Returns the total cost given an order of visited nodes"""
def get_total_cost(order):
    res = 0

    for i in range(0, len(order)):
        coord_start, coord_end = problem["DOTS_coord"][order[i]], problem["DOTS_coord"][order[(i+1)%len(order)]]
        prev = res
        res += (abs(coord_start[0] - coord_end[0])**2 + abs(coord_start[1] - coord_end[1])**2)**(1/2)
        assert res >= 0
        assert res >= prev

    return res


"""ANIMATION: Displays the title of the problem"""
def introduce_problem(grasp_scene):
    title = Text("Problema del viajante")
    grasp_scene.play(Create(title))
    return title


"""ANIMATION & CODE: performs the local search given the current solution in the dict.
Used first-fit approach. Shows the process and updates the dict"""
def do_local_search(grasp_scene, code_img):
    is_optimal = False
    
    solution = problem["DOTS_visited"].copy()
    solution_cost = get_total_cost(solution)

    # Make a copy of all the edges
    for edge in problem["EDGES_main_mobject"]:
        new_edge = Line(edge.start, edge.end, color=MAIN_LINE_COLOR, z_index=1)
        problem["EDGES_tmp_mobject"].append(new_edge)
    
    grasp_scene.add(*problem["EDGES_tmp_mobject"])

    original_cost_text = Text(f"{solution_cost:.2f}", color=VISITED_DOT_COLOR).scale(1/2).shift(DOWN * 3 + RIGHT * 5)
    new_cost_text = Text("¿?", color=EXPLORING_LINE_COLOR).scale(1/2).shift(DOWN * 3.5 + RIGHT * 5)

    # Color the edges
    grasp_scene.play(
        *[FadeToColor(l, EXPLORING_LINE_COLOR) for l in problem["EDGES_tmp_mobject"]],
        FadeIn(original_cost_text)
    )

    grasp_scene.wait()

    assert len(problem["DOTS_visited"]) == len(problem["EDGES_tmp_mobject"])
    for (i,edge) in enumerate(problem["EDGES_tmp_mobject"]):
        def the_updater(edge, i=i):
            edge.become(Line(
                problem["DOTS_mobject"][problem["DOTS_visited"][i]].get_center(),
                problem["DOTS_mobject"][problem["DOTS_visited"][(i+1)%len(problem["DOTS_visited"])]].get_center(),
                color=EXPLORING_LINE_COLOR
            ))
        edge.add_updater(the_updater)

    slow_iterations = 2
    n_neighborhood = 0
    ORIGINAL_NEW_SHIFT = DOWN

    # Main loop while current solution is not yet optimal
    while not is_optimal:
        n_neighborhood += 1
        if n_neighborhood > 1:
            grasp_scene.play(
                # Move the main after having found better solution
                *[line.animate.become(
                    Line(
                        problem["DOTS_mobject"][problem["DOTS_visited"][i]].get_center(),
                        problem["DOTS_mobject"][problem["DOTS_visited"][(i+1)%len(problem["DOTS_visited"])]].get_center(),
                        color = MAIN_LINE_COLOR
                    )
                ) for (i,line) in enumerate(problem["EDGES_main_mobject"])],

                # Update original cost and new cost
                original_cost_text.animate.become(Text(f"{solution_cost:.2f}", color=VISITED_DOT_COLOR).scale(1/2).shift(DOWN * 3 + RIGHT * 5)),
                FadeOut(new_cost_text),
                run_time = 1 if slow_iterations >= 0 else 1/3
            )

        is_optimal = True

        # Iterate over permutations 
        for i in range(len(solution)):
            slow_iterations -= 1

            new_solution = solution.copy()
            new_solution[i], new_solution[(i+1) % len(new_solution)] = new_solution[(i+1) % len(new_solution)], new_solution[i]
            new_cost = get_total_cost(new_solution)

            # Save time only showing the whole animation in first neighborhood (n_neighborhood==1)
            if n_neighborhood==1 or (new_cost < solution_cost):
                new_cost_text = Text(f"{new_cost:.2f}", color=EXPLORING_LINE_COLOR).scale(1/2).next_to(original_cost_text, ORIGINAL_NEW_SHIFT)
                # Do the permutation
                grasp_scene.play(
                    problem["DOTS_mobject"][problem["DOTS_visited"][i]].
                        animate.move_to(problem["DOTS_mobject"][problem["DOTS_visited"][(i+1)%len(problem["DOTS_mobject"])]].get_center()),
                    problem["DOTS_mobject"][problem["DOTS_visited"][(i+1)%len(problem["DOTS_mobject"])]].
                        animate.move_to(problem["DOTS_mobject"][problem["DOTS_visited"][i]].get_center()),
                    FadeIn(new_cost_text),
                    run_time = 1 if slow_iterations >= 0 else 1/3
                )

                if slow_iterations >= 0:
                    grasp_scene.wait()

                # Compare solutions
                if new_cost < solution_cost:
                    # Using first-fit or best-fit
                    solution, solution_cost = new_solution.copy(), new_cost
                    is_optimal = False
                    if n_neighborhood==1:
                        grasp_scene.wait()
                    break

                # Return animation to prev state: move edges to original position
                grasp_scene.play(
                    problem["DOTS_mobject"][problem["DOTS_visited"][i]].
                        animate.move_to(problem["DOTS_mobject"][problem["DOTS_visited"][(i+1)%len(problem["DOTS_mobject"])]].get_center()),
                    problem["DOTS_mobject"][problem["DOTS_visited"][(i+1)%len(problem["DOTS_mobject"])]].
                        animate.move_to(problem["DOTS_mobject"][problem["DOTS_visited"][i]].get_center()),
                    FadeOut(new_cost_text),
                    run_time = 1 if slow_iterations >= 0 else 1/3
                )
    
    for (i,edge) in enumerate(problem["EDGES_tmp_mobject"]):
        edge.clear_updaters()

    problem["DOTS_VISITED_search"].clear()
    problem["DOTS_VISITED_search"].extend(solution.copy())

    grasp_scene.play(
        FadeOut(code_img),
        *[FadeOut(e) for e in problem["EDGES_tmp_mobject"]],
        *[FadeOut(e) for e in problem["EDGES_main_mobject"]],
        *[FadeOut(d) for d in problem["DOTS_mobject"]],
        FadeOut(original_cost_text)
    )

    grasp_scene.wait()


"""ANIMATION: Displays graphs explainig how different values affect time and results"""
def explain_parameter_values(grasp_scene):
    # Exploration space
    x_values = [i/20 for i in range(21)]

    y_values = [
        # Alpha, best-fit
        [371.8, 383.0, 409.0, 435.4, 439.8, 445.8, 446.6, 450.6, 454.4, 460.4, 466.4, 469.4, 474.6, 479.8, 
        484.0, 491.0, 502.4, 502.4, 505.2, 508.6, 511.6],
        # Alpha, first-fit
        [712.626, 1051.872, 1314.65, 1520.61, 1629.814, 1676.84, 1711.762, 1709.992, 1799.788, 1843.382, 
        1908.316, 1943.17, 2050.55, 2132.592, 2180.584, 2278.868, 2477.104, 2581.67, 2669.234, 2751.74, 3060.692]
    ]

    for i in range(len(y_values)):
        y_values[i] = [int(val) for val in y_values[i]]

    axes = [
        Axes(
            x_range = [0, 1, 0.2] if i in [0,1] else [1, 20, 5],
            y_range = [0, max(y_values[i]), max(y_values[i])/5],
            tips = False,
            axis_config={"include_numbers": True}
        ).scale(2/5)
        for i in range(len(y_values))
    ]
    grouped = Group(axes[0], axes[1]).arrange(buff=1.5)

    titles_x = ["α", "α"]
    titles_x_text = [Text(titles_x[i]).scale(1/4).next_to(axes[i], DOWN) for i in range(len(axes))]

    titles_y = ["Soluciones exploradas (best-fit)", "Soluciones exploradas (first-fit)"]
    titles_y_text = [Text(titles_y[i]).scale(1/4).rotate(PI/2).next_to(axes[i], LEFT) for i in range(len(axes))]

    grasp_scene.play(
        *[Create(_) for _ in axes],
        *[Create(_) for _ in titles_x_text],
        *[Create(_) for _ in titles_y_text]
    )
    grasp_scene.wait()

    line_graphs = [
        axes[i].plot_line_graph(
            x_values = x_values,
            y_values = y_values[i],
            line_color = BLUE,
        )
        for i in range(len(axes))
    ]
    for lg in line_graphs:
        for d in lg["vertex_dots"]:
            d.scale(0.4)

    grasp_scene.play(
        *[Write(lg) for lg in line_graphs]
    )
    grasp_scene.wait()


    # Generated solutions
    x_values_3 = [i/20 for i in range(21)]
    y_values_3 = (
        # Alpha, min
        [175.30569440774562, 174.0577569441483, 209.1008958216621, 242.80537872674472, 296.22807910365066, 321.7270226686758,
        382.0762991315628, 418.75724849643, 463.2379460837739, 508.1351723986504, 560.914373587751, 574.9902551530935, 638.3334415766027,
        696.6702634411356, 702.5198689178217, 746.8927613710514, 821.7654263315491, 860.4991836545292, 862.7556022198372, 
        881.3866671044277, 903.1364299865438],
        # Alpha, avg
        [195.34661636635283, 215.92857993239903, 256.5670654132471, 305.9072814443351, 351.90764298836405, 392.62653833712557,
        443.2977043728582, 496.59624978262383, 551.8646667940421, 599.7569450615863, 647.1722965808575, 700.0633381329967, 
        748.4159993250053, 796.9061834648643, 846.4791104092621, 890.6299962842689, 932.3577893235419, 971.4950199088457, 
        1000.5633759197782, 1018.0601792453892, 1060.6032962146608],
        # Alpha, max
        [211.76310410253666, 252.02238146122815, 295.7042571754477, 346.7170363587187, 405.45556684380233, 459.4358524246897, 
        508.8764851122736, 570.8243938550233, 631.1469532311878, 677.4364034889157, 745.0211731938122, 812.7248849608063, 
        849.0806195502898, 920.9711017735721, 941.8674399288584, 992.6543706495888, 1082.4735378585679, 1087.9253063491976, 
        1134.2458929972856, 1162.3089731438708, 1187.041034650837]
    )

    axes_3 = [
        Axes(
            x_range = [0, 1, 0.2],
            y_range = [0, 1200, 200],
            tips = False,
            axis_config={"include_numbers": True}
        ).scale(3/5)
    ]

    titles_x_3 = ["α"]
    titles_x_text_3 = [Text(titles_x_3[i]).scale(1/3).next_to(axes_3[i], DOWN) for i in range(len(axes_3))]
    
    titles_y_3 = ["Solución construída"]
    titles_y_text_3 = [Text(titles_y_3[i]).scale(1/3).rotate(PI/2).next_to(axes_3[i], LEFT) for i in range(len(axes_3))]
    
    line_graphs_3 = [
        axes_3[i % len(axes_3)].plot_line_graph(
            x_values = x_values_3,
            y_values = y_values_3[i],
            line_color = [BLUE, WHITE, RED][i],
        )
        for i in range(len(y_values_3))
    ]

    for lg in line_graphs_3:
        for d in lg["vertex_dots"]:
            d.scale(0.4)
        
    grasp_scene.play(
        ReplacementTransform(axes[0], axes_3[0]),
        *[FadeOut(a) for a in axes[1:]],
        ReplacementTransform(titles_x_text[0], titles_x_text_3[0]),
        *[FadeOut(t) for t in titles_x_text[1:]],
        ReplacementTransform(titles_y_text[0], titles_y_text_3[0]),
        *[FadeOut(t) for t in titles_y_text[1:]],
        ReplacementTransform(line_graphs[0], line_graphs_3[0]),
        *[FadeOut(t) for t in line_graphs[1:]],
    )
    grasp_scene.play(
        *[Write(lg) for lg in line_graphs_3[1:]]
    )
    grasp_scene.wait()


    # Found solutions:
    x_values_2 = [i/20 for i in range(21)]
    y_values_2 = [
        # Alpha, best-fit, min
        [168.50160670605305, 163.26524848571154, 194.15894563003008, 219.88741702470622, 248.1739025989809, 285.4741417437415, 307.86171870918713, 
            351.17958326145214, 379.80450242623004, 399.41841895522623, 435.0087584123062, 461.0576212222607, 493.95605401910916, 515.9099984631524,
            558.5912989648706, 576.6230421590552, 603.677171880414, 607.7345072066798, 618.0881152777688, 618.0847211788407, 626.3431772338979],

        # Alpha, first-fit, min
        [171.68526972647282, 164.1225090476803, 195.87778597798132, 223.01502162098285, 256.69499021396473, 294.9835824610303, 332.5157664341469, 
            361.372418494939, 406.35542882465114, 439.9878253624741, 465.4852706730682, 486.6204279613743, 548.2805360738532, 552.9419029252903, 
            611.8304607543159, 634.3108603333178, 657.2549865717139, 683.7567004674659, 683.9006495467665, 692.2257871348861, 690.1799257130817],

        # Alpha, best-fit, avg
        [191.03831482729024, 201.90595040251395, 232.45576149342793, 269.8107671423632, 304.2448891058208, 336.61880015442364, 377.5955063630883,
        420.11360297666164, 460.48951382578986, 495.16884310407397, 528.079809510834, 566.3344607014195, 595.7586250096194, 625.1856655276769, 
        655.6145999185944, 679.9303946489774, 695.87813216888, 711.3558314441881, 721.8307968816777, 724.4600270545632, 731.6981218340487],

        # # Alpha, first-fit, avg
        [192.35364322811432, 204.89346866468478, 237.31876655164774, 277.5540055718985, 314.2710783030643, 349.26040759210986, 392.110108316167,
        438.2874357179185, 483.29498118825416, 522.5480495909779, 559.1361919428352, 602.2798880089982, 637.468255381899, 671.3298750151749,
        708.1799995109075, 738.0906914711593, 758.241273285356, 779.1796480620826, 792.7237461559469, 801.8649560815043, 815.2798775386423],

        # Alpha, best-fit, max
        [207.9313823810573, 234.39121903382764, 277.07871510664904, 310.35812121626094, 353.1156583584607, 402.2239294985039, 445.16275370808086,
        502.9754039866448, 535.5284994162733, 574.3016494627213, 611.8742419279469, 688.8107920739916, 699.4613332567577, 730.262222414041, 762.736261772604,
        791.709765078189, 828.6464208752834, 830.9670576658217, 828.552677044864, 844.0708993175447, 847.6969642525652],

        ## Alpha, first-fit, max
        [208.78466555476865, 241.27253976944158, 280.5219994505575, 317.590404971618, 362.1589938367566, 410.27502974417234, 461.70754853019383,
        516.3311576962503, 570.3774795362199, 604.0603308679256, 646.0479986157301, 729.5769744770848, 734.0473696975118, 775.5373342262163, 814.101013228809,
         863.3451146941735, 859.2233154138246, 872.6852054760211, 887.465495059633, 926.1314430097838, 913.8945015649533]
    ]

    axes_2 = [
        Axes(
            x_range = [0, 1, 0.2],
            y_range = [0, 700, 100],
            tips = False,
            axis_config={"include_numbers": True}
        ).scale(2/5)
        for i in range(2)
    ]
    grouped = Group(axes_2[0], axes_2[1]).arrange(buff=1.5)

    titles_x_2 = ["α", "α"]
    titles_x_text_2 = [Text(titles_x_2[i]).scale(1/4).next_to(axes_2[i], DOWN) for i in range(len(axes_2))]

    titles_y_2 = ["Soluciones generadas (best-fit)", "Soluciones generadas (first-fit)"]
    titles_y_text_2 = [Text(titles_y_2[i]).scale(1/4).rotate(PI/2).next_to(axes_2[i], LEFT) for i in range(len(axes_2))]

    line_graphs_2 = [
        axes_2[i % len(axes_2)].plot_line_graph(
            x_values = x_values_2,
            y_values = y_values_2[i],
            line_color = [BLUE, WHITE, RED][i//len(axes_2)],
        )
        for i in range(len(y_values_2))
    ]
    for lg in line_graphs_2:
        for d in lg["vertex_dots"]:
            d.scale(0.4)

    grasp_scene.play(
        ReplacementTransform(axes_3[0], axes_2[0]),
        *[Create(a) for a in axes_2[1:]],

        ReplacementTransform(titles_x_text_3[0], titles_x_text_2[0]),
        *[Create(a) for a in titles_x_text_2[1:]],

        ReplacementTransform(titles_y_text_3[0], titles_y_text_2[0]),
        *[Create(a) for a in titles_y_text_2[1:]],

        *[ReplacementTransform(line_graphs_3[i], line_graphs_2[i*len(axes_2)]) for i in range(len(line_graphs_3))],
        *[Write(line_graphs_2[i]) for i in range(1, len(line_graphs_2), 2)],
    )
    grasp_scene.wait()

    grasp_scene.play(
        *[FadeOut(_) for _ in axes_2],
        *[FadeOut(_) for _ in titles_x_text_2],
        *[FadeOut(_) for _ in titles_y_text_2],
        *[FadeOut(_) for _ in line_graphs_2]
    )
    grasp_scene.wait()


"""ANIMATION: Displays a list of extensions to the algorithm"""
def show_enhancements(grasp_scene):
    enhancement_list = {
        "Construcción":
            ["• Random Plus Greedy Construction",
            "• Sampled Greedy Construction",
            "• Restricciones relajadas + Reparación",
            "• Reactive GRASP",
            "• Cost perturbations",
            "• Memoria y aprendizaje"],
        "Path-relinking": None
    }

    to_add = []
    for (k,v) in enhancement_list.items():
        if len(to_add) == 0:
            to_add.append(Text(k).shift(UP*2))
        else:
            to_add.append(Text(k).next_to(to_add[-1], DOWN).shift(DOWN * 0.75))

        if v is not None:
            to_add.append(Paragraph(*v).scale(1/2).next_to(to_add[-1], DOWN))


    grasp_scene.play(*[FadeIn(_) for _ in to_add])
    grasp_scene.wait()
    grasp_scene.play(*[FadeOut(_) for _ in to_add])
    grasp_scene.wait()


"""ANIMATION: Displays the end of the video (member names and references)"""
def show_final_credits(grasp_scene):
    # Show references

    references_str = [
        r"[1]	‘The noising method: a new method for combinatorial optimization’, Operations Research Letters, vol. 14, no. 3, pp. 133–137, 1993.",
        r"[2]	A. R. Duarte, C. C. Ribeiro, and S. Urrutia, ‘A Hybrid ILS Heuristic to the Referee Assignment Problem with an Embedded MIP Strategy’, Hybrid Metaheuristics. Springer Berlin Heidelberg, pp. 82–95, 2007.",
        r"[3]	‘A probabilistic heuristic for a computationally difficult set covering problem’, Operations Research Letters, vol. 8, no. 2, pp. 67–71, 1989.",
        r"[4]	‘Improved Constructive Multistart Strategies for the Quadratic Assignment Problem Using Adaptive Memory’, INFORMS Journal on Computing, vol. 11, pp. 198–204, 05 1999.",
        r"[5]	‘Expanding Neighborhood GRASP for the Traveling Salesman Problem’, Computational Optimization and Applications, vol. 32, pp. 231–257, 12 2005.",
        r"[6]	‘GRASP with path-relinking for the generalized quadratic assignment problem’, Journal of Heuristics, 2011.",
        r"[7]	‘A greedy randomized adaptive search procedure application to solve the travelling salesman problem’, International Journal of Industrial Engineering and Management, vol. 10, pp. 238–242, 09 2019.",
        r"[8]	‘Semi-greedy heuristics: An empirical study’, Operations Research Letters, vol. 6, no. 3, pp. 107–114, 1987.",
        r"[9]	‘Reactive GRASP: An Application to a Matrix Decomposition Problem in TDMA Traffic Assignment’, INFORMS Journal on Computing, 2000.",
        r"[10]	‘Greedy Randomized Adaptive Search Procedures’, in Handbook of Metaheuristics, Springer US, 2003, pp. 219–249.",
        r"[11]	‘A Hybrid Heuristic for the p-Median Problem’, Journal of Heuristics, 2004."
    ]

    group = VGroup(*[Text(t).scale(1/3) for t in references_str])
    group.arrange(DOWN, center=False, aligned_edge=LEFT, buff=0.2)
    group.move_to(ORIGIN)

    # FIXME: the text is outside of screen
    grasp_scene.play(FadeIn(group))
    grasp_scene.wait()
    grasp_scene.play(FadeOut(group))
    grasp_scene.wait()

    # Show member names
    title_GRASP = Text("GRASP").shift(UP*1)
    group_name = Paragraph("Grupo 1", "\tDavid González Fernández", "\tSergio Arroni del Riego", "\tJosé Manuel Lamas Pérez").scale(2/3).shift(DOWN*2)

    grasp_scene.play(
        LaggedStart(
            Write(title_GRASP),
            Write(group_name),
            lag_ratio = 0.25
        )
    )
    grasp_scene.wait()


"""Enhancements"""
class TheEnd(Scene):
    def construct(self):
        # Explain how different values of the parameters affect the run-time and results
        explain_parameter_values(self)
        
        # Mention extensions and modifications of the algorithms
        show_enhancements(self)

        # Show the final credits
        show_final_credits(self)



"""GRASP animation"""
class GRASP(Scene):
    def construct(self):
        # Show the name and meaning
        show_introduction(self)

        # Show the general code for GRASP
        image, rect = show_code_grasp(self)
        self.play(FadeOut(image), FadeOut(rect))
        self.wait()

        # Introduce the problem to solve
        initialize_problem()
        title = introduce_problem(self)
        self.wait()
        # Draw the built problem on the screen
        display_problem(self, title)
        self.wait()

        # Show the general code and the code for construction
        code_img = show_code_grasp_focus_construct(self)
        # Visualize construction
        construct_initial_solution(self, code_img)
        
        # Show the general code and the code for local search
        code_img = show_code_grasp_focus_search(self)
        # Visualize local search
        do_local_search(self, code_img)

        # Show the code to do a recap of the code
        explain_code_grasp(self)

