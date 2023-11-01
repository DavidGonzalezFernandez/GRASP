from manim import *

def explain_code_grasp(self):
    image = show_code_grasp(self)
    line_height = 0.5

    rectangle_while = Rectangle(height=line_height, width=4.8).move_to(2.15*LEFT + 0.875*UP)
    rectangle_construct = Rectangle(height=line_height, width=5.4).move_to(1.14*LEFT + 0.425*UP)
    rectangle_whole_if_feasible = Rectangle(height=line_height*2 - 0.1, width=5.45).move_to(0.175*DOWN + 1.15*LEFT)
    rectangle_local_search = Rectangle(height=line_height, width=5.8).move_to(0.8*DOWN + 0.95*LEFT)

    # Focus con construct
    self.add(rectangle_construct)
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
    self.remove(image, rectangle_while)

def show_code_grasp(self):
    DOWN_SHIFT = 0
    image = ImageMobject("GRASP_code_snippet_v3.png").shift(DOWN_SHIFT * DOWN)

    # TODO do it more gently
    self.add(image)

    self.wait()
    return image

class ExplainConstructionOfCandidateList(Scene):
    def construct(self):
        # TODO: Show the name and meaning

        # TODO: Show the general code for GRASP
        show_code_grasp(self)
        self.wait()

        # TODO: Introduce the problem to solve

        # TODO: Show the general code for construction

        # TODO: Show the idea behind repair

        # TODO: Show local search

        # TODO: Show results and modifications of parameters

        # TODO: Show extensions and improvements