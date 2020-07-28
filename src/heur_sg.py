from heur import Heuristic, StopCriterion
import numpy as np


class ShootAndGo(Heuristic):

    """
    Implementation of generalized Shoot & Go heuristic
    """

    def __init__(self, of, maxeval, hmax=np.inf, params=[1., 10, 80, 10], random_descent=False, diameter=0.125*2):
        """
        Initialization
        :param of: any objective function to be optimized
        :param maxeval: maximum allowed number of evaluations
        :param hmax: maximum number of local improvements (0 = Random Shooting)
        :param random_descent: turns on random descent, instead of the steepest one (default)
        """
        Heuristic.__init__(self, of, maxeval)
        self.hmax = hmax
        self.random_descent = random_descent
        self.params = params
        self.diameter = diameter

    def steepest_descent(self, x):
        """
        Steepest/Random Hill Descent
        :param x: beginning point
        """
        desc_best_y = 0
        desc_best_x = x
        h = 0
        go = True
        while go and h < self.hmax:
            go = False
            h += 1

            neighborhood = self.of.get_neighborhood(desc_best_x, self.diameter)
            if self.random_descent:
                np.random.shuffle(neighborhood)

            for xn in neighborhood:
                params = [xn, self.params[0], self.params[1], self.params[2], self.params[3]]
                yn = self.evaluate(params) # trying the p parameter
                if yn > desc_best_y:
                    desc_best_y = yn
                    desc_best_x = xn
                    go = True
                    if self.random_descent:
                        break

    def search(self):
        """
        Core searching function
        :return: end result report
        """
        try:
            while True:
                # Shoot...
                x = self.of.generate_point()  # global search
                params = [x, self.params[0], self.params[1], self.params[2], self.params[3]]  # default n2v values except the p parameter
                self.evaluate(params)
                # ...and Go
                if self.hmax > 0:
                    self.steepest_descent(x)  # local search (optional)

        except StopCriterion:
            return self.report_end()
        except:
            raise
