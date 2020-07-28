from heur import Heuristic, StopCriterion
import numpy as np


class FastSimulatedAnnealing(Heuristic):

    """
    Implementation of Fast Simulated Annealing heuristic
    """

    def __init__(self, of, maxeval, T0, n0, alpha, mutation, params=[1., 10, 80, 10]):
        """
        Initialization
        :param of: any objective function to be optimized
        :param maxeval: maximum allowed number of evaluations
        :param T0: initial temperature
        :param n0: cooling strategy parameter - number of steps
        :param alpha: cooling strategy parameter - exponent
        :param mutation: mutation to be used for the specific objective function (see heur_aux.py)
        """
        Heuristic.__init__(self, of, maxeval)

        self.T0 = T0
        self.n0 = n0
        self.alpha = alpha
        self.mutation = mutation
        self.params = params

    def search(self):
        """
        Core searching function
        :return: end result report
        """
        try:
            x = self.of.generate_point()
            params = [x, self.params[0], self.params[1], self.params[2], self.params[3]]
            f_x = self.evaluate(params)
            while True:
                k = self.neval - 1  # because of the first obj. fun. evaluation
                T0 = self.T0
                n0 = self.n0
                alpha = self.alpha
                # evaluate the current temperature
                T = T0 / (1 + (k / n0) ** alpha) if alpha > 0 else T0 * np.exp(-(k / n0) ** -alpha)

                y = self.mutation.mutate(x)
                params_mut = [y, self.params[0], self.params[1], self.params[2], self.params[3]]
                f_y = self.evaluate(params_mut)
                s = (f_y - f_x)/T
                swap = np.random.uniform() < 1/2 + np.arctan(s)/np.pi
                self.log({'step': k, 'x': x, 'f_x': f_x, 'y': y, 'f_y': f_y, 'T': T, 'swap': swap})
                if swap:
                    x = y
                    f_x = f_y

        except StopCriterion:
            return self.report_end()
        except:
            raise
