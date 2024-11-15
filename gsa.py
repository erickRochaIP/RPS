from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.pso import PSO

from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

 
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.optimize import minimize
 
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
 
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
import numpy as np
from pymoo.termination import get_termination

from benchmark_functions import rosenbrock_function
 
n_threads = 4
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

xl, xu = -100, 100
n_var = 10

# f=funruido, x0=x0, max_avals=max_avals, lu=bounds

class ObjectiveFunction(ElementwiseProblem):
 
    def __init__(self, function, n_var, xl, xu, **kwargs):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu, vtype=float)
        self.function = function
 
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.function(x)

# problem = ObjectiveFunction(rosenbrock_function)

# algorithm = GA(pop_size=100,
#         sampling=FloatRandomSampling(),
#         crossover=SBX(prob=1.0, prob_var=1.0, eta=1.0), 
#         mutation=PolynomialMutation(prob=1.0, eta=0.9),
#         eliminate_duplicates=True)

# res = minimize(problem,
#                algorithm,
#                seed=1,
#                verbose=False,
#                termination = get_termination("n_eval", 300))

# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

def ga(f, x0=None, lu=None, max_avals=200,
       pop_size=100, prob_sbx=0.5, prob_var_sbx=0.5,
       eta_sbx=1.0, prob_mut = 0.5, eta_mut = 0.5):
    problem = ObjectiveFunction(f, len(lu), lu[0][0], lu[0][1])
    algorithm = GA(pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=prob_sbx, prob_var=prob_var_sbx, eta=eta_sbx), 
        mutation=PolynomialMutation(prob=prob_mut, eta=eta_mut),
        eliminate_duplicates=True
        )
    return minimize(
        problem,
        algorithm,
        get_termination("n_eval", max_avals),
        verbose = False
    )

def cmaes(f, x0=None, lu=None, max_avals=200,
       sigma=1, normalize=True, restarts=0, restart_from_best=True,
       incpopsize=1, eval_initial_x=True, noise_change_sigma_exponent=1,
       noise_kappa_exponent=0.2, bipop=True):
    problem = ObjectiveFunction(f, len(lu), lu[0][0], lu[0][1])
    algorithm = CMAES(x0=x0,
                      sigma=sigma,
                      normalize=normalize,
                      restarts=restarts,
                      restart_from_best=restart_from_best,
                      incpopsize=incpopsize,
                      eval_initial_x=eval_initial_x,
                      noise_change_sigma_exponent=noise_change_sigma_exponent,
                      bipop=bipop
                      )
    return minimize(
        problem,
        algorithm,
        get_termination("n_eval", max_avals),
        verbose = False
    )

def pso(f, x0=None, lu=None, max_avals=200,
        pop_size=5, w=1, c1=1, c2=1, adaptive=False,
        max_velocity_rate=1, pertube_best=False):
    problem = ObjectiveFunction(f, len(lu), lu[0][0], lu[0][1])
    algorithm = PSO(
        pop_size=pop_size,
        w=w,
        c1=c1,
        c2=c2,
        adaptive=adaptive,
        max_velocity_rate=max_velocity_rate,
        pertube_best=pertube_best
    )
    return minimize(
        problem,
        algorithm,
        get_termination("n_eval", max_avals),
        verbose = False
    )

# print(ga(rosenbrock_function, None, [(-100, 100)]*10, 500))