from scipy.optimize import minimize
import numpy as np
from copy import copy

class MinimizeWrapper():
    def __init__(self, threshold=None):
        # will store optimization path (params and values)
        self.optimizationPath = []
        self.objectiveFunction = None

        # will terminate the optimization once below threshold
        self.threshold = threshold
        self.threshold_reached = False


    def minimize(self, fun, x0, args=(), method=None, jac=None, hess=None,
                 hessp=None, bounds=None, constraints=(), tol=None,
                 callback=None, options=None):
        self.optimizationPath = []
        self.objectiveFunction = fun
        import time
        t1 = time.time()

        minimizationResult = minimize(self.wrapObjectiveFunction, x0, args=args, method=method, jac=jac, hess=hess,
                 hessp=hessp, bounds=bounds, constraints=constraints, tol=tol,
                 callback=callback, options=options)

        t2 = time.time()
        minimizationResult.optimizationPath = self.optimizationPath
        index = np.argmin(np.array(self.optimizationPath, dtype=object)[:,1])
        minimizationResult.bestValue = copy(self.optimizationPath[index])
        minimizationResult.bestIsIntermediate = True if index != len(self.optimizationPath)-1 else False
        minimizationResult.optimizationTime = t2-t1
        return minimizationResult


    def wrapObjectiveFunction(self, x0, *args):
        if self.threshold_reached:
            # cause minimization to stop since no more changes occur
            return -1

        result = self.objectiveFunction(x0, *args)
        self.optimizationPath.append([list(x0), result])
        if self.threshold and result < self.threshold:
            self.threshold_reached = True
        return result
