import time
import numpy as np
from typing import Callable, Optional, List, Tuple
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult, OptimizerSupportLevel

class ISOMAOptimizer(Optimizer):
    """
    iSOMA Optimizer using the Self-Organizing Migrating Algorithm (SOMA).

    The iSOMA (improved Self-Organizing Migrating Algorithm) is an advanced version of SOMA,
    utilizing an adaptive PRT vector based on the maximum number of function evaluations (Max_FEs).

    References:

    - Diep, Q.B., Zelinka, I., Das, S., & Senkerik, R. "SOMA T3A for solving the 100-digit challenge."

    Swarm, Evolutionary, and Memetic Computing and Fuzzy and Neural Computing, Springer (2019): 155-165.

    - Zelinka, I. "SOMA—self-organizing migrating algorithm."

    New Optimization Techniques in Engineering, Springer (2004): 167-217.

    - Zelinka, I. "SOMA—self-organizing migrating algorithm."

    Self-Organizing Migrating Algorithm: Methodology and Implementation (2016): 3-49.

    """

    def __init__(self, 
                 N_jump: int = 10, 
                 Step: float = 0.3, 
                 PopSize: int = 150, 
                 Max_Migration: int = 10000, 
                 Max_FEs: Optional[int] = None,
                 VarMin: float = -500.0, 
                 VarMax: float = 500.0, 
                 m: int = 10, 
                 n: int = 5, 
                 k: int = 15,
                 file_path: Optional[str] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None):
        
        super().__init__()
        
        self.N_jump = N_jump
        self.Step = Step
        self.PopSize = PopSize
        self.Max_Migration = Max_Migration
        self.Max_FEs = Max_FEs
        self.VarMin = VarMin
        self.VarMax = VarMax
        self.m = m
        self.n = n
        self.k = k
        self.callback = callback
        self.best_cost_history = []
        self.best_params_history = []
        self.file_path = file_path

    def get_support_level(self):
        """Specify optimizer properties (support level)."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.required,
            "initial_point": OptimizerSupportLevel.required,
        }

    def _isoma(self, fun: Callable, x0: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> OptimizerResult:
        """Internal method performing the iSOMA optimization."""
        dimension = len(x0)
        N_jump = self.N_jump
        Step = self.Step
        PopSize = self.PopSize
        Max_Migration = self.Max_Migration
        Max_FEs = self.Max_FEs if self.Max_FEs is not None else dimension * 14**4
        m = self.m
        n = self.n
        k = self.k

        starttime = time.time()

        # Initialize population bounded by vectors
        pop = lower_bounds[:, None] + np.random.rand(dimension, PopSize) * (upper_bounds[:, None] - lower_bounds[:, None])
        
        # Replaced apply_along_axis with list comprehension for efficiency
        fitness = np.array([fun(pop[:, i]) for i in range(PopSize)])
        FEs = PopSize
        
        best_idx = np.argmin(fitness)
        the_best_cost = fitness[best_idx]
        the_best_value = pop[:, best_idx].copy()

        best_cost_old = the_best_cost
        Migration, Count = 0, 0

        while FEs < Max_FEs and Migration < Max_Migration:
            Migration += 1

            # Select m migrants
            M = np.random.choice(range(PopSize), m, replace=False)
            M_sort = np.argsort(fitness[M])

            for j in range(n):
                migrant_idx = M[M_sort[j]]
                Migrant = pop[:, migrant_idx].reshape(dimension, 1)

                # Select k candidates for leader
                K = np.random.choice(range(PopSize), k, replace=False)
                K_sort = np.argsort(fitness[K])
                
                # Fix: 0-indexed leader selection
                Leader = pop[:, K[K_sort[0]]].reshape(dimension, 1)
                if migrant_idx == K[K_sort[0]]:
                    Leader = pop[:, K[K_sort[1]]].reshape(dimension, 1)

                flag, move = 0, 1
                while flag == 0 and move <= N_jump:
                    nstep = (N_jump - move + 1) * Step
                    PRT = 0.1 + 0.9 * (FEs / Max_FEs)
                    PRTVector = (np.random.rand(dimension, 1) < PRT) * 1

                    offspring = Migrant + (Leader - Migrant) * nstep * PRTVector
                    
                    # Fix: Vectorized bounds clipping
                    offspring = np.clip(offspring, lower_bounds[:, None], upper_bounds[:, None])
                    new_cost = fun(offspring.flatten())
                    FEs += 1

                    if new_cost <= fitness[migrant_idx]:
                        flag = 1
                        fitness[migrant_idx] = new_cost
                        pop[:, migrant_idx] = offspring.flatten()
                        
                        if new_cost <= the_best_cost:
                            the_best_cost = new_cost
                            the_best_value = offspring.flatten().copy()
                            Count = 0  # Reset stagnation counter on global improvement
                    move += 1

                # Fix: Increment Count only if the migrant failed to improve at all
                if flag == 0:
                    Count += 1

            # Fix: Stagnation reset baseline logic
            if Count > PopSize * 50:
                if the_best_cost >= best_cost_old:
                    rat = round(0.1 * PopSize)
                    pop_temp = lower_bounds[:, None] + np.random.rand(dimension, rat) * (upper_bounds[:, None] - lower_bounds[:, None])
                    fit_temp = np.array([fun(pop_temp[:, i]) for i in range(rat)])
                    FEs += rat
                    D = np.random.choice(range(PopSize), rat, replace=False)
                    pop[:, D] = pop_temp
                    fitness[D] = fit_temp
                
                best_cost_old = the_best_cost  # Always update baseline upon check
                Count = 0

            self.best_cost_history.append(the_best_cost)
            self.best_params_history.append(the_best_value.copy())

            if self.file_path:
                with open(self.file_path, 'a') as f:
                    f.write(f"{Migration},{FEs},{the_best_value.tolist()},{the_best_cost}\n")

        caltime = time.time() - starttime

        result = OptimizerResult()
        result.x = the_best_value
        result.fun = the_best_cost
        result.nfev = FEs
        result.njev = 0  # Fix: Set to 0 instead of None to prevent downstream type errors
        result.nit = Migration
        result.time = caltime

        return result

    def minimize(self, 
                 fun: Callable, 
                 x0: np.ndarray, 
                 jac: Optional[Callable] = None, 
                 bounds: Optional[List[Tuple[float, float]]] = None, 
                 options: Optional[dict] = None) -> OptimizerResult:
        """Minimize the objective function using iSOMA optimization."""
        self.best_cost_history = []
        self.best_params_history = []
        # Fix: Parse Qiskit bounds parameter into arrays
        if bounds is not None:
            lower_bounds = np.array([b[0] if b[0] is not None else self.VarMin for b in bounds])
            upper_bounds = np.array([b[1] if b[1] is not None else self.VarMax for b in bounds])
        else:
            lower_bounds = np.full(len(x0), self.VarMin)
            upper_bounds = np.full(len(x0), self.VarMax)

        self._num_function_evaluations = 0

        def wrapped_fun(x):
            value = fun(x)
            self._num_function_evaluations += 1
            
            if self.callback:
                # Fix: Removed forced print to prevent I/O bottlenecks during optimization
                self.callback(self._num_function_evaluations, x, value, 0.0)
            
            return value
        
        result = self._isoma(wrapped_fun, x0, lower_bounds, upper_bounds)
        
        opt_result = OptimizerResult()
        opt_result.x = result.x
        opt_result.fun = result.fun
        # Use the internal tracker to account for FEs generated inside _isoma (like the initialization block)
        opt_result.nfev = result.nfev
        opt_result.njev = result.njev
        opt_result.nit = result.nit
        opt_result.message = 'Optimization finished'

        return opt_result