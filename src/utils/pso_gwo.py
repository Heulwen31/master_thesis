import numpy as np

class PSOGWO:
    def __init__(self, obj_func, bounds, pop_size=10, max_iter=20, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        # Swarm parameters (Dynamic w)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        
        # Initialize population
        self.pos = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (pop_size, self.dim))
        self.vel = np.zeros((pop_size, self.dim))
        
        # PSO personal bests
        self.pbest_pos = self.pos.copy()
        self.pbest_score = np.full(pop_size, np.inf)
        
        # GWO leaders
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = np.inf
        
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = np.inf
        
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = np.inf

    def optimize(self):
        for t in range(self.max_iter):
            # Dynamic Inertia Weight: w decreases linearly from w_max to w_min
            w = self.w_max - t * ((self.w_max - self.w_min) / self.max_iter)
            
            # Update fitness
            for i in range(self.pop_size):
                # Boundary check
                self.pos[i] = np.clip(self.pos[i], self.bounds[:, 0], self.bounds[:, 1])
                
                score = self.obj_func(self.pos[i])
                
                # Update PSO pbest
                if score < self.pbest_score[i]:
                    self.pbest_score[i] = score
                    self.pbest_pos[i] = self.pos[i].copy()
                
                # Update GWO Alpha, Beta, Delta (Proper Hierarchy)
                if score < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = score
                    self.alpha_pos = self.pos[i].copy()
                
                elif score < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = score
                    self.beta_pos = self.pos[i].copy()
                    
                elif score < self.delta_score:
                    self.delta_score = score
                    self.delta_pos = self.pos[i].copy()
            
            # Linear decrease of 'a' from 2 to 0 (GWO param)
            a = 2 - t * (2 / self.max_iter)
            
            for i in range(self.pop_size):
                # 1. GWO Position Update Logic
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha_pos - self.pos[i])
                X1 = self.alpha_pos - A1 * D_alpha
                
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta_pos - self.pos[i])
                X2 = self.beta_pos - A2 * D_beta
                
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta_pos - self.pos[i])
                X3 = self.delta_pos - A3 * D_delta
                
                X_gwo = (X1 + X2 + X3) / 3
                
                # 2. Hybridization with PSO Velocity
                r1, r2 = np.random.rand(), np.random.rand()
                self.vel[i] = (w * self.vel[i] + 
                               self.c1 * r1 * (self.pbest_pos[i] - self.pos[i]) + 
                               self.c2 * r2 * (X_gwo - self.pos[i]))
                
                self.pos[i] = self.pos[i] + self.vel[i]
                
        return self.alpha_pos, self.alpha_score
