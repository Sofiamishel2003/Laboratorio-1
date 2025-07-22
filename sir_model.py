
class SIR():
    def __init__(self, N, i0, beta, gamma, dt):
        # Populatiom
        self.N = N
        self.i = i0
        self.s = 1-i0
        self.r = 0
        
        # Constants
        self.beta = beta
        self.gamma = gamma
        
        # Time
        self.dt= dt
        self.t = 0
        
        # History
        self.history = []
    
    def step(self):
        # flows
        flow_s = -self.beta * self.i * self.s
        flow_i = (self.beta * self.i * self.s) - self.gamma* self.i
        flow_r =  self.gamma* self.i
        
        # Stock Update
        self.s+= flow_s*self.dt
        self.i+= flow_i*self.dt
        self.r+= flow_r*self.dt
        selft+=self.dt
        
        # Save to History
        self.history.append(
            {"t": self.t, "S": self.s, "I": self.i, "R":self.r}
        )
        
    def sim(self, n):
        while(self.t <= n):
            self.step()
        


# 1. β = 0,3, γ = 0,1, v = 0,05


        