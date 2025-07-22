
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class SIR():
    def __init__(self, i0, beta, gamma, vaccination, dt):
        self.i0 = i0
        # Populatiom
        self.i = i0
        self.s = 1-i0
        self.r = 0
        
        # Constants
        self.beta = beta
        self.gamma = gamma
        self.v = vaccination
        # Time
        self.dt= dt
        self.t = 0
        
        # History
        self.history = []
    
    def step(self):
        # flows
        flow_s = -(self.beta * self.i * self.s) - (self.v * self.s)
        flow_i = (self.beta * self.i * self.s) - self.gamma* self.i
        flow_r =  (self.gamma* self.i) + (self.v * self.s)
        
        # Stock Update
        self.s+= flow_s*self.dt
        self.i+= flow_i*self.dt
        self.r+= flow_r*self.dt
        self.t+=self.dt
        
        # Save to History
        self.history.append(
            {"t": self.t, "S": self.s, "I": self.i, "R":self.r}
        )
        
    def sim(self, t_end):
        while(self.t <= t_end):
            self.step()
            
    def restart(self):
        self.t = 0
        self.i = self.i0
        self.s = 1-self.i
        self.r = 0
        self.history = []
            
    def sim_multiple(self, v_options ,t_end,do_plot=True, cols=2):
        
        histories = {}
        for i, v in enumerate(v_options):
            self.restart()
            self.v = v
            self.sim(t_end)
            histories[v] = [*self.history]
            
        if (do_plot):
            height = int(np.ceil(len(v_options)/cols))
            fig, ax = plt.subplots(height, cols)
            
            for x,v_ in enumerate(v_options):
                df = pd.DataFrame(histories[v_])
                i = int(x/cols)
                j = x%cols
                ax[i][j].set_title(f"v = {v_}")
                ax[i][j].plot(df["t"], df["S"], label="Susceptibles", color='b')
                ax[i][j].plot(df["t"], df["I"], label="Infectados", color='r')
                ax[i][j].plot(df["t"], df["R"], label="Recuperados", color='g')
            fig.tight_layout()
            fig.suptitle(f"Modelo SIR con Vacunacion\nβ = {self.beta}, γ = {self.gamma}, I0 = {self.i0}")
            fig.tight_layout(rect=[0, 0, 1, 0.95]) 
            plt.show()
        
    def plot(self):
        df = pd.DataFrame(self.history)
        plt.suptitle(f"Modelo SIR con Vacunacion\nβ = {self.beta}, γ = {self.gamma}, v = {self.v}")
        plt.plot(df["t"], df["S"], label="Susceptibles", color='b')
        plt.plot(df["t"], df["I"], label="Infectados", color='r')
        plt.plot(df["t"], df["R"], label="Recuperados", color='g')
        plt.ylabel("Proporcion de Poblacion")
        plt.xlabel("Dia")
        plt.legend()
        plt.show()
        
        
def ex1():
    # 1. β = 0,3, γ = 0,1, v = 0,05
    model = SIR(
        i0=0.05,
        beta=0.3,
        gamma=0.1,
        vaccination=0.00,
        dt=0.5  
    ) 
    # Sim por 180 dias
    model.sim(180)
    model.plot()

# 2.
def ex2():
    model = SIR(
        i0=0.05,
        beta=0.3,
        gamma=0.1,
        vaccination=0.00,
        dt=0.5  
    ) 
    v_options = [0, 0.3, 0.385, 0.4]
    model.sim_multiple(v_options, 180)