from physics.base import PhysicsProblem
import torch
import pandas as pd
import numpy as np
import scipy
import math
import time


class EMTMDProblem(PhysicsProblem):
    """
    EMTMD inverse design problem.
    Find the optimal values for the inductances (L) and resistances (R) of an electromechanical structure composed of
    9 electromechanical TMDs attached in a clamped-free rod, to minimize the H2 norm of the displacement at the
    free end of the rod. 
    """
        
    def __init__(self):
        self._targets = None
        self._input_data = None
        self.w_range = range(500,12501,5)                                # Frequency range
        df = pd.read_csv('data/Mem.csv',header=None)
        self.Mem = torch.tensor(df.values, dtype=torch.complex64)         # Mass matrix of the electromechanical system (109x109): 
        self.Mem = self.Mem                               # Inertia terms will be added later (Inductance)
        df = pd.read_csv('data/Cem.csv',header=None)
        self.Cem = torch.tensor(df.values, dtype=torch.complex64)         # Damping matrix of the electromechanical system (109x190)
        self.Cem = self.Cem                            # Damping terms will be added later (Resistance)
        df = pd.read_csv('data/Kem.csv',header=None)
        self.Kem = torch.tensor(df.values, dtype=torch.complex64)         # Stiffness matrix of the electromechanical system (118x118)
        self.Kem = self.Kem.unsqueeze(0)
        self.Force_vec = np.zeros((118,1), dtype = np.complex64)        # Excitation force vector
        self.Force_vec[3,0] = 1                                      
        self.Force_vec = torch.from_numpy(self.Force_vec).unsqueeze(0)
        self.s = 1j*2*math.pi*torch.tensor(list(self.w_range),dtype = torch.complex64).unsqueeze(-1).unsqueeze(-1)
        # Names for each design parameter: 1-9 values: Inductance (L), 10-18 values: Resistance (R)
        self.design_params = ['L1','L2','L3','L4','L5','L6','L7','L8','L9',
                              'R1','R2','R3','R4','R5','R6','R7','R8','R9'
        ]

        # Bounds for each design parameter
        self.bounds = [(1e-4, 0.22), (1e-4, 0.22), (1e-4, 0.22), (1e-4, 0.22), (1e-4, 0.22), (1e-4, 0.22), (1e-4, 0.22), (1e-4, 0.22), (1e-4, 0.22),
                       (1, 250), (1, 250), (1, 250), (1, 250), (1, 250), (1, 250), (1, 250), (1, 250), (1, 250)
        ]

    def get_input_output_dims(self):
        input_dim = 1
        output_dim = len(self.design_params)
        return input_dim, output_dim

    def get_bounds(self):
        return self.bounds

    def get_data_path(self):
        return 'data/emtmd_data.csv'

    def load_data(self, path):
        inp = torch.zeros((1,1))
        obs = torch.zeros((1,1))
        return inp, obs

    def forward_physics(self, inputs, predictions):
        output = self.compute_emtmd_response(predictions)
        return output

    def compute_emtmd_response(self, predictions):
        #start_time = time.perf_counter()
        # Set pi for convenience
        # predictions = torch.clamp(predictions, 1e-4, 1)
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18 = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3], predictions[:, 4], predictions[:, 5], predictions[:, 6], predictions[:, 7], predictions[:, 8], predictions[:, 9], predictions[:, 10], predictions[:, 11], predictions[:, 12], predictions[:, 13], predictions[:, 14], predictions[:, 15], predictions[:, 16], predictions[:, 17]
        #start_time = time.perf_counter()
        # Set up the system matrices
        Kem_l = self.Kem
        Force_l = self.Force_vec
        # Create the diagonal matrices for L and R values to be added to the Mem and Cem matrices
        L_mat = (torch.diag((torch.cat( (p1, p2, p3, p4, p5, p6, p7, p8, p9)))))
        R_mat = (torch.diag((torch.cat( (p10, p11, p12, p13, p14, p15, p16, p17, p18)))))
        Cem_l = torch.cat((torch.cat((self.Cem, torch.zeros(109,9) ),1), torch.cat( (torch.zeros(9,109),R_mat) ,1 )),0).unsqueeze(0)
        Mem_l = torch.cat((torch.cat((self.Mem, torch.zeros(109,9) ),1), torch.cat( (torch.zeros(9,109),L_mat) ,1 )),0).unsqueeze(0)
        
        DSM = Mem_l*self.s**2+Cem_l*self.s+Kem_l
        H = (torch.linalg.solve(DSM, Force_l))
        scale = 1e8
        output = torch.linalg.vector_norm(torch.abs(H[:,99,:]).squeeze(-1).squeeze(-1))*scale
        #end_time = time.perf_counter()
        #elapsed_time = end_time - start_time
        #print(f"Code executed in {elapsed_time:.4f} seconds")
        #output_np = H[:,99,:].detach().numpy()
        #df = pd.DataFrame(output_np)
        #df.to_csv("TestH.csv",index = False)
        print(output)

        return output

    def constraint_loss(self, predictions):
        return 0

    def save_results(self, history, epoch_results, output_dir, predictions, computed_output, inputs, targets):
        import os
        from visualization.plotting import plot_loss_curves, save_emtmd_epoch_results_csv, evaluate_rank

        if epoch_results is not None:
            epoch_csv_path = os.path.join(output_dir, 'epoch_results.csv')
            save_emtmd_epoch_results_csv(epoch_results, output_dir,history)

        plot_loss_curves(history, output_dir)

        import torch