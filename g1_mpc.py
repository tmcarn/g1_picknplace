import numpy as np
import mujoco
import mujoco.viewer
import cvxpy as cp

import os
from collections import defaultdict
import time

class SimpleMPCBalance:
    def __init__(self, mpc_frq=50):

        self.R_b = None

        # MPC Parameters
        self.mpc_frq = mpc_frq
        self.dt = 1 / self.mpc_frq
        self.horizon = 20 # steps

        # State weights (from paper)
        self.Q = np.diag([
            1500,
            2000,
            1000,
            1000, 
            1000,
            1000,
            1, 
            3, 
            10, 
            1, 
            1, 
            1, 
            1, 
            1, 
            1
        ])
        
        # Decrease R to allow more aggressive control
        self.R = np.diag([
            1, # F1
            1, 
            1, 
            1, # F2
            1,
            1,
            5, # M1
            5, 
            5, # M2
            5, 
            0, # F_ext
            0,
            0
        ]) * 10e-4

    def state_transition_model(self, x):
        '''
        x: [theta, p_c, omega, p_c_dot, g] (g is dummie variable)
        u: [F_1, F_2, M_1, M_2, F_ext] shape: (13,)

        Input: x --> (15,) 
        Output: dx/dt --> (15,)
        '''
        theta = x[:3]
        roll, pitch, yaw = theta

        A = np.zeros((15,15))

        c_theta = np.cos(pitch)
        s_theta = np.sin(pitch)
        c_psi = np.cos(yaw)
        s_psi = np.sin(yaw)
        
        # Roll Not Included because M_x is ignored
        self.R_b = np.array([
            [c_theta * c_psi,  -s_psi,  0],
            [c_theta * s_psi,   c_psi,  0],
            [-s_theta,          0,      1]
        ])

        A[:3, 6:9] = self.R_b
        A[3:6, 9:12] = np.eye(3)
        A[9:12, 12:] = np.eye(3)

        B = np.zeros((15,13))

        # Linear Forces Equation (3, 13)
        F = np.zeros((3,13))
        F[:,:3] = np.eye(3)
        F[:,3:6] = np.eye(3)
        F = F/self.box_mass
        
        
        # Moment Equations
        M = np.zeros((3, 13))

        L = np.array([[0,0],
                      [1,0],
                      [0,1]])
        
        distance_vectors = self.get_distance_vectors()
        
        r1x = self.skew_symmetric(distance_vectors[0])
        r2x = self.skew_symmetric(distance_vectors[1])
        r_ex = self.skew_symmetric(distance_vectors[2])

        I_G = self.get_inertia_tensor()
        I_G_inv = np.linalg.inv(I_G)

        M[:, :3] = I_G_inv @ r1x      # Effect of F1
        M[:, 3:6] = I_G_inv @ r2x     # Effect of F2
        M[:, 10:] = I_G_inv @ r_ex    # Effect of F_ext
        M[:, 6:8] = I_G_inv @ L       # Effect of M1
        M[:, 8:10] = I_G_inv @ L      # Effect of M2

        # Add to Matrix
        B[6:9, :] = M      
        B[9:12, :] = F     

        # Linearized State Transition Function
        return A, B
    
    def compute_optimal_control(self):
        x0 = self.get_state()
        A_c, B_c = self.state_transition_model(x0) # Gets the continous state transition matrices based on x_0
        
        # Convert to Discrete Time
        A_d = np.eye(A_c.shape[0]) + (A_c * self.dt)
        B_d = B_c * self.dt

        # Decision variables for each step in horizon
        x = [cp.Variable(15) for _ in range(self.horizon + 1)] # Included initial state
        u = [cp.Variable(13) for _ in range(self.horizon)]

        # Cost Function
        cost = 0
        constraints = []

        # Initial state constraint
        constraints.append(x[0] == x0)

        for k in range(self.horizon):
            # Cost
            x_error = x[k] - self.x_desired
            cost += cp.quad_form(x_error, self.Q)
            cost += cp.quad_form(u[k], self.R)

            # ===== DYNAMIC CONSTRAINTS =====
            constraints.append(x[k+1] == A_d @ x[k] + B_d @ u[k])

            # ===== FOOT 1 CONSTRAINTS =====
            # Normal force limits: F_min <= F1z <= F_max
            constraints.append(u[k][2] >= self.F_min)
            constraints.append(u[k][2] <= self.F_max)
            
            # Friction cone: -μ*F1z <= F1x <= μ*F1z
            constraints.append(u[k][0] >= -self.mu * u[k][2])
            constraints.append(u[k][0] <= self.mu * u[k][2])
            
            # Friction cone: -μ*F1z <= F1y <= μ*F1z
            constraints.append(u[k][1] >= -self.mu * u[k][2])
            constraints.append(u[k][1] <= self.mu * u[k][2])
            
            # ===== FOOT 2 CONSTRAINTS =====
            # Normal force limits: F_min <= F2z <= F_max
            constraints.append(u[k][5] >= self.F_min)
            constraints.append(u[k][5] <= self.F_max)
            
            # Friction cone: -μ*F2z <= F2x <= μ*F2z
            constraints.append(u[k][3] >= -self.mu * u[k][5])
            constraints.append(u[k][3] <= self.mu * u[k][5])
            
            # Friction cone: -μ*F2z <= F2y <= μ*F2z
            constraints.append(u[k][4] >= -self.mu * u[k][5])
            constraints.append(u[k][4] <= self.mu * u[k][5])

            # ===== EXTERNAL FORCE CONSTRAINTS =====
            constraints.append(u[k][10:] == self.gravity * self.box_mass)

            # ===== MOMENT BOUNDING CONSTRAINTS =====
            constraints.append(u[k][6] >= -self.M_max)
            constraints.append(u[k][6] <= self.M_max)
            constraints.append(u[k][7] >= -self.M_max)
            constraints.append(u[k][7] <= self.M_max)
            constraints.append(u[k][8] >= -self.M_max)
            constraints.append(u[k][8] <= self.M_max)
            constraints.append(u[k][9] >= -self.M_max)
            constraints.append(u[k][9] <= self.M_max)



        
        # TERMINAL COST
        x_error_final = x[self.horizon] - self.x_desired
        cost += cp.quad_form(x_error_final, self.Q * 10) # Terminal cost is emphasized
        
        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)

        problem.solve(solver=cp.OSQP, verbose=False)

        print("MPC Complete")

        return u[0].value
    
    def calculate_q_ddot_des(self):
        x = self.get_state()
        
        theta_curr = x[0:3]
        p_com_curr = x[3:6]
        omega_curr = x[6:9]
        p_com_dot_curr = x[9:12]
        
        # Desired state (from MPC)
        theta_des = self.x_desired[0:3]
        p_com_des = self.x_desired[3:6]
        
        # MUCH LOWER GAINS - Your gains are WAY too high!
        Kp_com = 5.0       # Down from 100
        Kd_com = 10.0      # Down from 50
        Kp_orient = 2.0    # Down from 50
        Kd_orient = 5.0    # Down from 20
        
        # Errors (for debugging)
        e_com = p_com_des - p_com_curr
        e_theta = theta_des - theta_curr
        
        # PD control (zero desired velocities)
        p_ddot_des = Kp_com * e_com - Kd_com * p_com_dot_curr
        omega_ddot_des = Kp_orient * e_theta - Kd_orient * omega_curr
        
        # Add joint damping
        q_dot = self.data.qvel.copy()
        joint_damping = -20.0 * q_dot  # Increased from -10
        
        # Full state
        q_ddot_des = np.zeros(self.model.nv)
        q_ddot_des[0:3] = p_ddot_des
        q_ddot_des[3:6] = omega_ddot_des
        q_ddot_des += joint_damping
        
        # CRITICAL: CLIP TO PREVENT EXPLOSION
        max_accel = 2.0  # rad/s² or m/s²
        q_ddot_des = np.clip(q_ddot_des, -max_accel, max_accel)
        
        print(f"COM Error: {e_com}")
        print(f"Theta Error: {e_theta}")
        print(f"COM_VEL Error: {p_com_dot_curr}")
        print(f"Omega Error: {omega_curr}")
        
        return q_ddot_des
 
        # Get mass matrix
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M