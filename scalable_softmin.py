import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.optimize import minimize
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#######

class simulation:
    def __init__(self, optimal_decision_rule, Q0, VF, objective_function, x_distr, balanced = False, n=2000):
        self.O = optimal_decision_rule
        self.Q0 = Q0
        self.VF = VF
        self.objective_function = objective_function
        self.x_distr = x_distr
        self.X_dict = defaultdict(list)
        self.X_dict_train = {}
        self.X_dict_val = {}
        self.X_dict_test = {}
        self.X_scaled_dict = defaultdict(list)
        self.y_dict = {}
        self.rewards_dict = {}
        self.total_rewards_dict = {}
        self.optimal_dict = {}
        self.optimal_decisions_dict = {}
        self.propensity_scores = {i: 0.5 for i in range(n)}
        self.scaler = MinMaxScaler()
        self.n = n
        self.l = None
        self.S = None
        self.smooth_clip = False
        self.intercept = False
        self.penalty_norm = False
        self.miss_min = []
        self.beta_callback = []
        self.balanced = balanced
        
    def generate_data(self, *x_distr_args):
        np.random.seed(48)
        for i in range(self.n):
            if self.balanced:
                num_measurements = 4
            else:
                num_measurements = int(np.random.uniform(1, 5, 1)[-1])
            self.X_dict[i] = self.x_distr(*x_distr_args, size = (num_measurements, 3))
            self.y_dict[i] = np.sign(np.random.binomial(1, 0.5, num_measurements))
            self.optimal_dict[i] = np.array([self.O(x) for x in self.X_dict[i]])
            self.optimal_decisions_dict[i] = np.array([np.sign(x) for x in self.optimal_dict[i]])
            self.rewards_dict[i] = [np.random.normal(self.Q0(x, o, a), 1) 
                                   for x, o, a in zip(self.X_dict[i], self.optimal_decisions_dict[i], self.y_dict[i])]
            self.total_rewards_dict[i] = np.sum(self.rewards_dict[i])

        x_master = self.X_dict[0]
        for k in range(1, self.n):
            x_master = np.concatenate([x_master, np.array(self.X_dict[k])], axis = 0) 
        self.scaler.fit(x_master)   
        for k in self.X_dict:
            self.X_scaled_dict[k] = self.scaler.transform(np.array(self.X_dict[k]))

    
    def obtain_results(self, betas, X_dict_test, correct_min_count = None):
        X_dict_test_original = {k: self.X_dict[k] for k in X_dict_test}
        optimal_vf = self.VF(X_dict_test_original, X_dict_test_original, self.O)
        obs_vf = self.VF(X_dict_test_original, X_dict_test_original, lambda x, pid, idx: self.y_dict[pid][idx])
        assign_rate = round(np.sum([np.sum(self.optimal_decisions_dict[k] > 0) for k in self.optimal_decisions_dict]) /\
                    sum([len(o) for o in self.optimal_decisions_dict.values()]) * 100, 2)
            
        if not self.intercept:
            estimated_vf = self.VF(X_dict_test_original, X_dict_test, lambda x, pid, idx: np.dot(betas, x))
            success_rate = round(np.sum([np.all(np.sign(X_dict_test[k].dot(betas)) == self.optimal_decisions_dict[k]) 
                                     for k in X_dict_test]) / len(X_dict_test) * 100, 2)
        else:
            estimated_vf = self.VF(X_dict_test_original, X_dict_test, lambda x, pid, idx: np.dot(betas, np.hstack([1, x])))
            success_rate = round(np.sum([np.all(np.sign(np.concatenate([np.ones(shape=(X_dict_test[k].shape[0], 1)),
                                                       X_dict_test[k]], axis = 1).dot(betas)) == self.optimal_decisions_dict[k]) 
                                 for k in X_dict_test]) / len(X_dict_test) * 100, 2)
        min_accuracy = round(correct_min_count[1] / correct_min_count[0] * 100, 2) if correct_min_count is not None else np.nan
        result_df = pd.DataFrame({"l": [self.l],
                                  'S': [self.S],
                                  'MinAccuracy': [min_accuracy],
                                 "Assign_Rate": [assign_rate], 
                                 "Accuracy": [success_rate], 
                                 "OptimalVF": [optimal_vf],
                                 "EstimatedVF": [estimated_vf], 
                                 "ObservedVF": [obs_vf],
                                 'Betas': " ".join([str(round(b, 2)) for b in betas])})
        return result_df


    def optimize(self, l = 0.5, S=1, x0=None, verbose = False, scale = False, smooth_clip = True, 
                 lb = -1, up = 1, intercept = 0, penalty_norm = False):
        self.l, self.S, self.smooth_clip, self.intercept, self.penalty_norm = l, S, smooth_clip, intercept != 0, penalty_norm
        train_idx, test_idx = train_test_split(list(range(self.n)), test_size = 0.2, random_state = 48)
        train_idx, val_idx = train_test_split(train_idx, test_size = 0.2, random_state = 48)
        X_dict = self.X_dict if not scale else self.X_scaled_dict
        self.X_dict_train = {k: X_dict[k] for k in train_idx}
        self.X_dict_val = {k: X_dict[k] for k in val_idx}
        self.X_dict_test = {k: X_dict[k] for k in test_idx}
        
        x0 = np.ones(len(self.X_dict[0][0])) if x0 is None else x0
        x0 = np.hstack([intercept, x0]) if intercept != 0 else x0
        correct_min_count = [0, 0]
        self.miss_min = []
        self.beta_callback = [x0]
        sim_results = minimize(self.objective_function, x0=x0,
                              args = (self.X_dict_train, self.y_dict, self.total_rewards_dict, 
                                      self.propensity_scores, l, S, self.smooth_clip, 
                                      correct_min_count, self.miss_min, self.intercept, penalty_norm),
                              method = 'L-BFGS-B', bounds = [(lb, up) for i in range(len(x0))], callback = lambda x: self.beta_callback.append(x),
                              options = {'disp': True, 'maxiter': 15000, 'ftol':1e-09, 'gtol':1e-09, 
                                         'eps': 1e-08, 'maxls': 20, 'maxcor': 10, 'maxfun': 15000})
            
        betas = sim_results.x
        result_df = self.obtain_results(betas, self.X_dict_val, correct_min_count)
        if verbose:
            display(results_df)
        return result_df
    
    def model_test_results(self, betas, scale = False):
        test_results = self.obtain_results(betas, self.X_dict_test)
        return test_results
    
    def cross_validation(self, l = 0.5, s = 1, x0 = None, verbose = False, scale = False, smooth_clip = True):
        self.l, self.S, self.smooth_clip = l, S, smooth_clip
        kf = KFold(n_splits=5)
        cv_results = pd.DataFrame()
        X_dict = self.X_dict if not scale else self.X_scaled_dict
        x0 = np.ones(len(self.X_dict[0][0])) if x0 is None else x0
        self.miss_min = []
        for train_idx, test_idx in kf.split(list(range(self.n))):
            X_dict_train = {k: X_dict[k] for k in train_idx}
            X_dict_test = {k: X_dict[k] for k in test_idx}
            correct_min_count = [0, 0]
            sim_results = minimize(self.objective_function, x0 = x0, 
                               args = (X_dict_train, self.y_dict, self.total_rewards_dict, 
                                       self.propensity_scores, l, S, self.smooth_clip, correct_min_count, self.miss_min), 
                               method = 'BFGS', tol = 1e-3, options = {"maxiter": 1000})
            betas = sim_results.x
            cv_result = pd.concat([cv_result, self.obtain_results(betas, self.X_dict_test)], axis = 0)
        return cv_result, cv_result.mean().to_frame().transpose()
    
    
    def reset_data(self, n, x_distr):
        self.n = n
        self.x_distr = x_distr
        self.X_dict = defaultdict(list)
        self.X_dict_train = {}
        self.X_dict_val = {}
        self.X_dict_test = {}
        self.X_scaled_dict = defaultdict(list)
        self.y_dict = {}
        self.rewards_dict = {}
        self.total_rewards_dict = {}
        self.optimal_dict = {}
        self.optimal_decisions_dict = {}
        self.propensity_scores = {i: 0.5 for i in range(n)}
        self.scaler = MinMaxScaler()
        
    def reset_optimal_decision_rule(self, decision = None):
        self.O = decision
        
    def reset_VF(self, VF = None):
        self.VF = VF
        
    def reset_Q0(self, Q0 = None):
        self.Q0 = Q0
    
    def reset_objective_function(self, objective_funcion = None):
        self.objective_function = objective_function
        
    def reset_l(self, l = None):
        self.l = l
        
    def reset(self, S = None):
        self.S = S
        
    def reset_all(self, n, x_distr, decision = None, VF = None, Q0 = None, objective_funcion = None, l = None, S = None):
        self.clear_data(n, x_distr)
        self.reset_optimal_decision_rule(decision)
        self.reset_VF(VF)
        self.reset_Q0(Q0)
        self.reset_objective_function(objective_funcion)
        self.reset_l(l)
        self.reset_S(S)
        self.smooth_clip = False
        self.intercept = False
        self.penalty_norm = False
        self.miss_min = []
        self.beta_callback = []