import numpy as np
from scipy import stats

class LinearRegressionStatisticalView:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.beta_0 = None
        self.beta_1 = None
        self.n = None
        self.x_bar = None
        self.y_bar = None
        self.Sxx = None
        self.Sxy = None
        self.SSE = None
        self.SST = None
        self.SSR = None
        self.MSR = None
        self.MSE = None
        self.df_model = None
        self.df_resid = None
        self.df_total = None
        self.f_statistic_value = None
        self.f_critical_value = None
        self.beta_0_lower_bound = None
        self.beta_0_upper_bound = None
        self.beta_1_lower_bound = None
        self.beta_1_upper_bound = None
        self.r_squared_value = None
        self.r_value = None
        
    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if len(X) != len(Y):
            raise ValueError("X and Y must be of equal length.")
        
        self.n = len(X)
        self.x_bar = np.mean(X)
        self.y_bar = np.mean(Y)
        self.Sxx = np.sum(X**2) - self.n * self.x_bar**2
        self.Sxy = np.sum(X*Y) - self.n * self.x_bar * self.y_bar
        self.beta_1 = self.Sxy / self.Sxx
        self.beta_0 = self.y_bar - self.beta_1 * self.x_bar
        y_hat = self.beta_0 + self.beta_1 * X
        self.SSR = np.sum((y_hat - self.y_bar)**2)
        self.SSE = np.sum((Y - y_hat)**2)
        self.SST = self.SSR + self.SSE
        self.df_model = 1
        self.df_resid = self.n - 2
        self.df_total = self.n - 1
    
    def f_statistic(self) -> float:
        self.MSR = self.SSR / self.df_model
        self.MSE = self.SSE / self.df_resid
        self.f_statistic_value = self.MSR / self.MSE
        return self.f_statistic_value
    
    def f_critical(self) -> float:
        self.f_critical_value = stats.f.isf(self.alpha, self.df_model, self.df_resid)
        return self.f_critical_value
    
    def beta_confidence_intervals(self) -> dict:
        t_crit = stats.t.isf(self.alpha / 2, df=self.df_resid)
        
        self.beta_0_lower_bound = self.beta_0 - t_crit * np.sqrt(self.MSE * (1 / self.n + self.x_bar ** 2 / self.Sxx))
        self.beta_0_upper_bound = self.beta_0 + t_crit * np.sqrt(self.MSE * (1 / self.n + self.x_bar ** 2 / self.Sxx))
        self.beta_1_lower_bound = self.beta_1 - t_crit * np.sqrt(self.MSE / self.Sxx)
        self.beta_1_upper_bound = self.beta_1 + t_crit * np.sqrt(self.MSE / self.Sxx)

        return {
            "beta_0": (self.beta_0_lower_bound, self.beta_0_upper_bound),
            "beta_1": (self.beta_1_lower_bound, self.beta_1_upper_bound)
        }

    def degrees_of_freedom(self) -> dict:
        return {
            "df_model": self.df_model,
            "df_resid": self.df_resid,
            "df_total": self.df_total
        }
        
    def r_squared(self) -> float:
        self.r_squared_value = self.SSR / self.SST
        return self.r_squared_value

    def r(self) -> float:
        sign = np.sign(self.beta_1)
        self.r_value = sign * np.sqrt(self.r_squared_value)
        return self.r_value

    def summary(self) -> dict:
        return {
            "f_critical": self.f_critical_value, 
            "f_stat": self.f_statistic_value, 
            "beta_confidence_intervals": { 
                "beta_0": (self.beta_0_lower_bound, self.beta_0_upper_bound), 
                "beta_1": (self.beta_1_lower_bound, self.beta_1_upper_bound) 
                }, 
            "degrees_of_freedom": { 
                "df_model": self.df_model, 
                "df_resid": self.df_resid, 
                "df_total": self.df_total 
                }, 
            "r_squareddef": self.r_squared_value, 
            "r": self.r_value 
        }

def compare(df, x_col_1: str, x_col_2: str, y_col: str, alpha: float = 0.05) -> dict:
    model_a = LinearRegressionStatisticalView(alpha=alpha)
    model_b = LinearRegressionStatisticalView(alpha=alpha)

    model_a.fit(df[x_col_1].values, df[y_col].values)
    model_b.fit(df[x_col_2].values, df[y_col].values)
    
    f_stat_1 = model_a.f_statistic()    
    f_stat_2 = model_b.f_statistic()
    
    if f_stat_1 > f_stat_2: 
        best_predictor = x_col_1 
    else:
        best_predictor = x_col_2
    
    return {
        "predictor_1": x_col_1, 
        "predictor_2": x_col_2, 
        "f_stat_1": f_stat_1, 
        "f_stat_2": f_stat_2, 
        "best_predictor": best_predictor  
    }