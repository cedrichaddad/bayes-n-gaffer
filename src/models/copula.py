"""
t-Copula Dependency Engine for Alpha-FPL.

Models the joint distribution of player residuals to capture:
1. Linear correlation (Pearson ρ)
2. Tail dependence (joint crashes/rallies)

Key insight: When Liverpool concedes 5 goals, ALL their defenders
score poorly together. Gaussian copulas underestimate this tail risk.
The t-Copula with low degrees of freedom captures this joint tail behavior.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma, gammaln
import polars as pl
from loguru import logger


@dataclass
class CopulaConfig:
    """Configuration for t-Copula fitting."""
    
    # Number of scenarios to generate
    n_scenarios: int = 1000
    
    # Minimum observations required to include a player pair
    min_observations: int = 10
    
    # Initial degrees of freedom for t-Copula
    initial_df: float = 5.0
    
    # Bounds for df optimization
    df_bounds: Tuple[float, float] = (2.1, 30.0)
    
    # Regularization for correlation matrix (ensures positive definiteness)
    ridge_alpha: float = 0.01


class TCopulaEngine:
    """
    Student's t-Copula for modeling player score dependencies.
    
    The t-Copula is parametrized by:
    - Σ (Sigma): Correlation matrix
    - ν (nu): Degrees of freedom (captures tail dependence)
    
    Low ν (e.g., 4) → Heavy tails → Joint extreme events more likely
    High ν (e.g., 30) → Approaches Gaussian → Independent tails
    
    Workflow:
    1. Fit Bayesian model to get E[Points]
    2. Compute residuals: R = Y_actual - E[Y_predicted]
    3. Transform to uniform via PIT
    4. Fit t-Copula to uniform margins
    5. Generate correlated scenarios
    """
    
    def __init__(self, config: CopulaConfig = None):
        """
        Initialize the copula engine.
        
        Args:
            config: Copula configuration
        """
        self.config = config or CopulaConfig()
        
        # Fitted parameters
        self.correlation_matrix = None
        self.degrees_of_freedom = None
        self.player_ids = None
        self.marginal_cdfs = {}
    
    def compute_residuals(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> np.ndarray:
        """
        Compute standardized residuals.
        
        Args:
            actual: Actual observed values (n_obs, n_players)
            predicted: Predicted expected values (n_obs, n_players)
            
        Returns:
            Standardized residuals
        """
        residuals = actual - predicted
        
        # Standardize each player's residuals
        means = np.nanmean(residuals, axis=0)
        stds = np.nanstd(residuals, axis=0)
        stds[stds < 1e-6] = 1.0  # Avoid division by zero
        
        standardized = (residuals - means) / stds
        
        return standardized
    
    def probability_integral_transform(
        self,
        residuals: np.ndarray,
    ) -> np.ndarray:
        """
        Transform residuals to uniform [0, 1] via empirical CDF.
        
        Args:
            residuals: Standardized residuals (n_obs, n_players)
            
        Returns:
            Uniform-transformed values
        """
        n_obs, n_players = residuals.shape
        uniform = np.zeros_like(residuals)
        
        for j in range(n_players):
            # Use rank-based empirical CDF
            r = residuals[:, j]
            valid_mask = ~np.isnan(r)
            ranks = stats.rankdata(r[valid_mask])
            # Adjust to avoid 0 and 1 exactly
            uniform[valid_mask, j] = ranks / (valid_mask.sum() + 1)
            uniform[~valid_mask, j] = 0.5  # Impute with median
            
            # Store empirical CDF for later inverse transform
            self.marginal_cdfs[j] = {
                "values": np.sort(r[valid_mask]),
                "cdf": np.linspace(0, 1, valid_mask.sum()),
            }
        
        return uniform
    
    def fit_correlation_matrix(
        self,
        uniform: np.ndarray,
    ) -> np.ndarray:
        """
        Fit the correlation matrix from uniform marginals.
        
        Uses the inverse t-CDF to transform to t-distributed,
        then computes Pearson correlation.
        
        Args:
            uniform: Uniform-transformed residuals
            
        Returns:
            Estimated correlation matrix
        """
        n_obs, n_players = uniform.shape
        
        # Initial df for transformation
        df = self.config.initial_df
        
        # Transform to t-distributed
        t_transformed = stats.t.ppf(np.clip(uniform, 0.001, 0.999), df=df)
        
        # Compute correlation matrix
        corr = np.corrcoef(t_transformed.T)
        
        # Ensure positive definiteness with ridge regularization
        corr = (1 - self.config.ridge_alpha) * corr + self.config.ridge_alpha * np.eye(n_players)
        
        return corr
    
    def fit_degrees_of_freedom(
        self,
        uniform: np.ndarray,
        correlation: np.ndarray,
    ) -> float:
        """
        Fit degrees of freedom via maximum likelihood.
        
        Args:
            uniform: Uniform marginals
            correlation: Fitted correlation matrix
            
        Returns:
            Optimal degrees of freedom
        """
        n_obs, n_players = uniform.shape
        
        def neg_log_likelihood(nu_arr):
            nu = nu_arr[0]
            if nu <= 2:
                return 1e10
            
            # Transform to t
            t_vals = stats.t.ppf(np.clip(uniform, 0.001, 0.999), df=nu)
            
            # Compute t-Copula density (simplified)
            # Full formula involves determinant and multivariate t
            try:
                L = np.linalg.cholesky(correlation)
                log_det = 2 * np.sum(np.log(np.diag(L)))
                
                # Quadratic form
                solved = np.linalg.solve(L, t_vals.T)
                quad_form = np.sum(solved ** 2, axis=0)
                
                # Log-likelihood contribution
                ll = (
                    gammaln((nu + n_players) / 2) -
                    gammaln(nu / 2) -
                    (n_players / 2) * np.log(nu * np.pi) -
                    0.5 * log_det -
                    ((nu + n_players) / 2) * np.log(1 + quad_form / nu)
                )
                
                return -np.mean(ll)
            except np.linalg.LinAlgError:
                return 1e10
        
        result = minimize(
            neg_log_likelihood,
            x0=[self.config.initial_df],
            bounds=[self.config.df_bounds],
            method="L-BFGS-B"
        )
        
        return result.x[0]
    
    def fit(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        player_ids: list = None,
    ) -> "TCopulaEngine":
        """
        Fit the t-Copula to observed data.
        
        Args:
            actual: Actual values (n_obs, n_players)
            predicted: Predicted values (n_obs, n_players)
            player_ids: Optional list of player IDs
            
        Returns:
            Fitted copula engine
        """
        logger.info(f"Fitting t-Copula to {actual.shape} observations...")
        
        self.player_ids = player_ids or list(range(actual.shape[1]))
        
        # Step 1: Compute residuals
        residuals = self.compute_residuals(actual, predicted)
        
        # Step 2: PIT transformation
        uniform = self.probability_integral_transform(residuals)
        
        # Step 3: Fit correlation matrix
        self.correlation_matrix = self.fit_correlation_matrix(uniform)
        
        # Step 4: Fit degrees of freedom
        self.degrees_of_freedom = self.fit_degrees_of_freedom(
            uniform, self.correlation_matrix
        )
        
        logger.info(f"Fitted t-Copula: ν = {self.degrees_of_freedom:.2f}")
        logger.info(f"Correlation matrix condition: {np.linalg.cond(self.correlation_matrix):.2f}")
        
        return self
    
    def generate_scenarios(
        self,
        expected_points: np.ndarray,
        point_stds: np.ndarray,
        n_scenarios: int = None,
    ) -> np.ndarray:
        """
        Generate correlated scenarios from the fitted copula.
        
        Args:
            expected_points: E[Points] for each player (n_players,)
            point_stds: Std[Points] for each player (n_players,)
            n_scenarios: Number of scenarios (defaults to config)
            
        Returns:
            Scenario matrix (n_scenarios, n_players)
        """
        if self.correlation_matrix is None:
            raise ValueError("Copula must be fitted before generating scenarios")
        
        n_scenarios = n_scenarios or self.config.n_scenarios
        n_players = len(expected_points)
        
        # Adjust correlation matrix size if needed
        corr = self.correlation_matrix
        if corr.shape[0] != n_players:
            # Subset or expand correlation matrix
            if corr.shape[0] > n_players:
                corr = corr[:n_players, :n_players]
            else:
                # Expand with uncorrelated players
                new_corr = np.eye(n_players)
                new_corr[:corr.shape[0], :corr.shape[1]] = corr
                corr = new_corr
        
        # Generate multivariate t samples
        # Method: Z = sqrt(nu/chi2) * MVN, where chi2 ~ chi2(nu)
        nu = self.degrees_of_freedom
        
        # MVN samples
        mvn = np.random.multivariate_normal(
            mean=np.zeros(n_players),
            cov=corr,
            size=n_scenarios
        )
        
        # Chi-squared samples for t-distribution
        chi2 = np.random.chisquare(nu, size=n_scenarios)
        scaling = np.sqrt(nu / chi2)
        
        # t-distributed samples
        t_samples = mvn * scaling[:, np.newaxis]
        
        # Transform to uniform via t-CDF
        uniform = stats.t.cdf(t_samples, df=nu)
        
        # Inverse transform to points scale
        # Assume Normal marginals for simplicity
        z_samples = stats.norm.ppf(np.clip(uniform, 0.001, 0.999))
        
        scenarios = expected_points + z_samples * point_stds
        
        # Ensure non-negative points
        scenarios = np.maximum(scenarios, 0)
        
        return scenarios
    
    def get_covariance_matrix(
        self,
        point_stds: np.ndarray,
    ) -> np.ndarray:
        """
        Get the covariance matrix for optimization.
        
        Σ = diag(σ) @ ρ @ diag(σ)
        
        Args:
            point_stds: Standard deviations for each player
            
        Returns:
            Covariance matrix
        """
        if self.correlation_matrix is None:
            # Return diagonal if not fitted
            return np.diag(point_stds ** 2)
        
        D = np.diag(point_stds)
        return D @ self.correlation_matrix @ D
    
    def compute_tail_dependence(self) -> float:
        """
        Compute the tail dependence coefficient.
        
        For t-Copula: λ = 2 * t_{ν+1}(-sqrt((ν+1)(1-ρ)/(1+ρ)))
        
        Returns:
            Lower tail dependence coefficient (0 to 1)
        """
        if self.degrees_of_freedom is None:
            return 0.0
        
        nu = self.degrees_of_freedom
        
        # Use average off-diagonal correlation
        rho = np.mean(self.correlation_matrix[np.triu_indices_from(
            self.correlation_matrix, k=1
        )])
        
        if rho <= -0.99:
            return 0.0
        
        arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        lambda_L = 2 * stats.t.cdf(arg, df=nu + 1)
        
        return lambda_L


class ScenarioGenerator:
    """
    Convenience class for generating optimization scenarios.
    
    Combines Bayesian predictions with copula dependence.
    """
    
    def __init__(
        self,
        copula: TCopulaEngine = None,
    ):
        self.copula = copula or TCopulaEngine()
    
    def generate_for_gameweek(
        self,
        player_predictions: dict,
        n_scenarios: int = 1000,
    ) -> np.ndarray:
        """
        Generate scenarios for a single gameweek.
        
        Args:
            player_predictions: Dict mapping player_id to {mean, std}
            n_scenarios: Number of scenarios
            
        Returns:
            Scenario matrix where rows are scenarios, columns are players
        """
        player_ids = list(player_predictions.keys())
        means = np.array([player_predictions[p]["mean"] for p in player_ids])
        stds = np.array([player_predictions[p]["std"] for p in player_ids])
        
        return self.copula.generate_scenarios(
            expected_points=means,
            point_stds=stds,
            n_scenarios=n_scenarios,
        ), player_ids
