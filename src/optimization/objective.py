"""
Objective Function Builder for Alpha-FPL Optimization.

Constructs the multi-objective function for the stochastic MIQP:
1. Expected Return (Mean Points)
2. Risk Penalty (Variance / CVaR)
3. Future Option Value (Planning Horizon)

Key innovation: Separates Starting XI from Bench valuation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class ObjectiveConfig:
    """Configuration for objective function."""
    
    # Risk aversion (Mean-Variance tradeoff)
    risk_aversion_lambda: float = 0.15
    
    # CVaR confidence level (worst α%)
    cvar_alpha: float = 0.05
    
    # Planning horizon (gameweeks ahead)
    planning_horizon: int = 3
    
    # Future value discount
    discount_factor: float = 0.95
    
    # DRO robustness radius
    dro_radius: float = 0.01
    
    # Bench valuation
    bench_auto_sub_prob: float = 0.10
    
    # Points weights by position
    position_goal_points: dict = None
    
    def __post_init__(self):
        if self.position_goal_points is None:
            self.position_goal_points = {
                "GKP": 6, "DEF": 6, "MID": 5, "FWD": 4
            }


class ObjectiveBuilder:
    """
    Builds the objective function components for the MIQP solver.
    
    The full objective is:
    Maximize:
        Σ(x_start · μ) +                              # Starting XI expected points
        Σ(x_bench · μ · P(sub_needed)) +              # Bench expected contribution
        Σ(x_captain · μ) -                            # Captain bonus (doubles points)
        λ · CVaR_α(Portfolio) +                       # Risk penalty
        γ · E[V_{t+1}(x)]                             # Future planning value
    """
    
    def __init__(self, config: ObjectiveConfig = None):
        """
        Initialize objective builder.
        
        Args:
            config: Objective configuration
        """
        self.config = config or ObjectiveConfig()
    
    def compute_expected_return_coefficients(
        self,
        expected_points: np.ndarray,
        xmins_factors: np.ndarray,
    ) -> tuple:
        """
        Compute expected return coefficients for starting and bench.
        
        Args:
            expected_points: E[Points|90] for each player
            xmins_factors: Minutes adjustment factor (E[mins]/90)
            
        Returns:
            (start_coefficients, bench_coefficients, captain_coefficients)
        """
        # Adjusted expected points
        adjusted_points = expected_points * xmins_factors
        
        # Starting XI: full expected points
        start_coef = adjusted_points.copy()
        
        # Bench: weighted by probability of needing sub
        bench_coef = adjusted_points * self.config.bench_auto_sub_prob
        
        # Captain: additional expected points (captain gets double)
        captain_coef = adjusted_points.copy()
        
        return start_coef, bench_coef, captain_coef
    
    def compute_cvar(
        self,
        scenarios: np.ndarray,
        weights: np.ndarray,
        alpha: float = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute Conditional Value at Risk.
        
        CVaR_α = E[X | X <= VaR_α]
        
        Args:
            scenarios: Scenario matrix (n_scenarios, n_players)
            weights: Portfolio weights (binary 0/1)
            alpha: CVaR confidence level
            
        Returns:
            (cvar_value, worst_scenarios)
        """
        alpha = alpha or self.config.cvar_alpha
        
        # Portfolio returns under each scenario
        portfolio_returns = scenarios @ weights
        
        # Find VaR (α-quantile)
        var = np.percentile(portfolio_returns, alpha * 100)
        
        # CVaR: mean of returns below VaR
        worst_mask = portfolio_returns <= var
        if worst_mask.sum() == 0:
            cvar = var
        else:
            cvar = portfolio_returns[worst_mask].mean()
        
        return cvar, portfolio_returns[worst_mask]
    
    def compute_variance_penalty(
        self,
        covariance_matrix: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute portfolio variance for risk penalty.
        
        Var(Portfolio) = w' Σ w
        
        Args:
            covariance_matrix: Player covariance matrix
            weights: Portfolio weights
            
        Returns:
            Portfolio variance
        """
        return float(weights.T @ covariance_matrix @ weights)
    
    def compute_dro_penalty(
        self,
        expected_points: np.ndarray,
        point_variances: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute Distributionally Robust penalty.
        
        This penalizes players with high estimation uncertainty.
        
        Penalty = ε * sqrt(Σ(w_i² * σ_i²))
        
        Args:
            expected_points: E[Points] for each player
            point_variances: Var[Points] for each player
            weights: Portfolio weights
            
        Returns:
            DRO penalty term
        """
        weighted_var = np.sum(weights ** 2 * point_variances)
        return self.config.dro_radius * np.sqrt(weighted_var)
    
    def compute_future_value(
        self,
        team_value: float,
        saved_transfers: int,
        expected_fixture_difficulty: np.ndarray = None,
    ) -> float:
        """
        Compute approximate future option value.
        
        V_{t+1} ≈ TeamValue + β × SavedTransfers + FixtureDifficultyBonus
        
        Args:
            team_value: Current team sell value
            saved_transfers: Number of free transfers banked
            expected_fixture_difficulty: Future FDR for owned players
            
        Returns:
            Estimated future value
        """
        # Transfer value: having 2 FT is worth more than 1
        transfer_value = saved_transfers * 2.0  # Points equivalent
        
        # Team value component (optional, for chip planning)
        value_component = team_value * 0.01  # Small weight
        
        future_value = value_component + transfer_value
        
        return future_value * self.config.discount_factor
    
    def build_objective_vector(
        self,
        expected_points: np.ndarray,
        point_variances: np.ndarray,
        xmins_factors: np.ndarray,
        covariance_matrix: np.ndarray = None,
        scenario_matrix: np.ndarray = None,
    ) -> dict:
        """
        Build all objective components for the solver.
        
        Args:
            expected_points: E[Points] per player
            point_variances: Var[Points] per player
            xmins_factors: E[mins]/90 per player
            covariance_matrix: Player covariance (from copula)
            scenario_matrix: Scenario samples for CVaR
            
        Returns:
            Dictionary with objective components
        """
        n_players = len(expected_points)
        
        # Expected return coefficients
        start_coef, bench_coef, captain_coef = self.compute_expected_return_coefficients(
            expected_points, xmins_factors
        )
        
        # Covariance for variance penalty
        if covariance_matrix is None:
            # Use diagonal covariance
            covariance_matrix = np.diag(point_variances)
        
        # DRO penalty coefficients
        dro_coef = self.config.dro_radius * np.sqrt(point_variances)
        
        return {
            # Linear coefficients
            "start_coefficients": start_coef,
            "bench_coefficients": bench_coef,
            "captain_coefficients": captain_coef,
            "dro_penalty_coefficients": dro_coef,
            
            # Quadratic (for Gurobi)
            "covariance_matrix": covariance_matrix,
            "risk_aversion": self.config.risk_aversion_lambda,
            
            # Scenarios for CVaR constraint
            "scenario_matrix": scenario_matrix,
            "cvar_alpha": self.config.cvar_alpha,
            
            # Metadata
            "n_players": n_players,
        }
    
    def evaluate_portfolio(
        self,
        weights_start: np.ndarray,
        weights_bench: np.ndarray,
        weights_captain: np.ndarray,
        objective_components: dict,
    ) -> dict:
        """
        Evaluate objective value for a given portfolio.
        
        Args:
            weights_start: Binary starting XI
            weights_bench: Binary bench
            weights_captain: Binary captain
            objective_components: From build_objective_vector
            
        Returns:
            Dictionary with objective breakdown
        """
        start_coef = objective_components["start_coefficients"]
        bench_coef = objective_components["bench_coefficients"]
        captain_coef = objective_components["captain_coefficients"]
        cov = objective_components["covariance_matrix"]
        lambda_risk = objective_components["risk_aversion"]
        
        # Expected return
        expected_return = (
            start_coef @ weights_start +
            bench_coef @ weights_bench +
            captain_coef @ weights_captain
        )
        
        # Variance penalty
        # Effective weight: Start (1x) + Bench (1x) + Captain_Bonus (1x) = Captain is 2x
        full_weights = weights_start + weights_bench + weights_captain
        variance = self.compute_variance_penalty(cov, full_weights)
        variance_penalty = lambda_risk * variance
        
        # CVaR if scenarios available
        if objective_components.get("scenario_matrix") is not None:
            cvar, _ = self.compute_cvar(
                objective_components["scenario_matrix"],
                full_weights,
                objective_components["cvar_alpha"]
            )
        else:
            cvar = 0
        
        total = expected_return - variance_penalty
        
        return {
            "expected_return": float(expected_return),
            "variance_penalty": float(variance_penalty),
            "cvar": float(cvar),
            "total_objective": float(total),
        }
