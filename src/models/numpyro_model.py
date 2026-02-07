"""
Hierarchical Bayesian Model for Alpha-FPL.

Implements the core probabilistic model for player performance prediction.
Uses NumPyro for MCMC inference with JAX backend for TPU acceleration.

Mathematical Specification:
- Goals: Zero-Inflated Poisson with hierarchical player/team effects
- Assists: Similar Poisson structure
- DefCon: Bernoulli-Logit based on CBIT features
- Clean Sheets: Team-level Bernoulli
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.handlers import condition, scale
import numpy as np
import polars as pl
from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for the Bayesian model."""
    
    # MCMC settings
    chains: int = 4
    samples: int = 2000
    warmup: int = 500
    
    # Prior hyperparameters
    alpha_sigma: float = 1.0  # Player ability shrinkage
    team_sigma: float = 0.5   # Team effect shrinkage
    position_sigma: float = 1.0
    
    # Time decay for likelihood weighting
    time_decay_kappa: float = 0.05
    
    # Device
    device: str = "cpu"  # cpu, gpu, tpu


class HierarchicalPointsModel:
    """
    Hierarchical Bayesian model for FPL points prediction.
    
    This model decomposes points into components:
    1. Goals (Zero-Inflated Poisson)
    2. Assists (Poisson)
    3. DefCon points (Bernoulli)
    4. Clean sheet contribution (Bernoulli)
    
    Hierarchical structure:
    - Player ability α_i ~ Normal(μ_pos, σ_pos)
    - Team attack δ_j ~ Normal(0, σ_team)
    - Opponent defense θ_k ~ Normal(0, σ_team)
    """
    
    def __init__(self, config: ModelConfig = None):
        """
        Initialize the hierarchical model.
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.mcmc = None
        self.posterior_samples = None
        
        # Set up JAX device
        self._setup_device()
    
    def _setup_device(self):
        """Configure JAX for the target device."""
        device = self.config.device.lower()
        if device == "tpu":
            try:
                jax.devices("tpu")
                logger.info("Using TPU backend")
            except RuntimeError:
                logger.warning("TPU not available, falling back to CPU")
        elif device == "gpu":
            try:
                jax.devices("gpu")
                logger.info("Using GPU backend")
            except RuntimeError:
                logger.warning("GPU not available, falling back to CPU")
        else:
            logger.info("Using CPU backend")
    
    def _goal_scoring_model(
        self,
        player_idx: jnp.ndarray,
        team_idx: jnp.ndarray,
        opp_idx: jnp.ndarray,
        position_idx: jnp.ndarray,
        is_home: jnp.ndarray,
        n_players: int,
        n_teams: int,
        n_positions: int,
        goals: jnp.ndarray = None,
    ):
        """
        Goal scoring component of the model.
        
        λ_i,t = exp(α_i + δ_team(i) - θ_opp(i) + γ * Home)
        π_i = logit^(-1)(φ_pos(i))
        Goals ~ ZeroInflatedPoisson(λ, π)
        """
        # =====================================================================
        # Hierarchical Priors
        # =====================================================================
        
        # Position-level priors (hyperpriors)
        mu_pos = numpyro.sample(
            "mu_pos_goals",
            dist.Normal(0, 1).expand([n_positions])
        )
        sigma_pos = numpyro.sample(
            "sigma_pos_goals",
            dist.HalfNormal(self.config.position_sigma).expand([n_positions])
        )
        
        # Player ability (hierarchical on position)
        with numpyro.plate("players_goals", n_players):
            alpha_raw = numpyro.sample("alpha_raw_goals", dist.Normal(0, 1))
        
        # Re-parameterized to reduce divergences
        alpha = mu_pos[position_idx] + sigma_pos[position_idx] * alpha_raw
        
        # Team attacking strength
        with numpyro.plate("teams_attack", n_teams):
            delta_team = numpyro.sample(
                "delta_team",
                dist.Normal(0, self.config.team_sigma)
            )
        
        # Opponent defensive strength
        with numpyro.plate("teams_defense", n_teams):
            theta_opp = numpyro.sample(
                "theta_opp",
                dist.Normal(0, self.config.team_sigma)
            )
        
        # Home advantage
        gamma_home = numpyro.sample("gamma_home", dist.Normal(0.2, 0.1))
        
        # Zero-inflation probability by position
        phi_pos = numpyro.sample(
            "phi_pos",
            dist.Beta(2, 5).expand([n_positions])
        )
        
        # =====================================================================
        # Likelihood
        # =====================================================================
        
        # Expected goals rate
        log_lambda = (
            alpha[player_idx] +
            delta_team[team_idx] -
            theta_opp[opp_idx] +
            gamma_home * is_home
        )
        lambda_goals = jnp.exp(jnp.clip(log_lambda, -5, 3))
        
        # Zero-inflation probability
        pi = phi_pos[position_idx[player_idx]]
        
        # Zero-Inflated Poisson likelihood with time decay weighting
        # NOTE: time_weights must be passed to the calling model() function
        return lambda_goals, pi
    
    def _defcon_model(
        self,
        position_idx: jnp.ndarray,
        opp_possession: jnp.ndarray,
        cbit_avg: jnp.ndarray,
        n_positions: int,
        defcon: jnp.ndarray = None,
    ):
        """
        DefCon points component (Bernoulli-Logit).
        
        p_defcon = logit^(-1)(β_0 + β_1 * Possession_opp + β_2 * CBIT_avg)
        """
        # Coefficients
        beta_0 = numpyro.sample("beta_0_defcon", dist.Normal(-2, 1))
        beta_1 = numpyro.sample("beta_1_possession", dist.Normal(0.5, 0.2))
        beta_2 = numpyro.sample("beta_2_cbit", dist.Normal(0.3, 0.1))
        
        # Position adjustment
        pos_adj = numpyro.sample(
            "pos_adj_defcon",
            dist.Normal(0, 0.5).expand([n_positions])
        )
        
        # Logit probability
        logit_p = (
            beta_0 +
            beta_1 * opp_possession +
            beta_2 * cbit_avg +
            pos_adj[position_idx]
        )
        p_defcon = jax.nn.sigmoid(logit_p)
        
        # Return probability for likelihood to be computed in parent
        return p_defcon
    
    def model(
        self,
        # Indices
        player_idx: jnp.ndarray,
        team_idx: jnp.ndarray,
        opp_idx: jnp.ndarray,
        position_idx: jnp.ndarray,
        # Features
        is_home: jnp.ndarray,
        opp_possession: jnp.ndarray,
        cbit_avg: jnp.ndarray,
        # Sizes
        n_players: int,
        n_teams: int,
        n_positions: int,
        # Time weights
        time_weights: jnp.ndarray = None,
        # Observations (None during prediction)
        goals: jnp.ndarray = None,
        assists: jnp.ndarray = None,
        defcon: jnp.ndarray = None,
    ):
        """
        Full hierarchical model combining all components.
        
        Args:
            player_idx: Player index for each observation
            team_idx: Team index for each observation
            opp_idx: Opponent index for each observation
            position_idx: Position index for each player
            is_home: Binary home/away indicator
            opp_possession: Opponent expected possession
            cbit_avg: Player's average CBIT
            n_players: Total number of players
            n_teams: Total number of teams
            n_positions: Number of positions (4)
            time_weights: Likelihood weights for time decay
            goals: Observed goals (None for prediction)
            assists: Observed assists (None for prediction)
            defcon: Observed DefCon (None for prediction)
        """
        # Default time weights to 1.0 if not provided
        if time_weights is None:
            time_weights = jnp.ones(len(player_idx))
        
        # ==================================================================
        # Goal scoring component
        # ==================================================================
        lambda_goals, pi_goals = self._goal_scoring_model(
            player_idx, team_idx, opp_idx, position_idx, is_home,
            n_players, n_teams, n_positions, goals=None  # Likelihood computed below
        )
        
        # ==================================================================
        # Assist component (similar structure, simplified)
        # ==================================================================
        mu_assist = numpyro.sample("mu_assist", dist.Normal(-0.5, 0.5))
        sigma_assist = numpyro.sample("sigma_assist", dist.HalfNormal(0.5))
        
        with numpyro.plate("players_assist", n_players):
            alpha_assist = numpyro.sample(
                "alpha_assist",
                dist.Normal(mu_assist, sigma_assist)
            )
        
        lambda_assists = jnp.exp(jnp.clip(alpha_assist[player_idx], -5, 2))
        
        # ==================================================================
        # DefCon component
        # ==================================================================
        p_defcon = self._defcon_model(
            position_idx[player_idx], opp_possession, cbit_avg,
            n_positions, defcon=None  # Likelihood computed below
        )
        
        # ==================================================================
        # CRITICAL FIX: Apply time_weights to all likelihoods
        # This ensures recent observations have more influence on the posterior
        # ==================================================================
        with numpyro.plate("observations", len(player_idx)):
            # Scale log-likelihood by time weights
            with scale(scale=time_weights):
                # Goals (Zero-Inflated Poisson)
                numpyro.sample(
                    "goals",
                    dist.ZeroInflatedPoisson(gate=pi_goals, rate=lambda_goals),
                    obs=goals
                )
                
                # Assists (Poisson)
                numpyro.sample(
                    "assists_obs",
                    dist.Poisson(lambda_assists),
                    obs=assists
                )
                
                # DefCon (Bernoulli)
                numpyro.sample(
                    "defcon",
                    dist.Bernoulli(probs=p_defcon),
                    obs=defcon
                )
    
    def fit(
        self,
        data: dict,
        rng_key: jax.random.PRNGKey = None,
    ) -> "HierarchicalPointsModel":
        """
        Fit the model using MCMC.
        
        Args:
            data: Dictionary with model inputs
            rng_key: JAX random key
            
        Returns:
            Self with fitted posterior samples
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        
        logger.info(f"Fitting model with {self.config.chains} chains, "
                   f"{self.config.samples} samples, {self.config.warmup} warmup")
        
        # Create NUTS sampler
        kernel = NUTS(self.model)
        
        # Run MCMC
        self.mcmc = MCMC(
            kernel,
            num_warmup=self.config.warmup,
            num_samples=self.config.samples,
            num_chains=self.config.chains,
            progress_bar=True
        )
        
        self.mcmc.run(rng_key, **data)
        self.posterior_samples = self.mcmc.get_samples()
        
        # Log diagnostics
        divergences = self.mcmc.get_extra_fields()["diverging"].sum()
        logger.info(f"MCMC complete. Divergences: {divergences}")
        
        return self
    
    def predict(
        self,
        data: dict,
        rng_key: jax.random.PRNGKey = None,
        num_samples: int = 1000,
    ) -> dict:
        """
        Generate posterior predictive samples.
        
        Args:
            data: Dictionary with prediction inputs (observations=None)
            rng_key: JAX random key
            num_samples: Number of posterior predictive samples
            
        Returns:
            Dictionary of predictive samples
        """
        if self.posterior_samples is None:
            raise ValueError("Model must be fit before prediction")
        
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        predictive = Predictive(
            self.model,
            posterior_samples=self.posterior_samples,
            num_samples=num_samples
        )
        
        # Remove observations from data for prediction
        pred_data = {k: v for k, v in data.items() 
                     if k not in ["goals", "assists", "defcon"]}
        pred_data["goals"] = None
        pred_data["assists"] = None
        pred_data["defcon"] = None
        
        return predictive(rng_key, **pred_data)
    
    def prepare_data(
        self,
        df: pl.DataFrame,
        player_col: str = "fpl_id",
        team_col: str = "team_name",
        opp_col: str = "opponent_team",
        position_col: str = "position",
    ) -> dict:
        """
        Prepare DataFrame for model input.
        
        Args:
            df: DataFrame with player-gameweek data
            player_col: Column name for player ID
            team_col: Column name for team
            opp_col: Column name for opponent
            position_col: Column name for position
            
        Returns:
            Dictionary ready for model input
        """
        # Create indices
        players = df[player_col].unique().sort().to_list()
        teams = df[team_col].unique().sort().to_list()
        positions = ["GKP", "DEF", "MID", "FWD"]
        
        player_to_idx = {p: i for i, p in enumerate(players)}
        team_to_idx = {t: i for i, t in enumerate(teams)}
        pos_to_idx = {p: i for i, p in enumerate(positions)}
        
        # Map to indices
        player_idx = jnp.array([player_to_idx[p] for p in df[player_col].to_list()])
        team_idx = jnp.array([team_to_idx.get(t, 0) for t in df[team_col].to_list()])
        opp_idx = jnp.array([team_to_idx.get(o, 0) for o in df[opp_col].to_list()])
        
        # Player position lookup
        player_positions = df.group_by(player_col).agg(
            pl.col(position_col).first()
        )
        position_idx = jnp.array([
            pos_to_idx.get(
                player_positions.filter(pl.col(player_col) == p)[position_col][0], 0
            ) for p in players
        ])
        
        # Features
        is_home = jnp.array(
            df["was_home"].cast(pl.Int32).fill_null(0).to_numpy() 
            if "was_home" in df.columns else [0] * len(df)
        )
        
        # CRITICAL: Use LAGGED opponent possession to avoid data leakage
        # opp_possession should be the opponent's average possession in their LAST N games
        # NOT the possession from the current game being predicted
        if "opp_possession_lagged" in df.columns:
            opp_possession = jnp.array(df["opp_possession_lagged"].fill_null(0.5).to_numpy())
        elif "opp_possession" in df.columns:
            # WARNING: Using non-lagged feature - potential data leakage!
            logger.warning("Using opp_possession without lag - potential data leakage")
            opp_possession = jnp.array(df["opp_possession"].fill_null(0.5).to_numpy())
        else:
            opp_possession = jnp.array([0.5] * len(df))
        
        cbit_avg = jnp.array(
            df["cbit"].fill_null(0).to_numpy()
            if "cbit" in df.columns else [0.0] * len(df)
        )
        
        # Observations
        goals = jnp.array(
            df["goals_scored"].fill_null(0).cast(pl.Int32).to_numpy()
            if "goals_scored" in df.columns else None
        )
        
        assists = jnp.array(
            df["assists"].fill_null(0).cast(pl.Int32).to_numpy()
            if "assists" in df.columns else None
        )
        
        defcon = jnp.array(
            (df["defcon_points"].fill_null(0) > 0).cast(pl.Int32).to_numpy()
            if "defcon_points" in df.columns else None
        )
        
        # Time weights for decay
        if "gameweek" in df.columns and "season" in df.columns:
            # Convert to time index
            max_gw = df.select(pl.max("gameweek")).item()
            time_diff = max_gw - df["gameweek"].to_numpy()
            time_weights = jnp.exp(-self.config.time_decay_kappa * time_diff)
        else:
            time_weights = jnp.ones(len(df))
        
        return {
            "player_idx": player_idx,
            "team_idx": team_idx,
            "opp_idx": opp_idx,
            "position_idx": position_idx,
            "is_home": is_home,
            "opp_possession": opp_possession,
            "cbit_avg": cbit_avg,
            "n_players": len(players),
            "n_teams": len(teams),
            "n_positions": len(positions),
            "time_weights": time_weights,
            "goals": goals,
            "assists": assists,
            "defcon": defcon,
            # Store mappings for later use
            "_player_to_idx": player_to_idx,
            "_team_to_idx": team_to_idx,
            "_pos_to_idx": pos_to_idx,
        }
    
    def get_player_expected_points(
        self,
        posterior_samples: dict,
        player_idx: int,
        position: str,
    ) -> dict:
        """
        Compute expected points distribution for a player.
        
        Args:
            posterior_samples: Predictive samples from model
            player_idx: Index of the player
            position: Player position (GKP/DEF/MID/FWD)
            
        Returns:
            Dictionary with mean, std, quantiles of expected points
        """
        # Points weights by position
        goal_pts = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}[position]
        assist_pts = 3
        defcon_pts = 2
        
        # Extract samples for this player
        goals = posterior_samples["goals"][:, player_idx]
        assists = posterior_samples["assists_obs"][:, player_idx]
        defcon = posterior_samples["defcon"][:, player_idx]
        
        # Compute points samples
        points = (
            goals * goal_pts +
            assists * assist_pts +
            defcon * defcon_pts
        )
        
        return {
            "mean": float(np.mean(points)),
            "std": float(np.std(points)),
            "q05": float(np.percentile(points, 5)),
            "q25": float(np.percentile(points, 25)),
            "q50": float(np.percentile(points, 50)),
            "q75": float(np.percentile(points, 75)),
            "q95": float(np.percentile(points, 95)),
            "samples": points,
        }
