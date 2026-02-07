"""
Expected Minutes (xMins) Model for Alpha-FPL.

Predicts the probability of starting and expected minutes for each player.
This is critical because Points = Points_per_90 × (Mins / 90).

Key factors:
- Rest days since last match
- Yellow flag status (75% / 50% / 25%)
- Rotation risk (manager patterns)
- Fixture congestion
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger


@dataclass
class XMinsConfig:
    """Configuration for xMins prediction."""
    
    # Factor weights
    rest_days_weight: float = 0.15
    flag_status_weight: float = 0.25
    rotation_risk_weight: float = 0.20
    fixture_congestion_weight: float = 0.15
    
    # Base probability of starting by position
    base_start_prob: dict = None
    
    # Minimum appearances to estimate personal rotation risk
    min_appearances: int = 5
    
    def __post_init__(self):
        if self.base_start_prob is None:
            self.base_start_prob = {
                "GKP": 0.95,  # GKs rarely rotate
                "DEF": 0.75,
                "MID": 0.70,
                "FWD": 0.80,
            }


class XMinsModel:
    """
    Expected minutes prediction model.
    
    This is a simpler feature-based model (not Bayesian) that estimates:
    1. Probability of starting (P(start))
    2. Expected minutes given start (E[mins|start])
    3. Expected minutes = P(start) × E[mins|start]
    
    The final expected points input for optimization is:
    E[Points] = E[Points|90] × (E[Mins] / 90)
    """
    
    def __init__(self, config: XMinsConfig = None):
        """
        Initialize xMins model.
        
        Args:
            config: Model configuration
        """
        self.config = config or XMinsConfig()
        
        # Historical patterns learned from data
        self.player_patterns = {}
        self.manager_rotation_rates = {}
    
    def fit(self, df: pl.DataFrame) -> "XMinsModel":
        """
        Learn player-specific patterns from historical data.
        
        Args:
            df: DataFrame with player appearances
            
        Returns:
            Self with learned patterns
        """
        logger.info("Fitting xMins model...")
        
        # Learn per-player start rates
        player_stats = df.group_by(["fpl_id", "position"]).agg([
            pl.col("minutes").mean().alias("avg_minutes"),
            pl.col("minutes").filter(pl.col("minutes") > 0).count().alias("appearances"),
            (pl.col("minutes") >= 60).mean().alias("full_game_rate"),
            (pl.col("minutes") > 0).mean().alias("start_rate"),
        ])
        
        for row in player_stats.iter_rows(named=True):
            self.player_patterns[row["fpl_id"]] = {
                "avg_minutes": row["avg_minutes"],
                "appearances": row["appearances"],
                "full_game_rate": row["full_game_rate"] or 0,
                "start_rate": row["start_rate"] or 0,
                "position": row["position"],
            }
        
        # Learn manager rotation rates by team
        if "team_name" in df.columns:
            team_rotation = df.group_by("team_name").agg([
                (pl.col("minutes") < 60).mean().alias("rotation_rate"),
            ])
            
            for row in team_rotation.iter_rows(named=True):
                self.manager_rotation_rates[row["team_name"]] = row["rotation_rate"] or 0.2
        
        logger.info(f"Learned patterns for {len(self.player_patterns)} players")
        return self
    
    def predict_start_probability(
        self,
        player_id: int,
        position: str,
        team: str = None,
        flag_status: int = 100,  # 100 = fit, 75/50/25 = doubt
        rest_days: int = 7,
        has_fixture_congestion: bool = False,
    ) -> float:
        """
        Predict probability of starting.
        
        Args:
            player_id: FPL player ID
            position: Player position
            team: Team name
            flag_status: Current flag (100=fit, 75/50/25=doubt levels)
            rest_days: Days since last match
            has_fixture_congestion: True if midweek games exist
            
        Returns:
            Probability of starting [0, 1]
        """
        # Start with base probability for position
        base_prob = self.config.base_start_prob.get(position, 0.70)
        
        # Adjust based on player history if available
        if player_id in self.player_patterns:
            pattern = self.player_patterns[player_id]
            if pattern["appearances"] >= self.config.min_appearances:
                # Blend historical rate with base (weighted by experience)
                exp_weight = min(pattern["appearances"] / 20, 0.8)
                base_prob = exp_weight * pattern["start_rate"] + (1 - exp_weight) * base_prob
        
        # Apply adjustments
        adjustments = 1.0
        
        # Flag status penalty
        if flag_status < 100:
            flag_penalty = (100 - flag_status) / 100 * self.config.flag_status_weight
            adjustments -= flag_penalty
        
        # Rest days adjustment
        if rest_days < 3:
            rest_penalty = (3 - rest_days) / 3 * self.config.rest_days_weight
            adjustments -= rest_penalty
        
        # Fixture congestion
        if has_fixture_congestion:
            adjustments -= self.config.fixture_congestion_weight
        
        # Manager rotation risk
        if team and team in self.manager_rotation_rates:
            rotation_rate = self.manager_rotation_rates[team]
            adjustments -= rotation_rate * self.config.rotation_risk_weight
        
        # Final probability (clamped)
        prob = base_prob * adjustments
        return max(0.01, min(0.99, prob))
    
    def predict_expected_minutes(
        self,
        player_id: int,
        position: str,
        **kwargs
    ) -> dict:
        """
        Predict expected minutes and related metrics.
        
        Args:
            player_id: FPL player ID
            position: Player position
            **kwargs: Additional arguments for start probability
            
        Returns:
            Dictionary with start_prob, expected_mins_given_start, expected_mins
        """
        start_prob = self.predict_start_probability(
            player_id, position, **kwargs
        )
        
        # Expected minutes given start
        if player_id in self.player_patterns:
            pattern = self.player_patterns[player_id]
            if pattern["appearances"] >= self.config.min_appearances:
                # Use full game rate to estimate expected minutes
                full_rate = pattern["full_game_rate"]
                mins_given_start = 60 + 30 * full_rate  # Range: 60-90
            else:
                mins_given_start = 75  # Default
        else:
            mins_given_start = 75  # Default
        
        expected_mins = start_prob * mins_given_start
        
        return {
            "start_probability": start_prob,
            "expected_mins_given_start": mins_given_start,
            "expected_minutes": expected_mins,
            "minutes_adjustment_factor": expected_mins / 90,
        }
    
    def predict_batch(
        self,
        player_df: pl.DataFrame,
        gameweek_context: dict = None,
    ) -> pl.DataFrame:
        """
        Predict xMins for a batch of players.
        
        Args:
            player_df: DataFrame with player info
            gameweek_context: Dict with {player_id: {rest_days, flag_status, etc.}}
            
        Returns:
            DataFrame with xMins predictions appended
        """
        gameweek_context = gameweek_context or {}
        
        predictions = []
        for row in player_df.iter_rows(named=True):
            player_id = row.get("fpl_id")
            position = row.get("position", "MID")
            team = row.get("team_name")
            
            # Get context if available
            ctx = gameweek_context.get(player_id, {})
            
            pred = self.predict_expected_minutes(
                player_id=player_id,
                position=position,
                team=team,
                flag_status=ctx.get("flag_status", 100),
                rest_days=ctx.get("rest_days", 7),
                has_fixture_congestion=ctx.get("fixture_congestion", False),
            )
            
            predictions.append({
                "fpl_id": player_id,
                **pred
            })
        
        pred_df = pl.DataFrame(predictions)
        
        return player_df.join(pred_df, on="fpl_id", how="left")
    
    def apply_to_points(
        self,
        expected_points_per_90: float,
        expected_minutes: float,
    ) -> float:
        """
        Apply xMins adjustment to expected points.
        
        Args:
            expected_points_per_90: E[Points | 90 mins]
            expected_minutes: E[Minutes]
            
        Returns:
            Adjusted expected points
        """
        return expected_points_per_90 * (expected_minutes / 90)
