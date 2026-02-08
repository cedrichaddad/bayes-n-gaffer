"""
Feature Engineering for Alpha-FPL.

Calculates rolling averages and lagged features to prevent data leakage
and capture player form/team strength trends.

Key Features:
- Rolling average of opponent possession (lagged)
- Rolling average of CBIT (Conceded Goals, Big Chances, Interceptions, Tackles)
- Lagged team strength metrics
"""

import polars as pl
from loguru import logger


class FeatureEngineer:
    """
    Generates derived features for the Alpha-FPL models.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        # Default window sizes
        self.windows = [3, 5, 10]
    
    def generate_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate all features for the dataset.
        
        Args:
            df: Merged dataframe with raw stats
            
        Returns:
            DataFrame with additional feature columns
        """
        logger.info("Generating encoded features...")
        
        df = df.sort(["season", "gameweek"])
        
        # 1. Opponent Possession (Lagged)
        # We need the opponent's average possession from THEIR previous games
        df = self._calculate_opponent_stats(df)
        
        # 2. Player Form (Rolling averages)
        df = self._calculate_player_rolling_stats(df)
        
        # 3. Team Strength (Rolling averages)
        df = self._calculate_team_stats(df)
        
        return df
    
    def _calculate_opponent_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate lagged opponent statistics.
        
        For each match (Team A vs Team B), we want Team B's average stats
        from *prior* gameweeks.
        """
        # First, calculate rolling stats for every team
        # We need a team-level dataframe first
        if "team_name" not in df.columns or "possession" not in df.columns:
            logger.warning("Missing columns for opponent stats. Skipping.")
            return df
            
        # Group by team and match
        team_stats = df.group_by(["season", "gameweek", "team_name"]).agg([
            pl.col("possession").first().alias("team_possession"),
            pl.col("cbit").mean().alias("team_cbit_avg"), # avg of players
        ]).sort(["season", "gameweek"])
        
        # Calculate rolling averages per team
        # Shift ensure we only use PAST data (Lag 1)
        rolling_exprs = []
        for w in self.windows:
            rolling_exprs.append(
                pl.col("team_possession")
                .shift(1)
                .rolling_mean(window_size=w, min_periods=1)
                .over("season", "team_name") # Reset per season
                .alias(f"team_possession_last_{w}")
            )
            
        team_features = team_stats.with_columns(rolling_exprs)
        
        # Now join back to valid matches to get "Opponent" stats
        # The main DF has "opponent_team" column.
        # We join team_features on (season, gameweek, team_name=opponent_team)
        
        # Rename columns for the join to represent "opponent"
        opp_features = team_features.rename({
            "team_name": "opponent_team",
            "team_possession": "opp_possession_raw",
            "team_cbit_avg": "opp_cbit_avg"
        })
        
        # Rename feature columns
        for w in self.windows:
            opp_features = opp_features.rename({
                f"team_possession_last_{w}": f"opp_possession_lagged_{w}"
            })
            
        # Join
        df = df.join(
            opp_features,
            on=["season", "gameweek", "opponent_team"],
            how="left"
        )
        
        # Create the primary 'opp_possession_lagged' column (using window 5 as default)
        if "opp_possession_lagged_5" in df.columns:
            df = df.with_columns(
                pl.col("opp_possession_lagged_5").fill_null(0.5).alias("opp_possession_lagged")
            )
            
        return df

    def _calculate_player_rolling_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate rolling statistics for players (Form).
        """
        # Ensure sorted by date/gameweek within player
        df = df.sort(["fpl_id", "season", "gameweek"])
        
        metrics = ["total_points", "goals_scored", "assists", "ict_index", "cbit"]
        valid_metrics = [m for m in metrics if m in df.columns]
        
        rolling_exprs = []
        for m in valid_metrics:
            for w in self.windows:
                # Shift 1 to ensure we don't use current game's stats for prediction
                rolling_exprs.append(
                    pl.col(m)
                    .shift(1)
                    .rolling_mean(window_size=w, min_periods=1)
                    .over("fpl_id") # Reset per season? Maybe continuous is better, but simpler per season
                    .alias(f"{m}_last_{w}")
                )
        
        df = df.with_columns(rolling_exprs)
        
        # Fill nulls with 0 or mean
        # For simplicity, fill with 0 for now, or global mean later
        
        # Create 'form' column if not exists, using points_last_5
        if "total_points_last_5" in df.columns:
             df = df.with_columns(
                pl.col("total_points_last_5").fill_null(0).alias("form_calculated")
            )
            
        return df

    def _calculate_team_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate team-level rolling stats (Attack/Defense strength).
        """
        # This mirrors opponent stats but for the player's OWN team
        # (e.g. is the team creating chances?)
        return df # Placeholder for now, focus on Opponent Lagged first as that was the critical bug
