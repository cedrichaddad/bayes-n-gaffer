"""
DefCon Back-Casting Module for Alpha-FPL.

Reconstructs historical points as if 2025/26 scoring rules (Defensive Contributions)
had applied to seasons 2021-2025.

Key transformations:
1. CBIT calculation (Tackles + Interceptions + Blocks + Clearances)
2. DefCon point addition (+2 for DEF with CBIT >= 10, +2 for MID/FWD with CBIT+Recov >= 12)
3. BPS reconstruction with 2026 weights
4. Bonus point recalculation with new BPS

This module produces the synthetic target variable Y (synthetic_total_points).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger


@dataclass
class DefConConfig:
    """Configuration for DefCon point calculation."""
    
    # CBIT thresholds
    def_cbit_threshold: int = 10
    mid_fwd_cbit_threshold: int = 12
    
    # Points awarded
    defcon_points: int = 2
    
    # 2026 BPS weights (updated from 2025/26 rules)
    bps_weights: dict = None
    
    def __post_init__(self):
        if self.bps_weights is None:
            self.bps_weights = {
                # Attacking actions
                "goals_scored": {
                    "GKP": 12, "DEF": 12, "MID": 18, "FWD": 24
                },
                "assists": 9,
                "open_play_key_passes": 1,
                "attempted_assists": 1,
                
                # Defensive actions (2026 updated)
                "tackles_won": 2,  # Updated in 2026
                "interceptions": 1,
                "blocks": 1,
                "clearances": 9,  # Updated - "G_Clearance" weight
                "recoveries": 1,
                
                # Negative actions
                "fouls_committed": -1,  # Net fouls
                "dispossessed": -1,
                "errors_leading_to_goal": -3,
                "own_goals": -6,
                "penalties_missed": -3,
                "yellow_cards": -3,
                "red_cards": -9,
                
                # GK specific
                "saves": 2,
                "penalties_saved": 15,
                
                # Clean sheet bonus
                "clean_sheet_60": {
                    "GKP": 12, "DEF": 12, "MID": 0, "FWD": 0
                },
            }


class BackCaster:
    """
    Transforms historical FPL data to use 2025/26 DefCon scoring rules.
    
    This creates a "Counterfactual History" where we know what points players
    WOULD have scored under the new rules.
    
    Algorithm:
    1. Calculate Base Points (goals × position_weight, assists × 3, etc.)
    2. Calculate CBIT = Tackles + Interceptions + Blocks + Clearances
    3. Apply DefCon logic based on position and CBIT threshold
    4. Reconstruct BPS using 2026 weights
    5. Assign bonus points {3, 2, 1} by BPS rank
    6. Compute synthetic_total_points = base + defcon + bonus
    """
    
    def __init__(self, config: DefConConfig = None):
        """
        Initialize the back-caster.
        
        Args:
            config: DefCon configuration (uses defaults if None)
        """
        self.config = config or DefConConfig()
        
        # Points per goal by position
        self.goal_points = {
            "GKP": 6, "DEF": 6, "MID": 5, "FWD": 4
        }
        
        # Clean sheet points by position
        self.cs_points = {
            "GKP": 4, "DEF": 4, "MID": 1, "FWD": 0
        }
    
    def calculate_cbit(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate CBIT (Combined Defensive Metric).
        
        CBIT = Tackles + Interceptions + Blocks + Clearances
        
        Args:
            df: DataFrame with defensive columns
            
        Returns:
            DataFrame with 'cbit' column added
        """
        # Handle missing columns gracefully
        defensive_cols = ["tackles", "interceptions", "blocks", "clearances"]
        
        for col in defensive_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))
        
        df = df.with_columns(
            (
                pl.col("tackles").fill_null(0) +
                pl.col("interceptions").fill_null(0) +
                pl.col("blocks").fill_null(0) +
                pl.col("clearances").fill_null(0)
            ).alias("cbit")
        )
        
        return df
    
    def calculate_defcon_points(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate DefCon bonus points based on position and CBIT.
        
        Rules:
        - DEF: +2 if CBIT >= 10
        - MID/FWD: +2 if (CBIT + Recoveries) >= 12
        
        Args:
            df: DataFrame with 'cbit', 'recoveries', and 'position' columns
            
        Returns:
            DataFrame with 'defcon_points' column added
        """
        # Ensure recoveries column exists
        if "recoveries" not in df.columns:
            df = df.with_columns(pl.lit(0).alias("recoveries"))
        
        # Calculate extended CBIT for non-defenders
        df = df.with_columns(
            (pl.col("cbit") + pl.col("recoveries").fill_null(0)).alias("cbit_extended")
        )
        
        # Apply DefCon rules
        df = df.with_columns(
            pl.when(
                (pl.col("position") == "DEF") &
                (pl.col("cbit") >= self.config.def_cbit_threshold)
            ).then(self.config.defcon_points)
            .when(
                (pl.col("position").is_in(["MID", "FWD"])) &
                (pl.col("cbit_extended") >= self.config.mid_fwd_cbit_threshold)
            ).then(self.config.defcon_points)
            .otherwise(0)
            .alias("defcon_points")
        )
        
        return df
    
    def calculate_base_points(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate base FPL points (excluding DefCon and bonus).
        
        Components:
        - Minutes: 1 point (1-59 mins), 2 points (60+ mins)
        - Goals: Position-dependent (6/6/5/4)
        - Assists: 3 points
        - Clean sheets: Position-dependent
        - Goals conceded: -1 per 2 for GKP/DEF
        - Saves: 1 point per 3 saves
        - Penalties saved: 5 points
        - Yellow cards: -1 point
        - Red cards: -3 points
        - Own goals: -2 points
        - Penalties missed: -2 points
        
        Args:
            df: DataFrame with match data
            
        Returns:
            DataFrame with 'base_points' column
        """
        # Minutes points
        df = df.with_columns(
            pl.when(pl.col("minutes") >= 60).then(2)
            .when(pl.col("minutes") > 0).then(1)
            .otherwise(0)
            .alias("minutes_points")
        )
        
        # Goals points (position-dependent)
        df = df.with_columns(
            pl.when(pl.col("position") == "GKP").then(
                pl.col("goals_scored").fill_null(0) * 6
            ).when(pl.col("position") == "DEF").then(
                pl.col("goals_scored").fill_null(0) * 6
            ).when(pl.col("position") == "MID").then(
                pl.col("goals_scored").fill_null(0) * 5
            ).when(pl.col("position") == "FWD").then(
                pl.col("goals_scored").fill_null(0) * 4
            ).otherwise(0)
            .alias("goal_points")
        )
        
        # Assists
        df = df.with_columns(
            (pl.col("assists").fill_null(0) * 3).alias("assist_points")
        )
        
        # Clean sheets (position-dependent, only if 60+ mins)
        df = df.with_columns(
            pl.when(
                (pl.col("minutes") >= 60) & 
                (pl.col("clean_sheet").fill_null(0) > 0)
            ).then(
                pl.when(pl.col("position") == "GKP").then(4)
                .when(pl.col("position") == "DEF").then(4)
                .when(pl.col("position") == "MID").then(1)
                .otherwise(0)
            ).otherwise(0)
            .alias("cs_points")
        )
        
        # Goals conceded penalty (GKP/DEF only, -1 per 2)
        if "goals_against" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("position").is_in(["GKP", "DEF"])).then(
                    -(pl.col("goals_against").fill_null(0) // 2)
                ).otherwise(0)
                .alias("gc_penalty")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("gc_penalty"))
        
        # Saves bonus (GKP only, 1 per 3)
        if "saves" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("position") == "GKP").then(
                    pl.col("saves").fill_null(0) // 3
                ).otherwise(0)
                .alias("save_points")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("save_points"))
        
        # Penalty saves
        if "penalties_saved" in df.columns:
            df = df.with_columns(
                (pl.col("penalties_saved").fill_null(0) * 5).alias("pen_save_points")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("pen_save_points"))
        
        # Yellow cards
        if "yellow_card" in df.columns:
            df = df.with_columns(
                (-pl.col("yellow_card").fill_null(0)).alias("yellow_penalty")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("yellow_penalty"))
        
        # Red cards
        if "red_card" in df.columns:
            df = df.with_columns(
                (-pl.col("red_card").fill_null(0) * 3).alias("red_penalty")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("red_penalty"))
        
        # Own goals
        if "own_goals" in df.columns:
            df = df.with_columns(
                (-pl.col("own_goals").fill_null(0) * 2).alias("og_penalty")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("og_penalty"))
        
        # Penalties missed
        if "penalties_missed" in df.columns:
            df = df.with_columns(
                (-pl.col("penalties_missed").fill_null(0) * 2).alias("pen_miss_penalty")
            )
        else:
            df = df.with_columns(pl.lit(0).alias("pen_miss_penalty"))
        
        # Sum all base points
        df = df.with_columns(
            (
                pl.col("minutes_points") +
                pl.col("goal_points") +
                pl.col("assist_points") +
                pl.col("cs_points") +
                pl.col("gc_penalty") +
                pl.col("save_points") +
                pl.col("pen_save_points") +
                pl.col("yellow_penalty") +
                pl.col("red_penalty") +
                pl.col("og_penalty") +
                pl.col("pen_miss_penalty")
            ).alias("base_points")
        )
        
        return df
    
    def reconstruct_bps(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Reconstruct BPS using 2026 weights.
        
        This recalculates the raw BPS score using updated defensive weights.
        
        Args:
            df: DataFrame with action columns
            
        Returns:
            DataFrame with 'bps_reconstructed' column
        """
        weights = self.config.bps_weights
        
        # Start with base BPS components
        bps_expr = pl.lit(0.0)
        
        # Tackles (updated weight: 2)
        if "tackles" in df.columns:
            bps_expr = bps_expr + (pl.col("tackles").fill_null(0) * 2)
        
        # Interceptions
        if "interceptions" in df.columns:
            bps_expr = bps_expr + (pl.col("interceptions").fill_null(0) * 1)
        
        # Blocks
        if "blocks" in df.columns:
            bps_expr = bps_expr + (pl.col("blocks").fill_null(0) * 1)
        
        # Clearances (updated weight: 9)
        if "clearances" in df.columns:
            bps_expr = bps_expr + (pl.col("clearances").fill_null(0) * 9)
        
        # Recoveries
        if "recoveries" in df.columns:
            bps_expr = bps_expr + (pl.col("recoveries").fill_null(0) * 1)
        
        # Goals - position dependent
        if "goals_scored" in df.columns:
            bps_expr = bps_expr + (
                pl.when(pl.col("position") == "FWD").then(
                    pl.col("goals_scored").fill_null(0) * 24
                ).when(pl.col("position") == "MID").then(
                    pl.col("goals_scored").fill_null(0) * 18
                ).otherwise(
                    pl.col("goals_scored").fill_null(0) * 12
                )
            )
        
        # Assists
        if "assists" in df.columns:
            bps_expr = bps_expr + (pl.col("assists").fill_null(0) * 9)
        
        # Saves (GKP)
        if "saves" in df.columns:
            bps_expr = bps_expr + (
                pl.when(pl.col("position") == "GKP").then(
                    pl.col("saves").fill_null(0) * 2
                ).otherwise(0)
            )
        
        # Negative actions
        if "fouls_committed" in df.columns:
            bps_expr = bps_expr - pl.col("fouls_committed").fill_null(0)
        
        if "own_goals" in df.columns:
            bps_expr = bps_expr - (pl.col("own_goals").fill_null(0) * 6)
        
        df = df.with_columns(bps_expr.alias("bps_reconstructed"))
        
        return df
    
    def calculate_bonus_points(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate bonus points from reconstructed BPS.
        
        Top 3 players in each match by BPS get 3, 2, 1 bonus points.
        Handles ties according to FPL rules.
        
        Args:
            df: DataFrame with 'bps_reconstructed', 'match_id', 'fpl_id'
            
        Returns:
            DataFrame with 'bonus_reconstructed' column
        """
        # Rank players within each match by BPS
        df = df.with_columns(
            pl.col("bps_reconstructed")
            .rank(method="ordinal", descending=True)
            .over(["season", "gameweek", "fixture_id"])
            .alias("bps_rank")
        )
        
        # Assign bonus: rank 1 = 3pts, rank 2 = 2pts, rank 3 = 1pt
        # Handle ties by checking for BPS ties
        df = df.with_columns(
            pl.when(pl.col("bps_rank") == 1).then(3)
            .when(pl.col("bps_rank") == 2).then(2)
            .when(pl.col("bps_rank") == 3).then(1)
            .otherwise(0)
            .alias("bonus_reconstructed")
        )
        
        return df
    
    def backcast(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Run the complete back-casting pipeline.
        
        Produces synthetic_total_points = base_points + defcon_points + bonus_reconstructed
        
        Args:
            df: Raw merged DataFrame with FPL + defensive metrics
            
        Returns:
            DataFrame with all synthetic point components and final target
        """
        logger.info("Starting DefCon back-casting pipeline...")
        
        # --- Fix for Column Mismatches ---
        # Ensure 'position' column exists (mapped from 'position_name' if needed)
        if "position" not in df.columns and "position_name" in df.columns:
            logger.info("Aliasing 'position_name' to 'position' for backcasting.")
            df = df.with_columns(pl.col("position_name").alias("position"))
        
        # Ensure defensive columns exist (fill 0 if FBref ingestion failed)
        required_def_cols = ["tackles", "interceptions", "blocks", "clearances", "recoveries"]
        missing_cols = [col for col in required_def_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing defensive columns {missing_cols}. Filling with 0.")
            df = df.with_columns([pl.lit(0).alias(col) for col in missing_cols])
        # ---------------------------------
        
        # --- FIX: Force Numeric Types ---
        # The raw CSVs contain "None" strings which makes Polars load these as String type.
        # We must cast them to Int32 (turning "None"->Null) and fill with 0 before doing math.
        numeric_cols = [
            "minutes", "goals_scored", "assists", "clean_sheet", 
            "goals_against", "saves", "penalties_saved", "yellow_card", 
            "red_card", "own_goals", "penalties_missed", "bonus", "bps"
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                # strict=False converts errors (strings) to Null
                df = df.with_columns(
                    pl.col(col).cast(pl.Int32, strict=False).fill_null(0)
                )
        # --------------------------------
        
        # Step 1: Calculate CBIT
        logger.info("Calculating CBIT...")
        df = self.calculate_cbit(df)
        
        # Step 2: Calculate DefCon points
        logger.info("Calculating DefCon points...")
        df = self.calculate_defcon_points(df)
        
        # Step 3: Calculate base points
        logger.info("Calculating base points...")
        df = self.calculate_base_points(df)
        
        # Step 4: Reconstruct BPS
        logger.info("Reconstructing BPS with 2026 weights...")
        df = self.reconstruct_bps(df)
        
        # Step 5: Calculate bonus points
        logger.info("Calculating bonus points...")
        if "fixture_id" in df.columns:
            df = self.calculate_bonus_points(df)
        else:
            # If no fixture_id, use existing bonus or 0
            if "bonus" in df.columns:
                df = df.with_columns(pl.col("bonus").alias("bonus_reconstructed"))
            else:
                df = df.with_columns(pl.lit(0).alias("bonus_reconstructed"))
        
        # Step 6: Compute synthetic total points
        logger.info("Computing synthetic total points...")
        df = df.with_columns(
            (
                pl.col("base_points") +
                pl.col("defcon_points") +
                pl.col("bonus_reconstructed")
            ).alias("synthetic_total_points")
        )
        
        # Log summary
        avg_defcon = df.select(pl.col("defcon_points").mean()).item()
        defcon_pct = df.filter(pl.col("defcon_points") > 0).height / df.height * 100
        logger.info(f"DefCon stats: avg={avg_defcon:.2f} pts, {defcon_pct:.1f}% of appearances")
        
        return df
    
    def save_backcast_data(
        self,
        df: pl.DataFrame,
        output_path: Path | str
    ) -> None:
        """
        Save back-casted dataset to parquet.
        
        Args:
            df: Back-casted DataFrame
            output_path: Path to save parquet file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.write_parquet(output_path)
        logger.info(f"Saved back-casted data: {output_path} ({len(df)} rows)")
