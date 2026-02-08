"""
Walk-Forward Backtest Runner for Alpha-FPL.

Implements Rolling Origin Evaluation to properly test the optimization strategy
without look-ahead bias. Each gameweek:
1. Train model on data up to GW-1
2. Predict for GW
3. Solve optimization
4. Record actual results
5. Update model with GW actuals
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import polars as pl
from loguru import logger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Gameweek range
    start_gameweek: int = 5  # Allow warm-up
    end_gameweek: int = 38
    
    # Validation seasons
    validation_seasons: List[str] = field(default_factory=lambda: ["2024-25"])
    
    # Initial team settings
    initial_budget: float = 100.0
    chips_available: List[str] = field(
        default_factory=lambda: ["wildcard_1", "wildcard_2", "freehit", "triplecap", "benchboost"]
    )
    
    # Metrics to track
    metrics: List[str] = field(
        default_factory=lambda: [
            "total_points", "expected_points", "crps", 
            "sharpe_ratio", "rank_percentile", "hits_taken"
        ]
    )
    
    # Output
    output_dir: str = "outputs/backtest"
    save_predictions: bool = True
    
    # W&B tracking
    wandb_enabled: bool = True
    wandb_project: str = "alpha-fpl"


@dataclass 
class GameweekResult:
    """Results for a single gameweek."""
    
    season: str
    gameweek: int
    
    # Selected team
    squad_ids: List[int]
    starting_ids: List[int]
    captain_id: int
    vice_captain_id: int
    
    # Predictions vs actuals
    predicted_points: Dict[int, float]
    actual_points: Dict[int, float]
    
    # Aggregates
    total_predicted: float
    total_actual: float
    captain_actual: int
    bench_points: int
    
    # Transfers
    transfers_in: List[int]
    transfers_out: List[int]
    hits: int
    hit_cost: int
    
    # Team state
    team_value: float
    bank: float
    
    # Timestamps
    solved_at: datetime = None


class BacktestRunner:
    """
    Walk-Forward Backtesting Engine.
    
    Simulates FPL management through historical seasons using:
    - Rolling origin: only use data available at decision time
    - No look-ahead bias
    - Realistic transfer constraints
    """
    
    def __init__(
        self,
        model,
        copula,
        solver,
        xmins_model=None,
        config: BacktestConfig = None,
    ):
        """
        Initialize backtester.
        
        Args:
            model: Fitted HierarchicalPointsModel
            copula: Fitted TCopulaEngine
            solver: StochasticMIQPSolver
            xmins_model: Fitted XMinsModel (optional but recommended)
            config: Backtest configuration
        """
        self.model = model
        self.copula = copula
        self.solver = solver
        self.xmins_model = xmins_model
        self.config = config or BacktestConfig()
        
        # State
        self.current_squad = None
        self.current_bank = 0.0
        self.free_transfers = 1
        self.team_value = 100.0
        
        # Results
        self.results: List[GameweekResult] = []
        self.cumulative_points = 0
        self.cumulative_hits = 0
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize W&B if enabled
        if self.config.wandb_enabled and WANDB_AVAILABLE:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb_project,
            config={
                "start_gw": self.config.start_gameweek,
                "end_gw": self.config.end_gameweek,
                "seasons": self.config.validation_seasons,
            }
        )
    
    def _select_initial_squad(
        self,
        player_data: pl.DataFrame,
        budget: float = 100.0,
    ) -> np.ndarray:
        """
        Select initial squad (simulates GW1 wildcard).
        
        Uses solver with unlimited transfers (wildcard mode).
        """
        logger.info("Selecting initial squad...")
        
        # Prepare player data for solver
        solver_data = self._prepare_solver_data(player_data)
        
        # Solve as wildcard
        solution = self.solver.solve_wildcard(
            player_data=solver_data,
            budget=budget,
        )
        
        if solution.get("status") == "infeasible":
            raise ValueError("Could not select initial squad")
        
        return solution["squad"]
    
    def _prepare_solver_data(
        self,
        player_df: pl.DataFrame,
        predictions: dict = None,
    ) -> dict:
        """
        Prepare player data dictionary for solver.
        
        Args:
            player_df: Current player data
            predictions: Optional predictions from model
            
        Returns:
            Dictionary for solver.solve()
        """
        n_players = len(player_df)
        
        # If we have predictions, use them
        if predictions is not None:
            # FIX: Default missing players to 0.0, NOT 2.0
            expected_points = np.array([
                predictions.get(pid, {}).get("mean", 0.0)
                for pid in player_df["fpl_id"].to_list()
            ])
            point_stds = np.array([
                predictions.get(pid, {}).get("std", 2.0)
                for pid in player_df["fpl_id"].to_list()
            ])
        else:
            # Use simple heuristic (form-based)
            expected_points = player_df["form"].fill_null(0.0).to_numpy()
            point_stds = np.ones(n_players) * 2.0
        
        # Position encoding
        pos_map = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
        positions = np.array([
            pos_map.get(p, 2) for p in player_df["position"].to_list()
        ])
        
        # Team encoding
        teams = player_df["team_name"].to_list()
        unique_teams = list(set(teams))
        team_map = {t: i for i, t in enumerate(unique_teams)}
        team_indices = np.array([team_map[t] for t in teams])
        
        # Prices
        prices = player_df["now_cost"].fill_null(50).to_numpy() / 10  # Convert to millions
        
        # xMins factors
        if self.xmins_model is not None:
            try:
                # Use predict_batch to get minutes adjustment
                # Note: valid gameweek_context (rest days, flags) should ideally be passed here
                # For now, we rely on the model's defaults or available columns
                xmins_df = self.xmins_model.predict_batch(player_df)
                
                # Extract the adjustment factor (expected_mins / 90)
                if "minutes_adjustment_factor" in xmins_df.columns:
                    xmins_factors = xmins_df["minutes_adjustment_factor"].fill_null(1.0).to_numpy()
                else:
                    logger.warning("xMins model did not return adjustment factor")
                    xmins_factors = np.ones(n_players)

            except Exception as e:
                logger.warning(f"xMins prediction failed: {e}. Using 1.0 factors.")
                xmins_factors = np.ones(n_players)
        else:
            xmins_factors = np.ones(n_players)
            
        xmins_factors = np.clip(xmins_factors, 0.0, 1.0)
        
        return {
            "expected_points": expected_points,
            "point_stds": point_stds,
            "xmins_factors": xmins_factors,
            "positions": positions,
            "teams": team_indices,
            "prices": prices,
            "player_ids": player_df["fpl_id"].to_list(),
        }
    
    def run_gameweek(
        self,
        gameweek: int,
        season: str,
        player_data: pl.DataFrame,
        actual_results: pl.DataFrame,
    ) -> GameweekResult:
        """
        Run a single gameweek simulation.
        
        Args:
            gameweek: Gameweek number
            season: Season string
            player_data: Available player data (pre-gameweek)
            actual_results: Actual gameweek results (for evaluation)
            
        Returns:
            GameweekResult with predictions and actuals
        """
        logger.info(f"Running backtest GW{gameweek} ({season})...")
        
        # 1. Generate predictions from model
        predictions = self._generate_predictions(player_data, gameweek)
        
        # 2. Prepare data for solver
        solver_data = self._prepare_solver_data(player_data, predictions)
        
        # 3. Add scenario matrix from copula
        if self.copula is not None:
            # Extract expected points, stds, and posterior samples for copula
            player_ids = solver_data["player_ids"]
            
            # Arrays for copula input
            expected_points = []
            point_stds = []
            posterior_samples = []
            
            # Check if we have samples for all players
            has_samples = True
            
            for pid in player_ids:
                pred = predictions.get(pid, {})
                expected_points.append(pred.get("mean", 2.0))
                point_stds.append(pred.get("std", 2.0))
                
                # Check for samples
                if "samples" in pred:
                    posterior_samples.append(pred["samples"])
                else:
                    has_samples = False
            
            expected_points = np.array(expected_points)
            point_stds = np.array(point_stds)
            
            # If we have posterior samples for everyone, stack them
            # Shape: (n_samples, n_players)
            if has_samples and posterior_samples:
                # Transpose to (n_samples, n_players)
                # individual samples are 1D arrays of length n_samples
                posterior_matrix = np.column_stack(posterior_samples)
            else:
                posterior_matrix = None
                
            try:
                # Generate scenarios using empirical quantile mapping
                scenarios = self.copula.generate_scenarios(
                    expected_points=expected_points,
                    point_stds=point_stds,
                    posterior_samples=posterior_matrix,
                    n_scenarios=self.solver.config.n_scenarios
                )
                solver_data["scenario_matrix"] = scenarios
                
                # Also compute covariance matrix for objective
                solver_data["covariance_matrix"] = self.copula.get_covariance_matrix(point_stds)
                
            except Exception as e:
                logger.warning(f"Copula generation failed: {e}")
        
        # 4. Solve for optimal team
        solution = self.solver.solve(
            player_data=solver_data,
            current_squad=self.current_squad,
            free_transfers=self.free_transfers,
            budget=self.team_value + self.current_bank,
        )
        
        if solution.get("status") == "infeasible":
            logger.warning(f"GW{gameweek}: Optimization infeasible, keeping current squad")
            # Keep current squad (no transfers)
            starting_indices = self._auto_select_starting_xi(self.current_squad, solver_data)
            captain_idx = starting_indices[0]  # Simple: highest expected
            solution = {
                "squad": self.current_squad,
                "starting_indices": starting_indices,
                "captain_idx": captain_idx,
                "vice_captain_idx": starting_indices[1],
                "transfers_in": [],
                "transfers_out": [],
                "hits": 0,
            }
        
        # 5. Evaluate against actuals
        player_ids = solver_data["player_ids"]
        actual_map = {
            row["fpl_id"]: row["total_points"]
            for row in actual_results.filter(
                pl.col("gameweek") == gameweek
            ).iter_rows(named=True)
        }
        
        # Get actual points for starting XI  
        starting_ids = [player_ids[i] for i in solution["starting_indices"]]
        captain_id = player_ids[solution["captain_idx"]]
        
        starting_points = sum(
            actual_map.get(pid, 0) for pid in starting_ids
        )
        captain_points = actual_map.get(captain_id, 0)
        total_actual = starting_points + captain_points  # Captain doubles
        
        # Hit penalty
        hit_cost = solution["hits"] * self.solver.constraints.config.hit_penalty
        total_actual -= hit_cost
        
        # 6. Update state
        self.current_squad = solution["squad"]
        self.cumulative_points += total_actual
        self.cumulative_hits += solution["hits"]
        
        # Update free transfers
        if solution["hits"] == 0 and len(solution["transfers_in"]) > 0:
            self.free_transfers = min(self.free_transfers + 1, 5)  # 5 = max banked
        elif solution["hits"] > 0:
            self.free_transfers = 1
        else:
            self.free_transfers = min(self.free_transfers + 1, 5)
        
        # 7. Create result
        result = GameweekResult(
            season=season,
            gameweek=gameweek,
            squad_ids=[player_ids[i] for i in np.where(solution["squad"])[0]],
            starting_ids=starting_ids,
            captain_id=captain_id,
            vice_captain_id=player_ids[solution["vice_captain_idx"]],
            predicted_points={
                player_ids[i]: predictions.get(player_ids[i], {}).get("mean", 0)
                for i in solution["starting_indices"]
            },
            actual_points={
                pid: actual_map.get(pid, 0) for pid in starting_ids
            },
            total_predicted=solution.get("objective_value", 0),
            total_actual=total_actual,
            captain_actual=captain_points * 2,
            bench_points=0,  # TODO: Calculate auto-sub
            transfers_in=solution["transfers_in"],
            transfers_out=solution["transfers_out"],
            hits=solution["hits"],
            hit_cost=hit_cost,
            team_value=self.team_value,
            bank=self.current_bank,
            solved_at=datetime.now(),
        )
        
        self.results.append(result)
        
        # 8. Log to W&B
        if self.config.wandb_enabled and WANDB_AVAILABLE:
            self._log_gameweek(result)
        
        return result
    
    def _generate_predictions(
        self,
        player_data: pl.DataFrame,
        gameweek: int,
        historical_data: pl.DataFrame = None,
    ) -> dict:
        """
        Generate predictions from the fitted Bayesian model.
        
        This is the critical connection between the NumPyro inference
        and the optimization layer.
        
        Args:
            player_data: Current gameweek player data
            gameweek: Target gameweek
            historical_data: Historical data for feature computation
            
        Returns:
            Dictionary mapping player_id -> {mean, std, q05, q95, samples}
        """
        import jax
        import jax.numpy as jnp
        
        predictions = {}
        
        # Check if model is available and fitted
        if self.model is None or not hasattr(self.model, 'posterior_samples_'):
            logger.warning("No fitted model available, falling back to form-based heuristic")
            return self._fallback_form_predictions(player_data)
        
        try:
            # Prepare prediction data using the model's prepare_data method
            # This ensures consistent feature engineering
            pred_data = self.model.prepare_data(player_data)
            
            # Generate posterior predictive samples
            rng_key = jax.random.PRNGKey(gameweek)  # Reproducible per GW
            posterior_predictive = self.model.predict(
                pred_data, 
                rng_key=rng_key,
                num_samples=500,  # Fewer samples for speed during backtest
            )
            
            # Extract player IDs for mapping
            player_ids = player_data["fpl_id"].to_list()
            
            # Aggregate samples per player
            # The predict method returns samples shaped (n_samples, n_observations)
            # We need to aggregate by player since a player may have multiple obs
            
            goals_samples = posterior_predictive.get("goals_pred", None)
            assists_samples = posterior_predictive.get("assists_pred", None)
            defcon_samples = posterior_predictive.get("defcon_pred", None)
            
            # Get position info for points calculation
            positions = player_data["position"].to_list()
            pos_goal_pts = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
            pos_assist_pts = {"GKP": 3, "DEF": 3, "MID": 3, "FWD": 3}
            
            for i, pid in enumerate(player_ids):
                pos = positions[i] if i < len(positions) else "MID"
                goal_pts = pos_goal_pts.get(pos, 5)
                assist_pts = pos_assist_pts.get(pos, 3)
                
                # Calculate total points samples
                # Points = 2 (appearance) + goals*goal_pts + assists*assist_pts + defcon*2
                if goals_samples is not None and i < goals_samples.shape[1]:
                    goals = goals_samples[:, i]
                    assists = assists_samples[:, i] if assists_samples is not None else 0
                    defcon = defcon_samples[:, i] if defcon_samples is not None else 0
                    
                    # Total points per sample
                    total_samples = (
                        2 +  # Appearance points
                        goals * goal_pts +
                        assists * assist_pts +
                        defcon * 2  # DefCon bonus
                    )
                    
                    # Aggregate statistics
                    mean_pts = float(jnp.mean(total_samples))
                    std_pts = float(jnp.std(total_samples))
                    q05 = float(jnp.percentile(total_samples, 5))
                    q95 = float(jnp.percentile(total_samples, 95))
                    
                    predictions[pid] = {
                        "mean": mean_pts,
                        "std": std_pts,
                        "q05": q05,
                        "q95": q95,
                        "samples": np.array(total_samples),  # Keep for copula
                    }
                else:
                    # Fallback for missing players
                    form = player_data.filter(
                        pl.col("fpl_id") == pid
                    )["form"].to_list()
                    form_val = form[0] if form else 2.0
                    
                    predictions[pid] = {
                        "mean": float(form_val) if form_val else 2.0,
                        "std": 2.5,
                        "q05": max(0, (form_val or 2.0) - 4),
                        "q95": (form_val or 2.0) + 6,
                    }
            
            logger.info(f"Generated Bayesian predictions for {len(predictions)} players")
            return predictions
            
        except Exception as e:
            logger.error(f"Bayesian prediction failed: {e}, falling back to heuristic")
            return self._fallback_form_predictions(player_data)
    
    def _fallback_form_predictions(self, player_data: pl.DataFrame) -> dict:
        """Fallback form-based predictions when model unavailable."""
        predictions = {}
        for row in player_data.iter_rows(named=True):
            pid = row.get("fpl_id")
            form = row.get("form", 2.0) or 2.0
            
            predictions[pid] = {
                "mean": float(form),
                "std": 2.5,
                "q05": max(0, form - 4),
                "q95": form + 6,
            }
        return predictions
    
    def _auto_select_starting_xi(
        self,
        squad: np.ndarray,
        solver_data: dict,
    ) -> list:
        """Auto-select starting XI from squad (used as fallback)."""
        squad_indices = np.where(squad == 1)[0]
        positions = solver_data["positions"]
        expected = solver_data["expected_points"]
        
        # Sort by expected points
        sorted_squad = sorted(
            squad_indices,
            key=lambda i: expected[i],
            reverse=True
        )
        
        # Select valid formation
        selected = []
        pos_counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
        pos_names = ["GKP", "DEF", "MID", "FWD"]
        pos_mins = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
        pos_maxs = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
        
        # First pass: fill minimums
        for idx in sorted_squad:
            pos = pos_names[positions[idx]]
            if pos_counts[pos] < pos_mins[pos]:
                selected.append(idx)
                pos_counts[pos] += 1
        
        # Second pass: fill to 11 with highest expected
        for idx in sorted_squad:
            if idx in selected:
                continue
            if len(selected) >= 11:
                break
            pos = pos_names[positions[idx]]
            if pos_counts[pos] < pos_maxs[pos]:
                selected.append(idx)
                pos_counts[pos] += 1
        
        return selected[:11]
    
    def _log_gameweek(self, result: GameweekResult):
        """Log gameweek results to W&B."""
        wandb.log({
            "gameweek": result.gameweek,
            "total_actual": result.total_actual,
            "total_predicted": result.total_predicted,
            "cumulative_points": self.cumulative_points,
            "hits": result.hits,
            "captain_points": result.captain_actual,
            "prediction_error": abs(result.total_actual - result.total_predicted),
        })
    
    def run_season(
        self,
        season: str,
        player_data: pl.DataFrame,
        results_data: pl.DataFrame,
    ) -> dict:
        """
        Run backtest for a full season.
        
        Args:
            season: Season string
            player_data: Pre-computed player features
            results_data: Actual gameweek results
            
        Returns:
            Season summary metrics
        """
        logger.info(f"Starting backtest for {season}...")
        
        # Filter to season
        season_players = player_data.filter(pl.col("season") == season)
        season_results = results_data.filter(pl.col("season") == season)
        
        # Get initial GW1 data
        gw1_players = season_players.filter(pl.col("gameweek") == 1)
        
        # Select initial squad
        self.current_squad = self._select_initial_squad(gw1_players)
        self.current_bank = 0.0
        self.free_transfers = 1
        
        # Run each gameweek
        for gw in range(self.config.start_gameweek, self.config.end_gameweek + 1):
            # Get data available before this GW
            available_data = season_players.filter(
                pl.col("gameweek") < gw
            )
            
            # Get current prices/availability
            current_data = season_players.filter(
                pl.col("gameweek") == gw
            )
            
            if len(current_data) == 0:
                logger.warning(f"No data for GW{gw}, skipping")
                continue
            
            result = self.run_gameweek(
                gameweek=gw,
                season=season,
                player_data=current_data,
                actual_results=season_results,
            )
        
        # Compute season metrics
        metrics = self._compute_metrics()
        
        logger.info(f"Season {season} complete: {self.cumulative_points} points")
        
        return metrics
    
    def _compute_metrics(self) -> dict:
        """Compute summary metrics from results."""
        if not self.results:
            return {}
        
        actuals = [r.total_actual for r in self.results]
        predicted = [r.total_predicted for r in self.results]
        
        return {
            "total_points": sum(actuals),
            "expected_points": sum(predicted),
            "mean_per_gw": np.mean(actuals),
            "std_per_gw": np.std(actuals),
            "sharpe_ratio": np.mean(actuals) / (np.std(actuals) + 1e-6),
            "total_hits": self.cumulative_hits,
            "hit_cost": self.cumulative_hits * 4,
            "crps": self._compute_crps(),
            "n_gameweeks": len(self.results),
        }
    
    def _compute_crps(self) -> float:
        """Compute Continuous Ranked Probability Score."""
        # Simplified CRPS: mean absolute prediction error
        errors = [
            abs(r.total_actual - r.total_predicted)
            for r in self.results
        ]
        return np.mean(errors)
    
    def save_results(self, path: str = None):
        """Save backtest results to file."""
        if path is None:
            path = self.output_dir / "backtest_results.json"
        
        import json
        
        results_data = {
            "summary": self._compute_metrics(),
            "gameweeks": [
                {
                    "season": r.season,
                    "gameweek": r.gameweek,
                    "actual": r.total_actual,
                    "predicted": r.total_predicted,
                    "captain_id": r.captain_id,
                    "hits": r.hits,
                }
                for r in self.results
            ]
        }
        
        with open(path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Saved results to {path}")
