"""
Main Pipeline Runner for Alpha-FPL.

This is the entry point for running the full optimization pipeline:
1. Data ingestion and back-casting
2. Model training (Bayesian inference)
3. Copula fitting
4. Optimization
5. Backtesting

Uses Hydra for configuration management.
"""

import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(cfg: DictConfig):
    """Configure logging based on config."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=cfg.logging.format,
        level=cfg.logging.level,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main pipeline entry point.
    
    Args:
        cfg: Hydra configuration
    """
    setup_logging(cfg)
    
    logger.info("=" * 60)
    logger.info(f"Starting Alpha-FPL Pipeline: {cfg.experiment.name}")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # =========================================================================
    # Step 1: Data Ingestion
    # =========================================================================
    logger.info("Step 1: Data Ingestion")
    
    from src.data.ingest import DataIngester
    from src.data.backcast import BackCaster, DefConConfig
    
    # Initialize ingester
    ingester = DataIngester(
        raw_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
        mappings_path=Path(cfg.data.mappings_dir) / "player_id_map.json",
        seasons=list(cfg.data.seasons),
    )
    
    # Create merged dataset
    try:
        merged_df = ingester.create_merged_dataset()
        logger.info(f"Merged dataset: {len(merged_df)} rows")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise
    
    # =========================================================================
    # Step 2: Back-Casting (DefCon Points)
    # =========================================================================
    logger.info("Step 2: Back-Casting")
    
    if cfg.data.recompute_defcon:
        defcon_config = DefConConfig(
            def_cbit_threshold=cfg.data.defcon.def_cbit_threshold,
            mid_fwd_cbit_threshold=cfg.data.defcon.mid_fwd_cbit_threshold,
            defcon_points=cfg.data.defcon.defcon_points,
        )
        
        backcaster = BackCaster(config=defcon_config)
        backcast_df = backcaster.backcast(merged_df)
        
        # Save
        output_path = Path(cfg.data.processed_dir) / "backcast_history.parquet"
        backcaster.save_backcast_data(backcast_df, output_path)
    else:
        # Load existing
        backcast_df = merged_df
    
    # =========================================================================
    # Step 3: Model Training
    # =========================================================================
    logger.info("Step 3: Bayesian Model Training")
    
    from src.models.numpyro_model import HierarchicalPointsModel, ModelConfig
    from src.models.xmins_model import XMinsModel
    from src.models.copula import TCopulaEngine, CopulaConfig
    
    # Configure model
    model_config = ModelConfig(
        chains=cfg.model.chains,
        samples=cfg.model.samples,
        warmup=cfg.model.warmup,
        alpha_sigma=cfg.model.priors.alpha_sigma,
        team_sigma=cfg.model.priors.team_sigma,
        time_decay_kappa=cfg.model.time_decay_kappa,
        device=cfg.model.device,
    )
    
    # Initialize and fit model
    model = HierarchicalPointsModel(config=model_config)
    
    # Prepare training data
    train_seasons = [s for s in cfg.data.seasons if s not in cfg.backtest.validation_seasons]
    train_df = backcast_df.filter(
        backcast_df["season"].is_in(train_seasons)
    )
    
    logger.info(f"Training on {len(train_df)} rows from seasons: {train_seasons}")
    
    # Fit model
    import jax
    rng_key = jax.random.PRNGKey(cfg.experiment.seed)
    
    try:
        train_data = model.prepare_data(train_df)
        model.fit(train_data, rng_key=rng_key)
        logger.info("Bayesian model training complete")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise
    
    # =========================================================================
    # Step 4: xMins Model
    # =========================================================================
    logger.info("Step 4: xMins Model Training")
    
    xmins_model = XMinsModel()
    xmins_model.fit(backcast_df)
    
    # =========================================================================
    # Step 5: Copula Fitting
    # =========================================================================
    logger.info("Step 5: Copula Fitting")
    
    copula_config = CopulaConfig(
        n_scenarios=cfg.copula.scenarios,
        min_observations=cfg.copula.min_observations,
        initial_df=cfg.copula.initial_df,
    )
    
    copula = TCopulaEngine(config=copula_config)
    
    # Fit copula on residuals (simplified - use backcast data directly)
    if "synthetic_total_points" in backcast_df.columns:
        try:
            # Get actual vs predicted
            actuals = backcast_df.pivot(
                index=["season", "gameweek"],
                on="fpl_id",
                values="synthetic_total_points",
            ).to_numpy()[:, 1:]  # Remove index columns
            
            predicted = backcast_df.pivot(
                index=["season", "gameweek"],
                on="fpl_id",
                values="base_points",  # Use base as "predicted"
            ).to_numpy()[:, 1:]
            
            # Handle NaN
            mask = ~np.isnan(actuals).any(axis=1)
            copula.fit(actuals[mask], predicted[mask])
        except Exception as e:
            logger.warning(f"Copula fitting failed, using uncorrelated: {e}")
    
    # =========================================================================
    # Step 6: Optimization Setup
    # =========================================================================
    logger.info("Step 6: Optimization Setup")
    
    from src.optimization.gurobi_solver import StochasticMIQPSolver, SolverConfig
    from src.optimization.constraints import ConstraintConfig
    from src.optimization.objective import ObjectiveConfig
    
    constraint_config = ConstraintConfig(
        initial_budget=cfg.optimization.initial_budget,
        hit_penalty=cfg.optimization.hit_penalty,
    )
    
    objective_config = ObjectiveConfig(
        risk_aversion_lambda=cfg.optimization.risk_aversion_lambda,
        cvar_alpha=cfg.optimization.cvar_alpha,
        dro_radius=cfg.optimization.dro_radius,
        bench_auto_sub_prob=cfg.optimization.bench_auto_sub_prob,
    )
    
    solver_config = SolverConfig(
        n_scenarios=cfg.copula.scenarios,
        constraint_config=constraint_config,
        objective_config=objective_config,
    )
    
    solver = StochasticMIQPSolver(config=solver_config)
    
    # =========================================================================
    # Step 7: Backtesting
    # =========================================================================
    logger.info("Step 7: Walk-Forward Backtest")
    
    from src.backtest.runner import BacktestRunner, BacktestConfig
    
    backtest_config = BacktestConfig(
        start_gameweek=cfg.backtest.start_gameweek,
        validation_seasons=list(cfg.backtest.validation_seasons),
        wandb_enabled=cfg.wandb.enabled,
        wandb_project=cfg.wandb.project,
    )
    
    backtester = BacktestRunner(
        model=model,
        copula=copula,
        solver=solver,
        config=backtest_config,
    )
    
    # Run backtest for each validation season
    for season in cfg.backtest.validation_seasons:
        logger.info(f"Backtesting season {season}...")
        
        season_data = backcast_df.filter(backcast_df["season"] == season)
        
        if len(season_data) == 0:
            logger.warning(f"No data for season {season}")
            continue
        
        metrics = backtester.run_season(
            season=season,
            player_data=season_data,
            results_data=season_data,  # Actuals
        )
        
        logger.info(f"Season {season} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.2f}")
    
    # Save results
    backtester.save_results()
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    
    final_metrics = backtester._compute_metrics()
    logger.info(f"Total Points: {final_metrics.get('total_points', 0):.0f}")
    logger.info(f"Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.3f}")
    logger.info(f"CRPS: {final_metrics.get('crps', 0):.2f}")
    
    return final_metrics


if __name__ == "__main__":
    import numpy as np
    main()
