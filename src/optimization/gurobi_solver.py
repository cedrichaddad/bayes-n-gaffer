"""
Stochastic MIQP Solver for Alpha-FPL.

Implements the Two-Stage Stochastic Mixed-Integer Quadratic Program
using Gurobi for optimal team selection.

Objective:
    Maximize E[Returns] - λ·CVaR + γ·FutureValue
    
Subject to:
    - FPL formation and squad constraints
    - Budget constraints  
    - Transfer constraints (with hit penalties)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
from loguru import logger

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    logger.warning("Gurobi not installed. Optimization will fail.")

from .constraints import FPLConstraints, ConstraintConfig
from .objective import ObjectiveBuilder, ObjectiveConfig


@dataclass  
class SolverConfig:
    """Configuration for the MIQP solver."""
    
    # Solver settings
    time_limit: float = 300.0  # Seconds
    mip_gap: float = 0.01  # Optimality gap tolerance
    threads: int = 0  # 0 = auto
    
    # Logging
    verbose: bool = True
    log_file: str = None
    
    # Scenario settings
    n_scenarios: int = 1000
    
    # Constraint and objective configs (optional overrides)
    constraint_config: ConstraintConfig = None
    objective_config: ObjectiveConfig = None


class StochasticMIQPSolver:
    """
    Two-Stage Stochastic MIQP Solver for FPL Team Selection.
    
    Stage 1 (Here-and-Now): Select squad for current gameweek
    Stage 2 (Recourse): Adapt starting XI based on scenarios
    
    The solver handles:
    - Mean-variance optimization with CVaR risk measure
    - DRO penalty for model uncertainty
    - Multi-week planning horizon
    """
    
    def __init__(self, config: SolverConfig = None):
        """
        Initialize the solver.
        
        Args:
            config: Solver configuration
        """
        if not GUROBI_AVAILABLE:
            raise ImportError(
                "Gurobi is required for optimization. "
                "Install with: pip install gurobipy"
            )
        
        self.config = config or SolverConfig()
        
        # Initialize constraint and objective builders
        self.constraints = FPLConstraints(
            self.config.constraint_config or ConstraintConfig()
        )
        self.objective_builder = ObjectiveBuilder(
            self.config.objective_config or ObjectiveConfig()
        )
        
        # Model state
        self.model = None
        self.solution = None
    
    def _create_variables(
        self,
        model: gp.Model,
        n_players: int,
        player_names: List[str] = None,
    ) -> Dict[str, gp.tupledict]:
        """
        Create decision variables for the optimization model.
        
        Args:
            model: Gurobi model
            n_players: Number of players in pool
            player_names: Optional player names for labeling
            
        Returns:
            Dictionary of variable groups
        """
        names = player_names or [f"p{i}" for i in range(n_players)]
        
        # Squad selection (binary)
        x_squad = model.addVars(
            n_players, vtype=GRB.BINARY, name="squad"
        )
        
        # Starting XI (binary)
        x_start = model.addVars(
            n_players, vtype=GRB.BINARY, name="start"
        )
        
        # Bench (binary)
        x_bench = model.addVars(
            n_players, vtype=GRB.BINARY, name="bench"
        )
        
        # Captain (binary)
        x_captain = model.addVars(
            n_players, vtype=GRB.BINARY, name="captain"
        )
        
        # Vice captain (binary)
        x_vice = model.addVars(
            n_players, vtype=GRB.BINARY, name="vice"
        )
        
        # Transfer variables (if applicable)
        x_buy = model.addVars(
            n_players, vtype=GRB.BINARY, name="buy"
        )
        x_sell = model.addVars(
            n_players, vtype=GRB.BINARY, name="sell"
        )
        
        # Number of hits taken (integer)
        n_hits = model.addVar(vtype=GRB.INTEGER, lb=0, name="hits")
        
        # CVaR auxiliary variables
        var_threshold = model.addVar(lb=-GRB.INFINITY, name="var")
        cvar_shortfall = model.addVars(
            self.config.n_scenarios, lb=0, name="shortfall"
        )
        
        return {
            "squad": x_squad,
            "start": x_start,
            "bench": x_bench,
            "captain": x_captain,
            "vice": x_vice,
            "buy": x_buy,
            "sell": x_sell,
            "hits": n_hits,
            "var": var_threshold,
            "shortfall": cvar_shortfall,
        }
    
    def _add_squad_constraints(
        self,
        model: gp.Model,
        variables: dict,
        n_players: int,
        positions: np.ndarray,
        teams: np.ndarray,
    ) -> None:
        """Add squad composition constraints."""
        x_squad = variables["squad"]
        
        # Total squad size = 15
        model.addConstr(
            gp.quicksum(x_squad[i] for i in range(n_players)) == 15,
            name="squad_size"
        )
        
        # Position limits
        position_limits = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
        pos_names = ["GKP", "DEF", "MID", "FWD"]
        
        for pos_idx, pos_name in enumerate(pos_names):
            player_indices = np.where(positions == pos_idx)[0]
            model.addConstr(
                gp.quicksum(x_squad[i] for i in player_indices) == position_limits[pos_name],
                name=f"squad_{pos_name}"
            )
        
        # Team limits (max 3 per team)
        unique_teams = np.unique(teams)
        for team_idx in unique_teams:
            player_indices = np.where(teams == team_idx)[0]
            model.addConstr(
                gp.quicksum(x_squad[i] for i in player_indices) <= 3,
                name=f"team_{team_idx}"
            )
    
    def _add_starting_constraints(
        self,
        model: gp.Model,
        variables: dict,
        n_players: int,
        positions: np.ndarray,
    ) -> None:
        """Add starting XI constraints."""
        x_start = variables["start"]
        x_squad = variables["squad"]
        x_bench = variables["bench"]
        
        # Total starting XI = 11
        model.addConstr(
            gp.quicksum(x_start[i] for i in range(n_players)) == 11,
            name="start_size"
        )
        
        # Starting position limits
        start_limits = {
            "GKP": (1, 1),
            "DEF": (3, 5),
            "MID": (2, 5),
            "FWD": (1, 3),
        }
        pos_names = ["GKP", "DEF", "MID", "FWD"]
        
        for pos_idx, pos_name in enumerate(pos_names):
            player_indices = np.where(positions == pos_idx)[0]
            min_pos, max_pos = start_limits[pos_name]
            
            model.addConstr(
                gp.quicksum(x_start[i] for i in player_indices) >= min_pos,
                name=f"start_{pos_name}_min"
            )
            model.addConstr(
                gp.quicksum(x_start[i] for i in player_indices) <= max_pos,
                name=f"start_{pos_name}_max"
            )
        
        # Linking: start <= squad, bench <= squad
        for i in range(n_players):
            model.addConstr(x_start[i] <= x_squad[i], name=f"start_in_squad_{i}")
            model.addConstr(x_bench[i] <= x_squad[i], name=f"bench_in_squad_{i}")
        
        # Partition: start + bench = squad
        for i in range(n_players):
            model.addConstr(
                x_start[i] + x_bench[i] == x_squad[i],
                name=f"partition_{i}"
            )
        
        # Bench size
        model.addConstr(
            gp.quicksum(x_bench[i] for i in range(n_players)) == 4,
            name="bench_size"
        )
        
        # Bench must have exactly 1 GK
        gk_indices = np.where(positions == 0)[0]
        model.addConstr(
            gp.quicksum(x_bench[i] for i in gk_indices) == 1,
            name="bench_gk"
        )
    
    def _add_captain_constraints(
        self,
        model: gp.Model,
        variables: dict,
        n_players: int,
    ) -> None:
        """Add captaincy constraints."""
        x_captain = variables["captain"]
        x_vice = variables["vice"]
        x_start = variables["start"]
        
        # Exactly one captain
        model.addConstr(
            gp.quicksum(x_captain[i] for i in range(n_players)) == 1,
            name="one_captain"
        )
        
        # Exactly one vice-captain
        model.addConstr(
            gp.quicksum(x_vice[i] for i in range(n_players)) == 1,
            name="one_vice"
        )
        
        # Captain and vice must be starters
        for i in range(n_players):
            model.addConstr(x_captain[i] <= x_start[i], name=f"captain_starts_{i}")
            model.addConstr(x_vice[i] <= x_start[i], name=f"vice_starts_{i}")
        
        # Captain != Vice
        for i in range(n_players):
            model.addConstr(
                x_captain[i] + x_vice[i] <= 1,
                name=f"captain_vice_diff_{i}"
            )
    
    def _add_budget_constraint(
        self,
        model: gp.Model,
        variables: dict,
        n_players: int,
        prices: np.ndarray,
        budget: float,
    ) -> None:
        """Add budget constraint."""
        x_squad = variables["squad"]
        
        model.addConstr(
            gp.quicksum(prices[i] * x_squad[i] for i in range(n_players)) <= budget,
            name="budget"
        )
    
    def _add_transfer_constraints(
        self,
        model: gp.Model,
        variables: dict,
        n_players: int,
        current_squad: np.ndarray,
        free_transfers: int,
    ) -> None:
        """Add transfer constraints with hit logic."""
        x_squad = variables["squad"]
        x_buy = variables["buy"]
        x_sell = variables["sell"]
        n_hits = variables["hits"]
        
        # Transfer balance: squad = current + buy - sell
        for i in range(n_players):
            model.addConstr(
                x_squad[i] == current_squad[i] + x_buy[i] - x_sell[i],
                name=f"transfer_balance_{i}"
            )
        
        # Can only buy what we don't own
        for i in range(n_players):
            model.addConstr(
                x_buy[i] <= 1 - current_squad[i],
                name=f"buy_new_{i}"
            )
        
        # Can only sell what we own
        for i in range(n_players):
            model.addConstr(
                x_sell[i] <= current_squad[i],
                name=f"sell_owned_{i}"
            )
        
        # Total buys = total sells (squad size stays 15)
        model.addConstr(
            gp.quicksum(x_buy[i] for i in range(n_players)) ==
            gp.quicksum(x_sell[i] for i in range(n_players)),
            name="transfer_balance_total"
        )
        
        # Hits = max(0, buys - free_transfers)
        total_buys = gp.quicksum(x_buy[i] for i in range(n_players))
        model.addConstr(
            n_hits >= total_buys - free_transfers,
            name="hit_lower"
        )
    
    def _add_cvar_constraints(
        self,
        model: gp.Model,
        variables: dict,
        scenario_returns: np.ndarray,
        alpha: float,
    ) -> None:
        """
        Add CVaR constraints using scenario-based formulation.
        
        CVaR_α = VaR - (1/α) * E[max(VaR - X, 0)]
        """
        x_start = variables["start"]
        x_captain = variables["captain"]
        var_threshold = variables["var"]
        shortfall = variables["shortfall"]
        
        n_scenarios, n_players = scenario_returns.shape
        
        for s in range(n_scenarios):
            # Portfolio return in scenario s (including captain double)
            scenario_return = gp.quicksum(
                scenario_returns[s, i] * x_start[i] +
                scenario_returns[s, i] * x_captain[i]  # Captain bonus
                for i in range(n_players)
            )
            
            # Shortfall constraint: shortfall[s] >= VaR - return
            model.addConstr(
                shortfall[s] >= var_threshold - scenario_return,
                name=f"shortfall_{s}"
            )
    
    def _set_objective(
        self,
        model: gp.Model,
        variables: dict,
        objective_components: dict,
        hit_penalty: int = 4,
    ) -> None:
        """Set the objective function."""
        x_start = variables["start"]
        x_bench = variables["bench"]
        x_captain = variables["captain"]
        n_hits = variables["hits"]
        var_threshold = variables["var"]
        shortfall = variables["shortfall"]
        
        start_coef = objective_components["start_coefficients"]
        bench_coef = objective_components["bench_coefficients"]
        captain_coef = objective_components["captain_coefficients"]
        risk_aversion = objective_components["risk_aversion"]
        cvar_alpha = objective_components["cvar_alpha"]
        
        n_players = len(start_coef)
        n_scenarios = len(shortfall)
        
        # Expected return from starting XI
        expected_return = gp.quicksum(
            start_coef[i] * x_start[i] for i in range(n_players)
        )
        
        # Captain bonus (captain gets double points)
        captain_bonus = gp.quicksum(
            captain_coef[i] * x_captain[i] for i in range(n_players)
        )
        
        # Bench contribution (weighted by sub probability)
        bench_contribution = gp.quicksum(
            bench_coef[i] * x_bench[i] for i in range(n_players)
        )
        
        # CVaR term: VaR - (1/alpha) * mean(shortfall)
        cvar_term = var_threshold - (1 / cvar_alpha) * gp.quicksum(
            shortfall[s] / n_scenarios for s in range(n_scenarios)
        )
        
        # Hit penalty
        hit_cost = hit_penalty * n_hits
        
        # Full objective
        objective = (
            expected_return +
            captain_bonus +
            bench_contribution +
            risk_aversion * cvar_term -
            hit_cost
        )
        
        model.setObjective(objective, GRB.MAXIMIZE)
    
    def solve(
        self,
        player_data: dict,
        current_squad: np.ndarray = None,
        free_transfers: int = 1,
        budget: float = 100.0,
    ) -> dict:
        """
        Solve the optimization problem.
        
        Args:
            player_data: Dictionary containing:
                - expected_points: E[Points] per player
                - point_stds: Std[Points] per player
                - xmins_factors: Minutes adjustment
                - positions: Position indices
                - teams: Team indices
                - prices: Player prices
                - scenario_matrix: (optional) Correlated scenarios
            current_squad: Binary vector of current squad (for transfers)
            free_transfers: Available free transfers
            budget: Available budget
            
        Returns:
            Solution dictionary with selected squad, starting XI, etc.
        """
        # Extract data
        expected_points = player_data["expected_points"]
        point_stds = player_data.get("point_stds", np.ones_like(expected_points) * 2)
        xmins_factors = player_data.get("xmins_factors", np.ones_like(expected_points))
        positions = player_data["positions"]
        teams = player_data["teams"]
        prices = player_data["prices"]
        
        n_players = len(expected_points)
        
        # Build objective components
        covariance = player_data.get("covariance_matrix", np.diag(point_stds ** 2))
        scenario_matrix = player_data.get("scenario_matrix")
        
        if scenario_matrix is None:
            # Generate simple uncorrelated scenarios
            scenario_matrix = np.random.normal(
                loc=expected_points,
                scale=point_stds,
                size=(self.config.n_scenarios, n_players)
            )
        
        objective_components = self.objective_builder.build_objective_vector(
            expected_points=expected_points,
            point_variances=point_stds ** 2,
            xmins_factors=xmins_factors,
            covariance_matrix=covariance,
            scenario_matrix=scenario_matrix,
        )
        
        # Create Gurobi model
        logger.info("Creating Gurobi model...")
        model = gp.Model("AlphaFPL")
        
        # Configure solver
        model.Params.TimeLimit = self.config.time_limit
        model.Params.MIPGap = self.config.mip_gap
        model.Params.Threads = self.config.threads
        if not self.config.verbose:
            model.Params.OutputFlag = 0
        
        # Create variables
        variables = self._create_variables(model, n_players)
        
        # Add constraints
        self._add_squad_constraints(model, variables, n_players, positions, teams)
        self._add_starting_constraints(model, variables, n_players, positions)
        self._add_captain_constraints(model, variables, n_players)
        self._add_budget_constraint(model, variables, n_players, prices, budget)
        
        if current_squad is not None:
            self._add_transfer_constraints(
                model, variables, n_players, current_squad, free_transfers
            )
        
        # Add CVaR constraints
        self._add_cvar_constraints(
            model, variables, scenario_matrix,
            objective_components["cvar_alpha"]
        )
        
        # Set objective
        self._set_objective(
            model, variables, objective_components,
            hit_penalty=self.constraints.config.hit_penalty
        )
        
        # Optimize
        logger.info("Solving optimization problem...")
        model.optimize()
        
        # Extract solution
        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            solution = self._extract_solution(model, variables, n_players)
            solution["objective_value"] = model.ObjVal
            solution["gap"] = model.MIPGap
            solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "time_limit"
            
            logger.info(f"Solution found: Objective = {solution['objective_value']:.2f}")
            self.solution = solution
            return solution
        else:
            logger.error(f"Optimization failed with status {model.Status}")
            return {"status": "infeasible"}
    
    def _extract_solution(
        self,
        model: gp.Model,
        variables: dict,
        n_players: int,
    ) -> dict:
        """Extract solution from solved model."""
        # Get variable values
        squad = np.array([variables["squad"][i].X for i in range(n_players)])
        start = np.array([variables["start"][i].X for i in range(n_players)])
        bench = np.array([variables["bench"][i].X for i in range(n_players)])
        captain = np.array([variables["captain"][i].X for i in range(n_players)])
        vice = np.array([variables["vice"][i].X for i in range(n_players)])
        
        # Round to binary (handle numerical precision)
        squad = (squad > 0.5).astype(int)
        start = (start > 0.5).astype(int)
        bench = (bench > 0.5).astype(int)
        captain = (captain > 0.5).astype(int)
        vice = (vice > 0.5).astype(int)
        
        # Get indices
        squad_indices = np.where(squad == 1)[0].tolist()
        start_indices = np.where(start == 1)[0].tolist()
        bench_indices = np.where(bench == 1)[0].tolist()
        captain_idx = int(np.argmax(captain))
        vice_idx = int(np.argmax(vice))
        
        # Transfers
        buy = np.array([variables["buy"][i].X for i in range(n_players)])
        sell = np.array([variables["sell"][i].X for i in range(n_players)])
        buy_indices = np.where(buy > 0.5)[0].tolist()
        sell_indices = np.where(sell > 0.5)[0].tolist()
        hits = int(round(variables["hits"].X))
        
        return {
            "squad": squad,
            "squad_indices": squad_indices,
            "starting_xi": start,
            "starting_indices": start_indices,
            "bench": bench,
            "bench_indices": bench_indices,
            "captain_idx": captain_idx,
            "vice_captain_idx": vice_idx,
            "transfers_in": buy_indices,
            "transfers_out": sell_indices,
            "hits": hits,
        }
    
    def solve_wildcard(
        self,
        player_data: dict,
        budget: float = 100.0,
    ) -> dict:
        """
        Solve for a wildcard (unlimited transfers, no hits).
        
        Args:
            player_data: Player data dictionary
            budget: Total budget
            
        Returns:
            Optimal squad from scratch
        """
        # Remove current squad context
        return self.solve(
            player_data=player_data,
            current_squad=None,
            free_transfers=15,  # Effectively unlimited
            budget=budget,
        )
