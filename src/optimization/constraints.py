"""
FPL Constraint Definitions for Alpha-FPL Optimization.

Implements all FPL game rules as linear constraints for Gurobi:
1. Squad composition (15 players, position limits, team limits)
2. Starting XI formation validity
3. Budget constraints
4. Transfer logic (with hit penalties)
5. Captaincy rules
"""

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
from loguru import logger


@dataclass
class ConstraintConfig:
    """Configuration for FPL constraints."""
    
    # Budget
    initial_budget: float = 100.0
    max_bank: float = 5.0
    
    # Squad composition
    squad_size: int = 15
    starting_size: int = 11
    bench_size: int = 4
    
    # Position limits (squad)
    squad_position_limits: dict = None
    
    # Position limits (starting XI)
    starting_position_limits: dict = None
    
    # Team limit (max 3 from any team)
    max_per_team: int = 3
    
    # Transfers
    max_free_transfers: int = 2  # Maximum banked
    hit_penalty: int = 4  # Points per extra transfer
    
    def __post_init__(self):
        if self.squad_position_limits is None:
            self.squad_position_limits = {
                "GKP": 2,
                "DEF": 5,
                "MID": 5,
                "FWD": 3,
            }
        
        if self.starting_position_limits is None:
            # Min and max for each position in starting XI
            self.starting_position_limits = {
                "GKP": (1, 1),   # Exactly 1
                "DEF": (3, 5),   # 3-5
                "MID": (2, 5),   # 2-5
                "FWD": (1, 3),   # 1-3
            }


class FPLConstraints:
    """
    Builds constraint matrices and vectors for the MIQP.
    
    Variables structure (for N players):
    - x_squad[i]: Binary, 1 if player i in squad
    - x_start[i]: Binary, 1 if player i in starting XI
    - x_bench[i]: Binary, 1 if player i on bench
    - x_captain[i]: Binary, 1 if player i is captain
    - x_vice[i]: Binary, 1 if player i is vice-captain
    - x_buy[i]: Binary, 1 if buying player i
    - x_sell[i]: Binary, 1 if selling player i
    """
    
    def __init__(self, config: ConstraintConfig = None):
        """
        Initialize constraint builder.
        
        Args:
            config: Constraint configuration
        """
        self.config = config or ConstraintConfig()
    
    def build_squad_constraints(
        self,
        n_players: int,
        positions: np.ndarray,
        teams: np.ndarray,
    ) -> dict:
        """
        Build squad composition constraints.
        
        Args:
            n_players: Total number of players
            positions: Position index for each player (0=GKP, 1=DEF, 2=MID, 3=FWD)
            teams: Team index for each player
            
        Returns:
            Dictionary of constraint matrices/vectors
        """
        constraints = {}
        
        # 1. Total squad size = 15
        constraints["squad_size"] = {
            "A": np.ones((1, n_players)),
            "b": np.array([self.config.squad_size]),
            "sense": "="
        }
        
        # 2. Position limits for squad
        position_names = ["GKP", "DEF", "MID", "FWD"]
        for pos_idx, pos_name in enumerate(position_names):
            mask = (positions == pos_idx).astype(float)
            limit = self.config.squad_position_limits[pos_name]
            constraints[f"squad_pos_{pos_name}"] = {
                "A": mask.reshape(1, -1),
                "b": np.array([limit]),
                "sense": "="
            }
        
        # 3. Team limits (max 3 from any team)
        unique_teams = np.unique(teams)
        for team_idx in unique_teams:
            mask = (teams == team_idx).astype(float)
            constraints[f"team_limit_{team_idx}"] = {
                "A": mask.reshape(1, -1),
                "b": np.array([self.config.max_per_team]),
                "sense": "<="
            }
        
        return constraints
    
    def build_starting_constraints(
        self,
        n_players: int,
        positions: np.ndarray,
    ) -> dict:
        """
        Build starting XI constraints.
        
        Args:
            n_players: Total number of players
            positions: Position index for each player
            
        Returns:
            Dictionary of constraint matrices
        """
        constraints = {}
        
        # 1. Exactly 11 starters
        constraints["starting_size"] = {
            "A": np.ones((1, n_players)),
            "b": np.array([self.config.starting_size]),
            "sense": "="
        }
        
        # 2. Position limits for starting XI
        position_names = ["GKP", "DEF", "MID", "FWD"]
        for pos_idx, pos_name in enumerate(position_names):
            mask = (positions == pos_idx).astype(float)
            min_pos, max_pos = self.config.starting_position_limits[pos_name]
            
            # Lower bound
            constraints[f"start_pos_{pos_name}_min"] = {
                "A": mask.reshape(1, -1),
                "b": np.array([min_pos]),
                "sense": ">="
            }
            
            # Upper bound  
            constraints[f"start_pos_{pos_name}_max"] = {
                "A": mask.reshape(1, -1),
                "b": np.array([max_pos]),
                "sense": "<="
            }
        
        # 3. Linking: x_start <= x_squad (can only start if in squad)
        # This is per-player: start_i <= squad_i
        # In matrix form with combined variables: [I, -I] @ [x_start; x_squad] <= 0
        constraints["start_in_squad"] = {
            "type": "linking",
            "description": "Start players must be in squad"
        }
        
        return constraints
    
    def build_bench_constraints(
        self,
        n_players: int,
        positions: np.ndarray,
    ) -> dict:
        """
        Build bench constraints.
        
        Bench must be: 1 GK + 3 outfield, ordered by bench priority
        """
        constraints = {}
        
        # 1. Exactly 4 on bench
        constraints["bench_size"] = {
            "A": np.ones((1, n_players)),
            "b": np.array([self.config.bench_size]),
            "sense": "="
        }
        
        # 2. Bench GKP = 1
        mask = (positions == 0).astype(float)
        constraints["bench_gk"] = {
            "A": mask.reshape(1, -1),
            "b": np.array([1]),
            "sense": "="
        }
        
        # 3. Bench outfield = 3
        outfield_mask = (positions > 0).astype(float)
        constraints["bench_outfield"] = {
            "A": outfield_mask.reshape(1, -1),
            "b": np.array([3]),
            "sense": "="
        }
        
        # 4. Linking: x_bench <= x_squad and x_bench + x_start = x_squad
        constraints["bench_linking"] = {
            "type": "linking",
            "description": "Bench = Squad - Starting"
        }
        
        return constraints
    
    def build_captaincy_constraints(
        self,
        n_players: int,
    ) -> dict:
        """
        Build captaincy constraints.
        
        - Exactly 1 captain
        - Exactly 1 vice-captain
        - Both must be in starting XI
        - Cannot be same player
        """
        constraints = {}
        
        # 1. Exactly 1 captain
        constraints["captain_count"] = {
            "A": np.ones((1, n_players)),
            "b": np.array([1]),
            "sense": "="
        }
        
        # 2. Exactly 1 vice-captain  
        constraints["vice_captain_count"] = {
            "A": np.ones((1, n_players)),
            "b": np.array([1]),
            "sense": "="
        }
        
        # 3. Linking: captain must be starter
        constraints["captain_is_starter"] = {
            "type": "linking",
            "description": "Captain must be in starting XI"
        }
        
        # 4. Captain != Vice (captain + vice <= 1 for each player)
        constraints["captain_vice_different"] = {
            "type": "per_player",
            "description": "Captain and vice must be different players"
        }
        
        return constraints
    
    def build_budget_constraints(
        self,
        n_players: int,
        prices: np.ndarray,
        current_bank: float,
    ) -> dict:
        """
        Build budget constraints.
        
        Σ(price_i × squad_i) <= budget + bank
        
        Args:
            n_players: Total players
            prices: Price for each player
            current_bank: Current bank balance
            
        Returns:
            Budget constraint
        """
        available = self.config.initial_budget + current_bank
        
        return {
            "budget": {
                "A": prices.reshape(1, -1),
                "b": np.array([available]),
                "sense": "<="
            }
        }
    
    def build_transfer_constraints(
        self,
        n_players: int,
        current_squad: np.ndarray,
        free_transfers: int,
    ) -> dict:
        """
        Build transfer constraints.
        
        Transfer balance: new_squad = current_squad + buys - sells
        Transfer limit: Σ(buys) <= free_transfers + hits
        Hit penalty: points -= 4 × max(0, Σ(buys) - free_transfers)
        
        Args:
            n_players: Total players
            current_squad: Binary vector of current squad
            free_transfers: Available free transfers
            
        Returns:
            Transfer constraints
        """
        constraints = {}
        
        # 1. Transfer balance (for each player)
        # squad_new[i] = squad_old[i] + buy[i] - sell[i]
        constraints["transfer_balance"] = {
            "type": "per_player",
            "description": "Squad = Old + Buys - Sells",
            "current_squad": current_squad,
        }
        
        # 2. Can only sell what you own
        # sell[i] <= squad_old[i]
        constraints["sell_owned"] = {
            "A": np.eye(n_players),
            "b": current_squad,
            "sense": "<="
        }
        
        # 3. Can only buy what you don't own
        # buy[i] <= 1 - squad_old[i]
        constraints["buy_unowned"] = {
            "A": np.eye(n_players),
            "b": 1 - current_squad,
            "sense": "<="
        }
        
        # 4. Free transfer limit (soft constraint - penalty in objective)
        constraints["free_transfers"] = {
            "free_transfers": free_transfers,
            "hit_penalty": self.config.hit_penalty,
        }
        
        return constraints
    
    def get_all_constraints(
        self,
        n_players: int,
        positions: np.ndarray,
        teams: np.ndarray,
        prices: np.ndarray,
        current_squad: np.ndarray = None,
        current_bank: float = 0.0,
        free_transfers: int = 1,
    ) -> dict:
        """
        Get all constraints combined.
        
        Args:
            n_players: Total players in pool
            positions: Position index array
            teams: Team index array  
            prices: Price array
            current_squad: Current squad (if transferring)
            current_bank: Bank balance
            free_transfers: Available FTs
            
        Returns:
            Combined constraint dictionary
        """
        all_constraints = {}
        
        # Squad constraints
        all_constraints["squad"] = self.build_squad_constraints(
            n_players, positions, teams
        )
        
        # Starting XI constraints
        all_constraints["starting"] = self.build_starting_constraints(
            n_players, positions
        )
        
        # Bench constraints
        all_constraints["bench"] = self.build_bench_constraints(
            n_players, positions
        )
        
        # Captaincy
        all_constraints["captain"] = self.build_captaincy_constraints(
            n_players
        )
        
        # Budget
        all_constraints["budget"] = self.build_budget_constraints(
            n_players, prices, current_bank
        )
        
        # Transfers (if we have current squad)
        if current_squad is not None:
            all_constraints["transfer"] = self.build_transfer_constraints(
                n_players, current_squad, free_transfers
            )
        
        return all_constraints
    
    def validate_solution(
        self,
        x_squad: np.ndarray,
        x_start: np.ndarray,
        x_captain: np.ndarray,
        positions: np.ndarray,
        teams: np.ndarray,
        prices: np.ndarray,
        budget: float,
    ) -> tuple:
        """
        Validate a solution against all constraints.
        
        Returns:
            (is_valid, list of violations)
        """
        violations = []
        
        # Squad size
        if x_squad.sum() != 15:
            violations.append(f"Squad size: {x_squad.sum()} != 15")
        
        # Starting size
        if x_start.sum() != 11:
            violations.append(f"Starting size: {x_start.sum()} != 11")
        
        # Position limits
        for pos_idx, (pos_name, limit) in enumerate([
            ("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)
        ]):
            count = ((positions == pos_idx) & (x_squad == 1)).sum()
            if count != limit:
                violations.append(f"Squad {pos_name}: {count} != {limit}")
        
        # Team limits
        unique_teams = np.unique(teams)
        for team_idx in unique_teams:
            count = ((teams == team_idx) & (x_squad == 1)).sum()
            if count > 3:
                violations.append(f"Team {team_idx}: {count} > 3")
        
        # Budget
        total_cost = (prices * x_squad).sum()
        if total_cost > budget:
            violations.append(f"Budget: {total_cost:.1f} > {budget:.1f}")
        
        # Captain in starting XI
        if (x_captain * x_start).sum() != 1:
            violations.append("Captain not in starting XI")
        
        # Starting XI position validity
        for pos_idx, (pos_name, (min_p, max_p)) in enumerate([
            ("GKP", (1, 1)), ("DEF", (3, 5)), ("MID", (2, 5)), ("FWD", (1, 3))
        ]):
            count = ((positions == pos_idx) & (x_start == 1)).sum()
            if count < min_p or count > max_p:
                violations.append(f"Starting {pos_name}: {count} not in [{min_p}, {max_p}]")
        
        return len(violations) == 0, violations
