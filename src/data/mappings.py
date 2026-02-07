"""
Static ID Mapping Loader for Alpha-FPL.

Handles loading and validation of pre-computed FBref ↔ FPL player ID mappings.
This module does NOT perform fuzzy matching at runtime - that's done by tools/generate_map.py.

The mapping file (data/meta/player_id_map.json) must be generated once per season
and manually verified for the ~10-20 ambiguous cases.
"""

import json
from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger


class IDMapper:
    """
    Loads and manages static player ID mappings between data sources.
    
    CRITICAL: This class fails hard if a required mapping is missing.
    No fuzzy matching is done at runtime to prevent silent errors
    (e.g., "Gabriel Magalhães" matching to "Gabriel Martinelli").
    
    Attributes:
        mappings: Dict mapping (source, source_id) -> fpl_id
        reverse_mappings: Dict mapping fpl_id -> {source: source_id}
    """
    
    def __init__(self, mappings_path: Path | str):
        """
        Load ID mappings from JSON file.
        
        Args:
            mappings_path: Path to player_id_map.json
            
        Raises:
            FileNotFoundError: If mappings file doesn't exist
            ValueError: If mappings file is malformed
        """
        self.mappings_path = Path(mappings_path)
        
        if not self.mappings_path.exists():
            raise FileNotFoundError(
                f"ID mappings file not found: {self.mappings_path}\n"
                f"Run 'python tools/generate_map.py' to create it."
            )
        
        self._load_mappings()
        logger.info(f"Loaded {len(self.fpl_to_fbref)} FPL ↔ FBref mappings")
        logger.info(f"Loaded {len(self.fpl_to_understat)} FPL ↔ Understat mappings")
    
    def _load_mappings(self) -> None:
        """Load and parse the JSON mappings file."""
        with open(self.mappings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        required_keys = ["fpl_to_fbref", "fpl_to_understat", "metadata"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Mappings file missing required key: {key}")
        
        self.fpl_to_fbref: dict[int, str] = {
            int(k): v for k, v in data["fpl_to_fbref"].items()
        }
        self.fpl_to_understat: dict[int, int] = {
            int(k): int(v) for k, v in data["fpl_to_understat"].items()
        }
        
        # Reverse mappings for convenience
        self.fbref_to_fpl: dict[str, int] = {v: k for k, v in self.fpl_to_fbref.items()}
        self.understat_to_fpl: dict[int, int] = {v: k for k, v in self.fpl_to_understat.items()}
        
        # Metadata
        self.metadata = data["metadata"]
        self.season = self.metadata.get("season", "unknown")
        self.generated_at = self.metadata.get("generated_at", "unknown")
    
    def get_fbref_id(self, fpl_id: int) -> str:
        """
        Get FBref player ID from FPL ID.
        
        Args:
            fpl_id: FPL player ID
            
        Returns:
            FBref player ID string
            
        Raises:
            KeyError: If mapping not found (fail hard!)
        """
        if fpl_id not in self.fpl_to_fbref:
            raise KeyError(
                f"No FBref mapping found for FPL ID {fpl_id}. "
                f"Add to tools/generate_map.py and regenerate."
            )
        return self.fpl_to_fbref[fpl_id]
    
    def get_understat_id(self, fpl_id: int) -> int:
        """
        Get Understat player ID from FPL ID.
        
        Args:
            fpl_id: FPL player ID
            
        Returns:
            Understat player ID
            
        Raises:
            KeyError: If mapping not found (fail hard!)
        """
        if fpl_id not in self.fpl_to_understat:
            raise KeyError(
                f"No Understat mapping found for FPL ID {fpl_id}. "
                f"Add to tools/generate_map.py and regenerate."
            )
        return self.fpl_to_understat[fpl_id]
    
    def get_fpl_id_from_fbref(self, fbref_id: str) -> int:
        """Get FPL ID from FBref ID."""
        if fbref_id not in self.fbref_to_fpl:
            raise KeyError(f"No FPL mapping found for FBref ID {fbref_id}")
        return self.fbref_to_fpl[fbref_id]
    
    def get_fpl_id_from_understat(self, understat_id: int) -> int:
        """Get FPL ID from Understat ID."""
        if understat_id not in self.understat_to_fpl:
            raise KeyError(f"No FPL mapping found for Understat ID {understat_id}")
        return self.understat_to_fpl[understat_id]
    
    def has_fbref_mapping(self, fpl_id: int) -> bool:
        """Check if FBref mapping exists for FPL ID."""
        return fpl_id in self.fpl_to_fbref
    
    def has_understat_mapping(self, fpl_id: int) -> bool:
        """Check if Understat mapping exists for FPL ID."""
        return fpl_id in self.fpl_to_understat
    
    def merge_fbref_data(
        self,
        fpl_df: pl.DataFrame,
        fbref_df: pl.DataFrame,
        fpl_id_col: str = "fpl_id",
        fbref_id_col: str = "player_id",
    ) -> pl.DataFrame:
        """
        Merge FBref data into FPL DataFrame using static mappings.
        
        Args:
            fpl_df: FPL DataFrame with fpl_id column
            fbref_df: FBref DataFrame with player_id column
            fpl_id_col: Name of FPL ID column
            fbref_id_col: Name of FBref ID column
            
        Returns:
            Merged DataFrame (left join on FPL)
        """
        # Create mapping DataFrame
        mapping_df = pl.DataFrame({
            "fpl_id": list(self.fpl_to_fbref.keys()),
            "fbref_id": list(self.fpl_to_fbref.values()),
        })
        
        # Add FBref ID to FPL data
        fpl_with_fbref = fpl_df.join(
            mapping_df,
            left_on=fpl_id_col,
            right_on="fpl_id",
            how="left"
        )
        
        # Merge FBref data
        result = fpl_with_fbref.join(
            fbref_df.rename({fbref_id_col: "fbref_id"}),
            on="fbref_id",
            how="left"
        )
        
        # Log coverage
        matched = result.filter(pl.col("fbref_id").is_not_null()).height
        total = result.height
        logger.info(f"FBref merge: {matched}/{total} players matched ({matched/total*100:.1f}%)")
        
        return result
    
    def merge_understat_data(
        self,
        fpl_df: pl.DataFrame,
        understat_df: pl.DataFrame,
        fpl_id_col: str = "fpl_id",
        understat_id_col: str = "player_id",
    ) -> pl.DataFrame:
        """
        Merge Understat data into FPL DataFrame using static mappings.
        
        Args:
            fpl_df: FPL DataFrame with fpl_id column
            understat_df: Understat DataFrame with player_id column
            fpl_id_col: Name of FPL ID column
            understat_id_col: Name of Understat ID column
            
        Returns:
            Merged DataFrame (left join on FPL)
        """
        # Create mapping DataFrame
        mapping_df = pl.DataFrame({
            "fpl_id": list(self.fpl_to_understat.keys()),
            "understat_id": list(self.fpl_to_understat.values()),
        })
        
        # Add Understat ID to FPL data
        fpl_with_understat = fpl_df.join(
            mapping_df,
            left_on=fpl_id_col,
            right_on="fpl_id",
            how="left"
        )
        
        # Merge Understat data
        result = fpl_with_understat.join(
            understat_df.rename({understat_id_col: "understat_id"}),
            on="understat_id",
            how="left"
        )
        
        # Log coverage
        matched = result.filter(pl.col("understat_id").is_not_null()).height
        total = result.height
        logger.info(f"Understat merge: {matched}/{total} players matched ({matched/total*100:.1f}%)")
        
        return result
    
    def validate_coverage(
        self,
        fpl_ids: list[int],
        min_coverage: float = 0.95
    ) -> tuple[bool, list[int]]:
        """
        Validate that enough FPL players have mappings.
        
        Args:
            fpl_ids: List of FPL player IDs to check
            min_coverage: Minimum coverage threshold (0-1)
            
        Returns:
            Tuple of (passes_threshold, missing_ids)
        """
        missing_fbref = [pid for pid in fpl_ids if pid not in self.fpl_to_fbref]
        missing_understat = [pid for pid in fpl_ids if pid not in self.fpl_to_understat]
        
        fbref_coverage = 1 - len(missing_fbref) / len(fpl_ids)
        understat_coverage = 1 - len(missing_understat) / len(fpl_ids)
        
        logger.info(f"FBref coverage: {fbref_coverage:.1%}")
        logger.info(f"Understat coverage: {understat_coverage:.1%}")
        
        missing_both = set(missing_fbref) | set(missing_understat)
        passes = (fbref_coverage >= min_coverage) and (understat_coverage >= min_coverage)
        
        return passes, list(missing_both)
