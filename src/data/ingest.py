"""
Data Ingestion Module for Alpha-FPL.

Handles downloading and preprocessing data from:
1. FPL API / vaastav/Fantasy-Premier-League GitHub repo
2. FBref via soccerdata (defensive metrics)
3. Understat via understatapi (xG/xA context)

IMPORTANT: This module uses static ID mappings (data/meta/player_id_map.json).
No fuzzy matching is performed at ingestion time.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import requests
from loguru import logger

try:
    import soccerdata as sd
except ImportError:
    sd = None
    logger.warning("soccerdata not installed. FBref ingestion will be unavailable.")

try:
    from understatapi import UnderstatClient
except ImportError:
    UnderstatClient = None
    logger.warning("understatapi not installed. Understat ingestion will be unavailable.")

from .mappings import IDMapper


# FPL API endpoints
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP_URL = f"{FPL_BASE_URL}/bootstrap-static/"
FPL_PLAYER_HISTORY_URL = f"{FPL_BASE_URL}/element-summary/{{player_id}}/"
FPL_FIXTURES_URL = f"{FPL_BASE_URL}/fixtures/"

# GitHub raw data URL (vaastav/Fantasy-Premier-League)
VAASTAV_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


class DataIngester:
    """
    Ingests and merges data from FPL, FBref, and Understat sources.
    
    This class handles:
    1. Downloading FPL player/match data from API or GitHub cache
    2. Scraping FBref for defensive metrics (tackles, interceptions, etc.)
    3. Fetching Understat for xG/xA data
    4. Merging all sources using static ID mappings
    
    Attributes:
        raw_dir: Directory for raw downloaded data
        processed_dir: Directory for processed parquet files
        id_mapper: IDMapper instance for cross-source joining
    """
    
    def __init__(
        self,
        raw_dir: str | Path,
        processed_dir: str | Path,
        mappings_path: str | Path,
        seasons: list[str] = None,
    ):
        """
        Initialize the data ingester.
        
        Args:
            raw_dir: Directory to store raw downloads
            processed_dir: Directory for processed data
            mappings_path: Path to player_id_map.json
            seasons: List of seasons to ingest (e.g., ["2021-22", "2022-23"])
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.seasons = seasons or ["2023-24", "2024-25"]
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ID mappings
        try:
            self.id_mapper = IDMapper(mappings_path)
        except FileNotFoundError:
            logger.warning(
                "ID mappings not found. Run tools/generate_map.py first. "
                "Proceeding with FPL-only ingestion."
            )
            self.id_mapper = None
    
    # =========================================================================
    # FPL Data Ingestion
    # =========================================================================
    
    def fetch_fpl_bootstrap(self) -> dict:
        """
        Fetch current FPL bootstrap data (players, teams, positions).
        
        Returns:
            Dict with 'elements', 'teams', 'element_types' keys
        """
        logger.info("Fetching FPL bootstrap data...")
        response = requests.get(FPL_BOOTSTRAP_URL, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def fetch_fpl_player_history(self, player_id: int) -> dict:
        """
        Fetch complete history for a single player.
        
        Args:
            player_id: FPL player ID
            
        Returns:
            Dict with 'history' (gameweek data) and 'fixtures' keys
        """
        url = FPL_PLAYER_HISTORY_URL.format(player_id=player_id)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def ingest_fpl_season(self, season: str) -> pl.DataFrame:
        """
        Ingest FPL data for a specific season from GitHub cache.
        
        Uses vaastav/Fantasy-Premier-League repository for historical data.
        
        Args:
            season: Season string (e.g., "2023-24")
            
        Returns:
            DataFrame with player-gameweek level data
        """
        logger.info(f"Ingesting FPL data for season {season}...")
        
        # Convert season format for GitHub
        year = int("20" + season.split("-")[0])
        github_season = f"{year}-{year % 100 + 1:02d}"
        
        # Try to load from raw cache first
        cache_path = self.raw_dir / f"fpl_{season}.parquet"
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return pl.read_parquet(cache_path)
        
        # Construct URLs for gameweek files
        dfs = []
        for gw in range(1, 39):
            gw_url = f"{VAASTAV_BASE_URL}/{github_season}/gws/gw{gw}.csv"
            try:
                gw_df = pl.read_csv(gw_url)
                gw_df = gw_df.with_columns(
                    pl.lit(gw).alias("gameweek"),
                    pl.lit(season).alias("season"),
                )
                dfs.append(gw_df)
                logger.debug(f"Loaded GW{gw}: {len(gw_df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load GW{gw} for {season}: {e}")
                continue
        
        if not dfs:
            raise ValueError(f"No gameweek data found for season {season}")
        
        df = pl.concat(dfs)
        
        # Standardize column names
        df = self._standardize_fpl_columns(df)
        
        # Cache to parquet
        df.write_parquet(cache_path)
        logger.info(f"Cached {len(df)} rows to {cache_path}")
        
        return df
    
    def _standardize_fpl_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize FPL column names across seasons."""
        # Column name mappings (handle variations across seasons)
        renames = {
            "element": "fpl_id",
            "name": "player_name", 
            "position": "position_name",
            "team": "team_name",
            "total_points": "points",
            "clean_sheets": "clean_sheet",
            "goals_conceded": "goals_against",
            "yellow_cards": "yellow_card",
            "red_cards": "red_card",
            "saves": "saves",
            "bonus": "bonus",
            "bps": "bps",
            "influence": "influence",
            "creativity": "creativity",
            "threat": "threat",
            "ict_index": "ict_index",
            "value": "price",
            "transfers_balance": "transfers_net",
            "selected": "selected_by",
            "transfers_in": "transfers_in",
            "transfers_out": "transfers_out",
        }
        
        for old, new in renames.items():
            if old in df.columns and old != new:
                df = df.rename({old: new})
        
        return df
    
    def ingest_fpl_all_seasons(self) -> pl.DataFrame:
        """
        Ingest FPL data for all configured seasons.
        
        Returns:
            Combined DataFrame with all seasons
        """
        dfs = []
        for season in self.seasons:
            try:
                df = self.ingest_fpl_season(season)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to ingest season {season}: {e}")
        
        if not dfs:
            raise ValueError("No FPL data ingested for any season")
        
        combined = pl.concat(dfs, how="align")
        logger.info(f"Combined FPL data: {len(combined)} rows across {len(dfs)} seasons")
        return combined
    
    # =========================================================================
    # FBref Data Ingestion  
    # =========================================================================
    
    def ingest_fbref_season(self, season: str) -> pl.DataFrame:
        """
        Ingest FBref defensive metrics for a season.
        
        Fetches tackles, interceptions, blocks, clearances, etc.
        
        Args:
            season: Season string (e.g., "2023-24")
            
        Returns:
            DataFrame with player-match defensive metrics
        """
        if sd is None:
            raise ImportError("soccerdata required for FBref ingestion")
        
        logger.info(f"Ingesting FBref defensive data for season {season}...")
        
        cache_path = self.raw_dir / f"fbref_{season}.parquet"
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return pl.read_parquet(cache_path)
        
        # Use soccerdata to fetch FBref
        # Convert season format
        year = int("20" + season.split("-")[0])
        sd_season = f"{year}"
        
        try:
            fbref = sd.FBref(leagues="ENG-Premier League", seasons=sd_season)
            
            # Fetch player stats
            player_stats = fbref.read_player_season_stats(stat_type="defense")
            
            # Convert to polars
            df = pl.from_pandas(player_stats.reset_index())
            
            # Add season column
            df = df.with_columns(pl.lit(season).alias("season"))
            
            # Cache
            df.write_parquet(cache_path)
            logger.info(f"Cached {len(df)} FBref rows to {cache_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"FBref ingestion failed: {e}")
            raise
    
    def ingest_fbref_all_seasons(self) -> pl.DataFrame:
        """Ingest FBref data for all seasons."""
        dfs = []
        for season in self.seasons:
            try:
                df = self.ingest_fbref_season(season)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to ingest FBref {season}: {e}")
        
        if not dfs:
            logger.warning("No FBref data ingested")
            return pl.DataFrame()
        
        return pl.concat(dfs, how="align")
    
    # =========================================================================
    # Understat Data Ingestion
    # =========================================================================
    
    def ingest_understat_season(self, season: str) -> pl.DataFrame:
        """
        Ingest Understat xG/xA data for a season.
        
        Args:
            season: Season string (e.g., "2023-24")
            
        Returns:
            DataFrame with player xG, xA, shots data
        """
        if UnderstatClient is None:
            raise ImportError("understatapi required for Understat ingestion")
        
        logger.info(f"Ingesting Understat data for season {season}...")
        
        cache_path = self.raw_dir / f"understat_{season}.parquet"
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return pl.read_parquet(cache_path)
        
        # Convert season format
        year = int("20" + season.split("-")[0])
        understat_season = str(year)
        
        try:
            client = UnderstatClient()
            
            # Get all EPL player stats
            league_data = client.league(league="EPL").get_player_data(season=understat_season)
            
            # Convert to DataFrame
            df = pl.DataFrame(league_data)
            df = df.with_columns(pl.lit(season).alias("season"))
            
            # Cache
            df.write_parquet(cache_path)
            logger.info(f"Cached {len(df)} Understat rows to {cache_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Understat ingestion failed: {e}")
            raise
    
    def ingest_understat_all_seasons(self) -> pl.DataFrame:
        """Ingest Understat data for all seasons."""
        dfs = []
        for season in self.seasons:
            try:
                df = self.ingest_understat_season(season)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to ingest Understat {season}: {e}")
        
        if not dfs:
            logger.warning("No Understat data ingested")
            return pl.DataFrame()
        
        return pl.concat(dfs, how="align")
    
    # =========================================================================
    # Merged Dataset Creation
    # =========================================================================
    
    def create_merged_dataset(self) -> pl.DataFrame:
        """
        Create the complete merged dataset with all sources.
        
        Joins FPL + FBref + Understat using static ID mappings.
        
        Returns:
            Merged DataFrame with all features
        """
        logger.info("Creating merged dataset...")
        
        # Ingest all sources
        fpl_df = self.ingest_fpl_all_seasons()
        
        # Try FBref (optional)
        try:
            fbref_df = self.ingest_fbref_all_seasons()
        except Exception as e:
            logger.warning(f"FBref ingestion failed, proceeding without: {e}")
            fbref_df = None
        
        # Try Understat (optional)
        try:
            understat_df = self.ingest_understat_all_seasons()
        except Exception as e:
            logger.warning(f"Understat ingestion failed, proceeding without: {e}")
            understat_df = None
        
        # Merge if we have ID mapper and additional data
        merged = fpl_df
        
        if self.id_mapper is not None:
            if fbref_df is not None and len(fbref_df) > 0:
                merged = self.id_mapper.merge_fbref_data(merged, fbref_df)
            
            if understat_df is not None and len(understat_df) > 0:
                merged = self.id_mapper.merge_understat_data(merged, understat_df)
        
        # Save merged dataset
        output_path = self.processed_dir / "merged_raw.parquet"
        merged.write_parquet(output_path)
        logger.info(f"Saved merged dataset: {output_path} ({len(merged)} rows)")
        
        return merged
    
    def get_current_gameweek_data(self) -> pl.DataFrame:
        """
        Fetch current (live) gameweek data from FPL API.
        
        Used for real-time predictions during an active season.
        
        Returns:
            DataFrame with current player status/prices
        """
        bootstrap = self.fetch_fpl_bootstrap()
        
        # Extract player data
        elements = bootstrap["elements"]
        df = pl.DataFrame(elements)
        
        # Add team names
        teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
        df = df.with_columns(
            pl.col("team").replace_strict(teams).alias("team_name")
        )
        
        # Add position names
        positions = {p["id"]: p["singular_name_short"] for p in bootstrap["element_types"]}
        df = df.with_columns(
            pl.col("element_type").replace_strict(positions).alias("position")
        )
        
        return df
