"""
ID Mapping Generator Tool.

This script generates static mappings between FPL IDs, FBref IDs, and Understat IDs.
It uses fuzzy matching to find corresponding players across data sources.

IMPORTANT: Run this ONCE per season, then manually verify the ~10-20 ambiguous matches.
The output file should be committed to version control.

Usage:
    python tools/generate_map.py --season 2024-25
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import requests
from loguru import logger
from thefuzz import fuzz, process


# FPL API
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def fetch_fpl_players() -> pl.DataFrame:
    """Fetch current FPL player data."""
    logger.info("Fetching FPL players...")
    
    response = requests.get(FPL_BOOTSTRAP_URL, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # Extract players
    players = data["elements"]
    teams = {t["id"]: t["name"] for t in data["teams"]}
    positions = {p["id"]: p["singular_name_short"] for p in data["element_types"]}
    
    df = pl.DataFrame(players)
    df = df.with_columns([
        pl.col("team").replace_strict(teams).alias("team_name"),
        pl.col("element_type").replace_strict(positions).alias("position"),
    ])
    
    # Normalize names for matching
    df = df.with_columns([
        (pl.col("first_name") + " " + pl.col("second_name")).alias("full_name"),
        pl.col("web_name").alias("display_name"),
    ])
    
    logger.info(f"Fetched {len(df)} FPL players")
    return df


def fetch_fbref_players(season: str) -> pl.DataFrame:
    """Fetch FBref player list."""
    logger.info(f"Fetching FBref players for {season}...")
    
    try:
        import soccerdata as sd
        
        year = int("20" + season.split("-")[0])
        fbref = sd.FBref(leagues="ENG-Premier League", seasons=str(year))
        
        # Get player stats (any stat type to get player list)
        stats = fbref.read_player_season_stats(stat_type="standard")
        df = pl.from_pandas(stats.reset_index())
        
        # Extract player name from index
        if "player" in df.columns:
            df = df.with_columns(pl.col("player").alias("fbref_name"))
        
        logger.info(f"Fetched {len(df)} FBref players")
        return df
        
    except Exception as e:
        logger.error(f"FBref fetch failed: {e}")
        return pl.DataFrame()


def fetch_understat_players(season: str) -> pl.DataFrame:
    """Fetch Understat player list."""
    logger.info(f"Fetching Understat players for {season}...")
    
    try:
        from understatapi import UnderstatClient
        
        year = int("20" + season.split("-")[0])
        client = UnderstatClient()
        
        players = client.league(league="EPL").get_player_data(season=str(year))
        df = pl.DataFrame(players)
        
        df = df.with_columns(pl.col("player_name").alias("understat_name"))
        
        logger.info(f"Fetched {len(df)} Understat players")
        return df
        
    except Exception as e:
        logger.error(f"Understat fetch failed: {e}")
        return pl.DataFrame()


def fuzzy_match_player(
    name: str,
    candidates: list[tuple],
    threshold: int = 80,
) -> Optional[tuple]:
    """
    Find best fuzzy match for a player name.
    
    Args:
        name: Player name to match
        candidates: List of (name, id) tuples
        threshold: Minimum match score (0-100)
        
    Returns:
        (matched_name, id, score) or None
    """
    if not candidates:
        return None
    
    # Create name -> id mapping
    name_to_id = {c[0]: c[1] for c in candidates}
    names = list(name_to_id.keys())
    
    # Find best match
    result = process.extractOne(name, names, scorer=fuzz.token_sort_ratio)
    
    if result is None:
        return None
    
    matched_name, score = result[0], result[1]
    
    if score >= threshold:
        return (matched_name, name_to_id[matched_name], score)
    
    return None


def generate_mappings(
    fpl_df: pl.DataFrame,
    fbref_df: pl.DataFrame,
    understat_df: pl.DataFrame,
    threshold: int = 80,
) -> dict:
    """
    Generate ID mappings using fuzzy matching.
    
    Returns:
        Dictionary with mappings and confidence scores
    """
    logger.info("Generating mappings...")
    
    fpl_to_fbref = {}
    fpl_to_understat = {}
    ambiguous = []  # Needs manual review
    unmatched = []
    
    # Prepare candidates
    if len(fbref_df) > 0 and "fbref_name" in fbref_df.columns:
        fbref_candidates = [
            (row["fbref_name"], row.get("player_id", row.get("fbref_name")))
            for row in fbref_df.iter_rows(named=True)
        ]
    else:
        fbref_candidates = []
    
    if len(understat_df) > 0 and "understat_name" in understat_df.columns:
        understat_candidates = [
            (row["understat_name"], row.get("id", row["understat_name"]))
            for row in understat_df.iter_rows(named=True)
        ]
    else:
        understat_candidates = []
    
    # Match each FPL player
    for row in fpl_df.iter_rows(named=True):
        fpl_id = row["id"]
        fpl_name = row["full_name"]
        
        # Match to FBref
        if fbref_candidates:
            fbref_match = fuzzy_match_player(fpl_name, fbref_candidates, threshold)
            if fbref_match:
                matched_name, fbref_id, score = fbref_match
                fpl_to_fbref[fpl_id] = fbref_id
                
                if score < 90:  # Flag for review
                    ambiguous.append({
                        "fpl_id": fpl_id,
                        "fpl_name": fpl_name,
                        "source": "fbref",
                        "matched_name": matched_name,
                        "matched_id": fbref_id,
                        "score": score,
                    })
            else:
                unmatched.append({"fpl_id": fpl_id, "fpl_name": fpl_name, "source": "fbref"})
        
        # Match to Understat
        if understat_candidates:
            understat_match = fuzzy_match_player(fpl_name, understat_candidates, threshold)
            if understat_match:
                matched_name, understat_id, score = understat_match
                fpl_to_understat[fpl_id] = understat_id
                
                if score < 90:
                    ambiguous.append({
                        "fpl_id": fpl_id,
                        "fpl_name": fpl_name,
                        "source": "understat",
                        "matched_name": matched_name,
                        "matched_id": understat_id,
                        "score": score,
                    })
            else:
                unmatched.append({"fpl_id": fpl_id, "fpl_name": fpl_name, "source": "understat"})
    
    logger.info(f"Matched {len(fpl_to_fbref)} to FBref")
    logger.info(f"Matched {len(fpl_to_understat)} to Understat")
    logger.info(f"{len(ambiguous)} matches need review")
    logger.info(f"{len(unmatched)} unmatched")
    
    return {
        "fpl_to_fbref": fpl_to_fbref,
        "fpl_to_understat": fpl_to_understat,
        "ambiguous": ambiguous,
        "unmatched": unmatched,
    }


def save_mappings(
    mappings: dict,
    output_path: Path,
    season: str,
):
    """Save mappings to JSON file."""
    output_data = {
        "fpl_to_fbref": {str(k): v for k, v in mappings["fpl_to_fbref"].items()},
        "fpl_to_understat": {str(k): v for k, v in mappings["fpl_to_understat"].items()},
        "metadata": {
            "season": season,
            "generated_at": datetime.now().isoformat(),
            "n_fbref_mappings": len(mappings["fpl_to_fbref"]),
            "n_understat_mappings": len(mappings["fpl_to_understat"]),
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved mappings to {output_path}")
    
    # Save review file separately
    if mappings["ambiguous"]:
        review_path = output_path.parent / "needs_review.json"
        with open(review_path, "w") as f:
            json.dump(mappings["ambiguous"], f, indent=2)
        logger.warning(f"Review needed: {review_path}")
    
    if mappings["unmatched"]:
        unmatched_path = output_path.parent / "unmatched.json"
        with open(unmatched_path, "w") as f:
            json.dump(mappings["unmatched"], f, indent=2)
        logger.warning(f"Unmatched players: {unmatched_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate ID mappings between FPL, FBref, and Understat"
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2024-25",
        help="Season to generate mappings for (e.g., 2024-25)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/meta/player_id_map.json",
        help="Output file path",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=80,
        help="Fuzzy match threshold (0-100)",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Generating ID mappings for season {args.season}")
    
    # Fetch player lists from all sources
    fpl_df = fetch_fpl_players()
    fbref_df = fetch_fbref_players(args.season)
    understat_df = fetch_understat_players(args.season)
    
    # Generate mappings
    mappings = generate_mappings(
        fpl_df, fbref_df, understat_df,
        threshold=args.threshold,
    )
    
    # Save
    save_mappings(mappings, Path(args.output), args.season)
    
    logger.info("Done!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"FPL → FBref:     {len(mappings['fpl_to_fbref'])} mapped")
    print(f"FPL → Understat: {len(mappings['fpl_to_understat'])} mapped")
    print(f"Needs review:    {len(mappings['ambiguous'])}")
    print(f"Unmatched:       {len(mappings['unmatched'])}")
    print("=" * 60)
    
    if mappings["ambiguous"]:
        print("\n⚠️  Please review 'data/meta/needs_review.json'")
        print("    Manually verify low-confidence matches before running pipeline.")


if __name__ == "__main__":
    main()
