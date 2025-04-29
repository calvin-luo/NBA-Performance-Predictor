import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

from flask import Flask, jsonify, render_template, request

# ────────────────────────────────────────────────────────────────────────────────
#  Internal modules (kept lean ‑ only what we still use)
# ────────────────────────────────────────────────────────────────────────────────
from data.database import Database
from scrapers.game_scraper import NBAApiScraper  # pulls today’s schedule
from analysis.time_series import GamePredictor, PlayerTimeSeriesAnalyzer
from analysis.player_stats import PlayerStatsCollector

# ────────────────────────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s · %(levelname)s · %(name)s · %(message)s",
)
logger = logging.getLogger("StatLemon.app")

# ────────────────────────────────────────────────────────────────────────────────
#  Flask app & core singletons
# ────────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

db = Database()
# Ensure tables exist at boot so first request doesn’t crash
try:
    db.initialize_database()
except Exception as e:
    logger.error("DB init failed: %s", e)

game_scraper = NBAApiScraper(db)
player_stats = PlayerStatsCollector()
series_analyzer = PlayerTimeSeriesAnalyzer(min_games=10)
game_predictor = GamePredictor(min_games=10)

# Simple in‑memory caches (swap for Redis later)
CACHE: Dict[str, Any] = {}
TTL: Dict[str, int] = {
    "player_stats": 60 * 60,      # 1 h
    "today_games": 10 * 60,       # 10 min
    "lineup_projection": 10 * 60,
}

# ────────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _cache_get(key: str):
    entry = CACHE.get(key)
    if not entry:
        return None
    value, ts = entry
    if datetime.utcnow() - ts > timedelta(seconds=TTL.get(key, 0)):
        CACHE.pop(key, None)
        return None
    return value

def _cache_set(key: str, value):
    CACHE[key] = (value, datetime.utcnow())

# ────────────────────────────────────────────────────────────────────────────────
#  Page routes
# ────────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/compare")
def compare_form():
    return render_template("compare.html")


@app.route("/compare/results")
def compare_results():
    # This placeholder simply echoes query params; real compare logic trimmed for MVP
    players = request.args.getlist("player")
    return render_template("compare_results.html", players=players)


@app.route("/lineup-builder")
def lineup_builder():
    return render_template("lineup_builder.html")

# ────────────────────────────────────────────────────────────────────────────────
#  API routes
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/api/today_games")
def api_today_games():
    """Return today’s games (Time, Away, Home, Venue). Auto‑scrapes and caches."""
    # --- caching first
    cached = _cache_get("today_games")
    if cached is not None:
        return jsonify({"games": cached})

    target_date = datetime.utcnow().strftime("%Y-%m-%d")
    logger.info("Fetching games for %s", target_date)

    try:
        db.initialize_database()  # ensure table exists
        games = db.get_games_by_date(target_date)
    except Exception as db_err:
        logger.error("DB error: %s", db_err)
        games = []

    # If DB empty, try live scrape once
    if not games:
        try:
            scraped = game_scraper.scrape_and_save_games(days_ahead=0)
            logger.info("Scraped & stored %d games", scraped)
            games = db.get_games_by_date(target_date)
        except Exception as scrape_err:
            logger.warning("Scrape failed: %s", scrape_err)
            return jsonify({"games": []}), 503

    # Trim to the 4 required columns
    trimmed: List[Dict[str, str]] = [
        {
            "game_time": g["game_time"],
            "away_team": g["away_team"],
            "home_team": g["home_team"],
            "venue": g.get("venue", "") or "TBD",
        }
        for g in games
    ]

    _cache_set("today_games", trimmed)
    return jsonify({"games": trimmed})


@app.post("/api/lineup_projection")
def api_lineup_projection():
    """Project the next‑game stats for a five‑player lineup using SARIMA."""
    payload = request.get_json(force=True)
    player_names: List[str] = payload.get("playerIds", [])[:5]

    if len(player_names) != 5:
        return jsonify({"error": "Exactly 5 players required"}), 400

    cache_key = "|".join(sorted(player_names))
    cached = _cache_get(f"lineup:{cache_key}")
    if cached is not None:
        return jsonify({"projections": cached})

    projections: Dict[str, Dict[str, float]] = {}

    for name in player_names:
        try:
            stats_df = player_stats.get_player_stats(name)
            if stats_df is None or stats_df.empty:
                continue
            forecast = series_analyzer.forecast_player_performance(name, stats_df)
            projections[name] = {
                metric: round(val["forecast"], 2) for metric, val in forecast.items()
            }
        except Exception as e:
            logger.warning("Projection failed for %s: %s", name, e)
            continue

    _cache_set(f"lineup:{cache_key}", projections)
    return jsonify({"projections": projections})


# ────────────────────────────────────────────────────────────────────────────────
#  Entry‑point helper (so `python app.py` just works)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1 – Use an explicit, absolute DB path
    DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "nba_stats.db"))
    db = Database(DB_PATH)
    game_scraper = NBAApiScraper(db)

    # 2 – Initialise schema and seed today's games
    db.initialize_database()
    game_scraper.scrape_and_save_games(days_ahead=0)

    # 3 – Start Flask (routes must use the *same* db instance or DB_PATH)
    app.config["DB_PATH"] = DB_PATH          # pass it along if you like
    app.run(debug=True, port=5000)
