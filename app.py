import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s · %(levelname)s · %(name)s · %(message)s")
logger = logging.getLogger(__name__)

import os
from datetime import datetime, timedelta
from typing import Dict, Any
from nba_api.stats.library.http import NBAStatsHTTP


from flask import Flask, jsonify, render_template, request
from data.database import Database          # your DB wrapper
from scrapers.game_scraper import NBAApiScraper
from analysis.player_stats import PlayerStatsCollector
from analysis.time_series import PlayerTimeSeriesAnalyzer, KEY_METRICS, CATEGORIES
# ── safe Cloudflare-bypass headers ──────────────────────────────────────────
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

# Detect or create headers dict attribute
if hasattr(NBAStatsHTTP, "_HEADERS"):
    target_headers = NBAStatsHTTP._HEADERS
elif hasattr(NBAStatsHTTP, "HEADERS"):
    target_headers = NBAStatsHTTP.HEADERS
else:
    target_headers = {}
    NBAStatsHTTP._HEADERS = target_headers

target_headers.update(_BROWSER_HEADERS)
app = Flask(__name__)

db = Database()
# Ensure tables exist at boot so first request doesn't crash
try:
    db.initialize_database()
except Exception as e:
    logger.error("DB init failed: %s", e)

game_scraper = NBAApiScraper(db)
player_stats = PlayerStatsCollector()
series_analyzer = PlayerTimeSeriesAnalyzer(min_games=10)

# Simple in‑memory caches
CACHE: Dict[str, Any] = {}
TTL: Dict[str, int] = {
    "player_stats": 60 * 60,      # 1 h
    "today_games": 10 * 60,       # 10 min
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
    # This function displays results of player comparison
    # Get individual player names from form
    player1 = request.args.get("player1")
    player2 = request.args.get("player2")
    return render_template("compare_results.html", player1=player1, player2=player2)


@app.route("/lineup-builder")
def lineup_builder():
    return render_template("lineup_builder.html")


@app.route("/player/<player_name>")
def player_detail(player_name):
    """Player detail page that shows stats and predictions."""
    return render_template("player.html", player_name=player_name)

# ────────────────────────────────────────────────────────────────────────────────
#  API routes
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/api/today_games")
def api_today_games():
    """Return today's games (Time, Away, Home, Venue). Auto‑scrapes and caches."""
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
    trimmed = [
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


@app.route("/api/search_player", methods=["GET"])
def api_search_player():
    """Search for players by name - simple placeholder for MVP."""
    query = request.args.get("q", "").lower()
    if not query or len(query) < 2:
        return jsonify([])
    
    # Simplified list of common NBA players
    players = [
        "LeBron James", "Stephen Curry", "Kevin Durant", 
        "Giannis Antetokounmpo", "Luka Dončić", "Nikola Jokić",
        "Joel Embiid", "Kawhi Leonard", "Jayson Tatum", "Damian Lillard"
    ]
    
    # Simple filtering
    results = [p for p in players if query in p.lower()]
    return jsonify(results[:10])  # Limit to 10 results


@app.route("/api/player_stats/<player_name>")
def api_player_stats(player_name):
    """Get player stats for analysis and visualization."""
    refresh = request.args.get("refresh", "false").lower() == "true"
    
    # Check cache
    cache_key = f"player_stats_{player_name}"
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return jsonify({"stats": cached})
    
    # Get player stats from collector
    stats_df = player_stats.get_player_stats(player_name)
    
    if stats_df is None or stats_df.empty:
        return jsonify({"error": f"No stats found for {player_name}"}), 404
    
    # Convert to dict for JSON serialization
    stats = stats_df.to_dict('records')
    
    # Cache results
    _cache_set(cache_key, stats)
    
    return jsonify({"stats": stats})

@app.route("/api/player_info/<player_name>")
def api_player_info(player_name):
    """Get player information including team and position."""
    # Get player info from collector
    player_info = player_stats.get_player_info(player_name)
    
    if player_info is None:
        return jsonify({"error": f"No information found for {player_name}"}), 404
    
    # Cast any numpy scalar types to native Python before serializing
    from numpy import generic, integer, floating
    player_info_native = {k: (v.item() if isinstance(v, (integer, floating, generic)) else v)
                          for k, v in player_info.items()}
    return jsonify({"info": player_info_native}), 200

@app.route("/api/metric_categories")
def api_metric_categories():
    """Get the metric categories structure."""
    return jsonify({
        "categories": CATEGORIES,
        "key_metrics": KEY_METRICS
    })

@app.route("/api/player_prediction/<player_name>")
def api_player_prediction(player_name):
    """Get player predictions based on time series analysis."""
    # Get player stats
    stats_df = player_stats.get_player_stats(player_name)
    
    if stats_df is None or stats_df.empty:
        return jsonify({"error": f"No stats found for {player_name}"}), 404
    
    # Generate prediction
    forecast = series_analyzer.forecast_player_performance(player_name, stats_df)
    
    return jsonify({"prediction": forecast})


@app.post("/api/lineup_projection")
def api_lineup_projection():
    """Project the next‑game stats for a five‑player lineup using SARIMA."""
    payload = request.get_json(force=True)
    player_names = payload.get("players", [])[:5]

    if len(player_names) < 3:
        return jsonify({"error": "At least 3 players required"}), 400

    cache_key = "|".join(sorted(player_names))
    cached = _cache_get(f"lineup:{cache_key}")
    if cached is not None:
        return jsonify({"projections": cached})

    projections = {}

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
#  Error handlers
# ────────────────────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Internal server error"), 500


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

    # 3 – Start Flask
    app.config["DB_PATH"] = DB_PATH
    app.run(debug=True, port=5000)