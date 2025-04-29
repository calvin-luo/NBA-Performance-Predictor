import logging
import os
from datetime import datetime, timedelta

from flask import Flask, jsonify, redirect, render_template, request, url_for

# ────────────────────────────────────────────────────────────────────────────────
#  Internal modules
#  (These existed in the original repo – we keep the ones that are still used.)
# ────────────────────────────────────────────────────────────────────────────────
from data.database import Database
from scrapers.game_scraper import NBAApiScraper
from scrapers.player_scraper import RotowireScraper  # still used for injuries / starters
from analysis.player_stats import PlayerStatsCollector
from analysis.time_series import GamePredictor, PlayerTimeSeriesAnalyzer

# ────────────────────────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("StatLemon.app")

# ────────────────────────────────────────────────────────────────────────────────
#  Flask App
# ────────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ────────────────────────────────────────────────────────────────────────────────
#  Core services (DB, scrapers, analytics) – singletons for the life of the app
# ────────────────────────────────────────────────────────────────────────────────
db: Database = Database()

game_scraper = NBAApiScraper(db)
rotowire_scraper = RotowireScraper(db)

player_stats_collector = PlayerStatsCollector()
series_analyzer = PlayerTimeSeriesAnalyzer(min_games=10)
game_predictor = GamePredictor(min_games=10)

# ────────────────────────────────────────────────────────────────────────────────
#  Simple in‑memory caches to avoid hammering external APIs on every request.
#  (In production you’d swap this for Redis or similar.)
# ────────────────────────────────────────────────────────────────────────────────
PLAYER_STATS_CACHE: dict[str, dict] = {}
PLAYER_STATS_TTL = 60 * 60  # 1 hour

LINEUP_PROJECTION_CACHE: dict[str, dict] = {}
LINEUP_CACHE_TTL = 10 * 60  # 10 minutes

TODAY_GAMES_CACHE: dict[str, dict] = {}
TODAY_GAMES_TTL = 10 * 60  # refresh scoreboard every 10 mins


# ────────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────────────────────

def _cache_get(cache: dict, key: str, ttl: int):
    """Return cached item if it exists and isn’t stale."""
    entry = cache.get(key)
    if not entry:
        return None
    value, timestamp = entry
    if datetime.utcnow() - timestamp > timedelta(seconds=ttl):
        cache.pop(key, None)
        return None
    return value


def _cache_set(cache: dict, key: str, value, ttl: int):
    cache[key] = (value, datetime.utcnow())


# ────────────────────────────────────────────────────────────────────────────────
#  Page Routes (HTML)
# ────────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    # Placeholder page – content trimmed in Step 1
    return render_template("about.html")


@app.route("/compare")
def compare_form():
    return render_template("compare.html")


@app.post("/compare_results")
def compare_results():
    """Handle the compare form; minimal logic until Step 2 refactor."""
    player1 = request.form["player1"].strip()
    player2 = request.form["player2"].strip()
    # TODO: call analytics layer – for now just echo
    return render_template(
        "compare_results.html",
        player1=player1,
        player2=player2,
    )


@app.route("/lineup_builder")
def lineup_builder():
    return render_template("lineup_builder.html")

# ❌  The “/schedule” route and template were removed in Step 1 – no longer here.

# ────────────────────────────────────────────────────────────────────────────────
#  API Routes (JSON)
# ────────────────────────────────────────────────────────────────────────────────

@app.get("/api/today_games")
def api_today_games():
    """Return today’s NBA games with minimal info for the home‑page table."""
    today = datetime.utcnow().date().isoformat()
    cached = _cache_get(TODAY_GAMES_CACHE, today, TODAY_GAMES_TTL)
    if cached:
        return jsonify(cached)

    games = game_scraper.get_games_for_date(today)
    # Map / trim fields
    payload = {
        "date": today,
        "games": [
            {
                "game_id": g["game_id"],
                "game_time": g.get("game_time", "TBD"),
                "away_team": g["away_team"],
                "home_team": g["home_team"],
                "venue": g.get("venue", "TBD"),
            }
            for g in games
        ],
    }
    _cache_set(TODAY_GAMES_CACHE, today, payload, TODAY_GAMES_TTL)
    return jsonify(payload)


@app.post("/api/lineup_projection")
def api_lineup_projection():
    """Return SARIMA projections for the next game of the supplied players."""
    data = request.get_json(force=True)
    player_ids = data.get("playerIds")
    if not player_ids or len(player_ids) != 5:
        return jsonify({"error": "Exactly five playerIds required."}), 400

    cache_key = "-".join(sorted(player_ids))
    cached = _cache_get(LINEUP_PROJECTION_CACHE, cache_key, LINEUP_CACHE_TTL)
    if cached:
        return jsonify(cached)

    projections: dict[str, dict] = {}
    for pid in player_ids:
        series = series_analyzer.to_series(pid)
        projections[pid] = game_predictor.sarima_next_game(series)

    _cache_set(LINEUP_PROJECTION_CACHE, cache_key, projections, LINEUP_CACHE_TTL)
    return jsonify({"projections": projections})


# ────────────────────────────────────────────────────────────────────────────────
#  Error Handlers
# ────────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
@app.errorhandler(500)
def handle_error(err):
    message = getattr(err, "description", str(err))
    return render_template("error.html", message=message), err.code if hasattr(err, "code") else 500


# ────────────────────────────────────────────────────────────────────────────────
#  Entrypoint
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=bool(os.environ.get("FLASK_DEBUG")))
