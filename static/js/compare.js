// Variables to store player data
let player1Stats = null;
let player2Stats = null;
let player1Prediction = null;
let player2Prediction = null;
let player1Info = null;
let player2Info = null;
let currentMetric = 'GAME_SCORE';

// Chart objects
let radarChart = null;
let fantasyChart = null;
let volumeChart = null;
let efficiencyChart = null;
let recentGamesChart = null;

const params = new URLSearchParams(window.location.search);
const p1Raw = params.get('player1') ?? '';
const p2Raw = params.get('player2') ?? '';
const p1 = encodeURIComponent(p1Raw);
const p2 = encodeURIComponent(p2Raw);

$(document).ready(function() {
    // Load player data
    loadPlayerData();
    
    // Handle metric selection for recent games chart
    $('[data-metric]').click(function() {
        $('[data-metric]').removeClass('active');
        $(this).addClass('active');
        currentMetric = $(this).data('metric');
        updateRecentGamesChart();
    });
});

// Load player data
function loadPlayerData() {
    // 1️⃣ kick off all network calls in parallel
    const info1 = $.getJSON(`/api/player_info/${p1}`);
    const info2 = $.getJSON(`/api/player_info/${p2}`);
    const stats1 = $.getJSON(`/api/player_stats/${p1}`);
    const stats2 = $.getJSON(`/api/player_stats/${p2}`);

    // 2️⃣ wait for *all* of them, even if one 404s
    Promise.allSettled([info1, info2, stats1, stats2]).then(
        ([i1, i2, s1, s2]) => {
            /* — team / position panels — */
            if (i1.status === 'fulfilled')
                updatePlayerTeamPosition(i1.value.info, 'player1-team-position');
            else $('#player1-team-position').text('—');

            if (i2.status === 'fulfilled')
                updatePlayerTeamPosition(i2.value.info, 'player2-team-position');
            else $('#player2-team-position').text('—');

            /* — stats & charts — */
            if (s1.status === 'fulfilled') player1Stats = s1.value.stats;
            if (s2.status === 'fulfilled') player2Stats = s2.value.stats;

            if (player1Stats && player2Stats) {
                updateKeyStatsTable();
                createRadarChart();
                createCategoryCharts();
                createRecentGamesChart();
                loadPredictions();   // this uses the encoded names too
            } else {
                $('#key-stats-table').html(
                    '<tr><td colspan="4" class="text-center text-danger">'+
                    'Stats unavailable for one or both players.</td></tr>');
            }
        });
}


// Fetch player information including team and position
function fetchPlayerInfo(playerName, playerNum) {
    const endpoint = `/api/player_info/${playerName}`;
    
    $.getJSON(endpoint, function(data) {
        if (playerNum === '1') {
            player1Info = data.info;
            updatePlayerTeamPosition(player1Info, 'player1-team-position');
        } else {
            player2Info = data.info;
            updatePlayerTeamPosition(player2Info, 'player2-team-position');
        }
    }).fail(function() {
        // Set default values in case of error
        $(`#player${playerNum}-team-position`).text('—');
    });
}

// Update player team and position display
function updatePlayerTeamPosition(playerInfo, elementId) {
    if (playerInfo) {
        // Format: "Team Name | Position"
        const teamName = playerInfo.team_name || '—';
        const position = playerInfo.position || '';
        
        $(`#${elementId}`).text(`${teamName} | ${position}`);
    } else {
        $(`#${elementId}`).text('—');
    }
}

// Load player predictions
function loadPredictions() {
    const pred1 = $.getJSON(`/api/player_prediction/${p1}`);
    const pred2 = $.getJSON(`/api/player_prediction/${p2}`);
    
    Promise.allSettled([pred1, pred2])
        .then(([p1Result, p2Result]) => {
            if (p1Result.status === 'fulfilled') 
                player1Prediction = p1Result.value.prediction;
            if (p2Result.status === 'fulfilled') 
                player2Prediction = p2Result.value.prediction;
            
            // Update the prediction table if we have at least one prediction
            if (player1Prediction || player2Prediction) {
                updatePredictionTable();
            } else {
                $('#prediction-table').html(
                    '<tr><td colspan="4" class="text-center text-danger">'+
                    'Could not load predictions for either player.</td></tr>');
            }
        });
}

// Calculate average for a specific metric
function calculateAverage(playerStats, metric) {
    if (!playerStats || playerStats.length === 0) return 0;
    
    const values = playerStats.map(game => parseFloat(game[metric])).filter(val => !isNaN(val));
    return values.reduce((sum, val) => sum + val, 0) / values.length;
}

// Update key stats table
function updateKeyStatsTable() {
    if (!player1Stats || !player2Stats) {
        return;
    }
    
    // Define metrics to display
    const metrics = [
        { key: 'PTS_PER_MIN', label: 'Points Per 36 Min', formatter: (val) => (val * 36).toFixed(1) },
        { key: 'FG_PCT', label: 'Field Goal %', formatter: (val) => (val * 100).toFixed(1) + '%' },
        { key: 'TS_PCT', label: 'True Shooting %', formatter: (val) => (val * 100).toFixed(1) + '%' },
        { key: 'PLUS_MINUS', label: 'Plus/Minus', formatter: (val) => val.toFixed(1) },
        { key: 'GAME_SCORE', label: 'Game Score', formatter: (val) => val.toFixed(1) },
        { key: 'AST_TO_RATIO', label: 'Assist to TO Ratio', formatter: (val) => val.toFixed(1) }
    ];
    
    let tableHtml = '';
    
    metrics.forEach(metric => {
        const player1Avg = calculateAverage(player1Stats, metric.key);
        const player2Avg = calculateAverage(player2Stats, metric.key);
        const diff = player1Avg - player2Avg;
        
        // Determine which player is better for this metric
        let player1Class = '';
        let player2Class = '';
        let comparisonHtml = '';
        
        if (diff > 0) {
            player1Class = 'text-success fw-bold';
            comparisonHtml = `<i class="fas fa-angle-double-left text-success"></i>`;
        } else if (diff < 0) {
            player2Class = 'text-success fw-bold';
            comparisonHtml = `<i class="fas fa-angle-double-right text-success"></i>`;
        } else {
            comparisonHtml = `<i class="fas fa-equals text-muted"></i>`;
        }
        
        tableHtml += `
            <tr>
                <td>${metric.label}</td>
                <td class="text-center ${player1Class}">${metric.formatter(player1Avg)}</td>
                <td class="text-center">${comparisonHtml}</td>
                <td class="text-center ${player2Class}">${metric.formatter(player2Avg)}</td>
            </tr>
        `;
    });
    
    $('#key-stats-table').html(tableHtml);
}

// Create radar chart
function createRadarChart() {
    if (!player1Stats || !player2Stats) {
        return;
    }
    
    const ctx = document.getElementById('radarChart').getContext('2d');
    
    // Calculate normalized values for radar chart (0 to 100 scale)
    const player1Data = [
        calculateAverage(player1Stats, 'PTS_PER_MIN') * 30, // Scale points per min
        calculateAverage(player1Stats, 'TS_PCT') * 100,     // True shooting percentage
        (calculateAverage(player1Stats, 'PLUS_MINUS') + 10) * 5, // Normalize plus/minus
        calculateAverage(player1Stats, 'GAME_SCORE') * 4,   // Game score
        Math.min(calculateAverage(player1Stats, 'AST_TO_RATIO') * 10, 100) // Assist to turnover ratio
    ];
    
    const player2Data = [
        calculateAverage(player2Stats, 'PTS_PER_MIN') * 30,
        calculateAverage(player2Stats, 'TS_PCT') * 100,
        (calculateAverage(player2Stats, 'PLUS_MINUS') + 10) * 5,
        calculateAverage(player2Stats, 'GAME_SCORE') * 4,
        Math.min(calculateAverage(player2Stats, 'AST_TO_RATIO') * 10, 100)
    ];
    
    if (radarChart) {
        radarChart.destroy();
    }
    
    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Scoring', 'Efficiency', 'Impact', 'Game Score', 'Playmaking'],
            datasets: [
                {
                    label: p1Raw,
                    data: player1Data,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgb(54, 162, 235)',
                    pointBackgroundColor: 'rgb(54, 162, 235)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgb(54, 162, 235)'
                },
                {
                    label: p2Raw,
                    data: player2Data,
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgb(255, 99, 132)',
                    pointBackgroundColor: 'rgb(255, 99, 132)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgb(255, 99, 132)'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Player Performance Comparison'
                }
            },
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            }
        }
    });
}

// Create category charts
function createCategoryCharts() {
    if (!player1Stats || !player2Stats) {
        return;
    }
    
    // Create charts for each category
    createCategoryChart('fantasy', 'fantasyChart');
    createCategoryChart('volume', 'volumeChart');
    createCategoryChart('efficiency', 'efficiencyChart');
}

// Create a chart for a specific category
function createCategoryChart(category, canvasId) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Get metrics for this category
    const metrics = CATEGORIES[category];
    
    // Calculate averages for each player
    const player1Avgs = metrics.map(metric => calculateAverage(player1Stats, metric));
    const player2Avgs = metrics.map(metric => calculateAverage(player2Stats, metric));
    
    // Format labels for display
    const labels = metrics.map(getMetricLabel);
    
    // Destroy existing chart if exists
    if (window[canvasId + 'Chart']) {
        window[canvasId + 'Chart'].destroy();
    }
    
    // Create new chart
    window[canvasId + 'Chart'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: p1Raw,
                    data: player1Avgs,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgb(54, 162, 235)',
                    borderWidth: 1
                },
                {
                    label: p2Raw,
                    data: player2Avgs,
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgb(255, 99, 132)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: category.charAt(0).toUpperCase() + category.slice(1) + ' Metrics'
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

function updatePredictionTable() {
    if (!player1Prediction || !player2Prediction) {
        return;
    }
    
    // Define metrics to display
    const metrics = [
        // Core metrics
        { key: 'PTS_PER_MIN', label: 'Points Per 36 Min', formatter: (val) => (val * 36).toFixed(1), multiplier: 36 },
        { key: 'FG_PCT', label: 'Field Goal %', formatter: (val) => (val * 100).toFixed(1) + '%', multiplier: 100 },
        { key: 'TS_PCT', label: 'True Shooting %', formatter: (val) => (val * 100).toFixed(1) + '%', multiplier: 100 },
        { key: 'PLUS_MINUS', label: 'Plus/Minus', formatter: (val) => val.toFixed(1), multiplier: 1 },
        { key: 'GAME_SCORE', label: 'Game Score', formatter: (val) => val.toFixed(1), multiplier: 1 },
        
        // Minutes prediction
        { key: 'MIN', label: 'Minutes', formatter: (val) => val.toFixed(1), multiplier: 1 },
        
        // Fantasy metrics
        { key: 'THREES', label: '3-Pointers', formatter: (val) => val.toFixed(1), multiplier: 1 },
        { key: 'TWOS', label: '2-Pointers', formatter: (val) => val.toFixed(1), multiplier: 1 },
        { key: 'FTM', label: 'Free Throws', formatter: (val) => val.toFixed(1), multiplier: 1 },
        { key: 'REB', label: 'Rebounds', formatter: (val) => val.toFixed(1), multiplier: 1 },
        { key: 'AST', label: 'Assists', formatter: (val) => val.toFixed(1), multiplier: 1 },
        { key: 'BLK', label: 'Blocks', formatter: (val) => val.toFixed(1), multiplier: 1 },
        { key: 'STL', label: 'Steals', formatter: (val) => val.toFixed(1), multiplier: 1 },
        { key: 'TOV', label: 'Turnovers', formatter: (val) => val.toFixed(1), multiplier: 1 }
    ];
    
    let tableHtml = '';
    
    // Create header section for core metrics
    tableHtml += '<tr class="table-light"><th colspan="4">Core Metrics</th></tr>';
    
    // Add core metrics (first 5)
    for (let i = 0; i < 5; i++) {
        const metric = metrics[i];
        addMetricRow(metric);
    }
    
    // Create header for minutes prediction
    tableHtml += '<tr class="table-light"><th colspan="4">Minutes Projection</th></tr>';
    
    // Add minutes prediction
    addMetricRow(metrics[5]);
    
    // Create header for fantasy metrics
    tableHtml += '<tr class="table-light"><th colspan="4">Fantasy Metrics</th></tr>';
    
    // Add fantasy metrics (last 8)
    for (let i = 6; i < metrics.length; i++) {
        const metric = metrics[i];
        addMetricRow(metric);
    }
    
    // Helper function to add a metric row
    function addMetricRow(metric) {
        // Check if we have predictions for this metric for both players
        const player1Has = player1Prediction && player1Prediction[metric.key];
        const player2Has = player2Prediction && player2Prediction[metric.key];
        
        if (player1Has && player2Has) {
            const player1Val = player1Prediction[metric.key].forecast;
            const player2Val = player2Prediction[metric.key].forecast;
            const diff = (player1Val - player2Val) * metric.multiplier;
            
            // Determine which player is better for this metric
            let player1Class = '';
            let player2Class = '';
            let diffHtml = '';
            
            // For TOV (turnovers), lower is better
            const isTOV = metric.key === 'TOV';
            
            if ((diff > 0 && !isTOV) || (diff < 0 && isTOV)) {
                player1Class = 'text-success fw-bold';
                diffHtml = `<span class="text-success">+${Math.abs(diff).toFixed(1)}</span>`;
            } else if ((diff < 0 && !isTOV) || (diff > 0 && isTOV)) {
                player2Class = 'text-success fw-bold';
                diffHtml = `<span class="text-success">+${Math.abs(diff).toFixed(1)}</span>`;
            } else {
                diffHtml = `<span class="text-muted">0.0</span>`;
            }
            
            tableHtml += `
                <tr>
                    <td>${metric.label}</td>
                    <td class="text-center ${player1Class}">${metric.formatter(player1Val)}</td>
                    <td class="text-center">${diffHtml}</td>
                    <td class="text-center ${player2Class}">${metric.formatter(player2Val)}</td>
                </tr>
            `;
        }
    }
    
    if (tableHtml === '') {
        tableHtml = `
            <tr>
                <td colspan="4" class="text-center text-muted py-3">
                    No prediction data available
                </td>
            </tr>
        `;
    }
    
    $('#prediction-table').html(tableHtml);
}

// Create recent games chart
function createRecentGamesChart() {
    if (!player1Stats || !player2Stats) {
        return;
    }
    
    const ctx = document.getElementById('recentGamesChart').getContext('2d');
    
    // Get data for the current metric
    const player1Data = getCurrentMetricData(player1Stats);
    const player2Data = getCurrentMetricData(player2Stats);
    
    // Labels (Game 1, Game 2, etc.)
    const labels = Array.from({ length: 10 }, (_, i) => `Game ${i + 1}`);
    
    if (recentGamesChart) {
        recentGamesChart.destroy();
    }
    
    recentGamesChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: p1Raw,
                    data: player1Data,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    tension: 0.1
                },
                {
                    label: p2Raw,
                    data: player2Data,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: getMetricTitle()
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                },
                x: {
                    title: {
                        display: true,
                        text: 'Recent Games'
                    }
                }
            }
        }
    });
}

// Update recent games chart when metric changes
function updateRecentGamesChart() {
    if (!recentGamesChart || !player1Stats || !player2Stats) {
        return;
    }
    
    // Get data for the new metric
    const player1Data = getCurrentMetricData(player1Stats);
    const player2Data = getCurrentMetricData(player2Stats);
    
    // Update chart data
    recentGamesChart.data.datasets[0].data = player1Data;
    recentGamesChart.data.datasets[1].data = player2Data;
    
    // Update chart title
    recentGamesChart.options.plugins.title.text = getMetricTitle();
    
    // Update the chart
    recentGamesChart.update();
}

// Get data for the current metric
function getCurrentMetricData(stats) {
    // Get last 10 games
    const recentGames = stats.slice(0, 10).reverse();
    
    // Format data based on the metric
    let data;
    
    switch (currentMetric) {
        case 'PTS_PER_MIN':
            // Convert points per minute to points per game
            data = recentGames.map(game => game.PTS_PER_MIN * game.MINUTES_PLAYED);
            break;
        case 'FG_PCT':
        case 'TS_PCT':
            // Convert percentages to 0-100 scale
            data = recentGames.map(game => game[currentMetric] * 100);
            break;
        default:
            data = recentGames.map(game => game[currentMetric]);
    }
    
    return data;
}

// Get title for the current metric
function getMetricTitle() {
    const titles = {
        'GAME_SCORE': 'Game Score (Last 10 Games)',
        'PTS_PER_MIN': 'Points (Last 10 Games)',
        'PLUS_MINUS': 'Plus/Minus (Last 10 Games)',
        'FG_PCT': 'Field Goal % (Last 10 Games)'
    };
    
    return titles[currentMetric] || `${currentMetric} (Last 10 Games)`;
}

// Get a metric label for display
function getMetricLabel(metric) {
    const labels = {
        'PTS_PER_MIN': 'Points Per Min',
        'FG_PCT': 'Field Goal %',
        'TS_PCT': 'True Shooting %',
        'PLUS_MINUS': 'Plus/Minus',
        'GAME_SCORE': 'Game Score',
        'OFF_RATING': 'Offensive Rating',
        'DEF_RATING': 'Defensive Rating',
        'AST_TO_RATIO': 'Assist/TO Ratio',
        'MINUTES_PLAYED': 'Minutes',
        'THREES': '3-Pointers',
        'TWOS': '2-Pointers',
        'FTM': 'Free Throws',
        'REB': 'Rebounds',
        'AST': 'Assists',
        'BLK': 'Blocks',
        'STL': 'Steals',
        'TOV': 'Turnovers',
        'MIN': 'Minutes',
        'FGM': 'Field Goals Made',
        'FGA': 'Field Goals Att.',
        'THREE_PCT': '3PT %'
    };
    
    return labels[metric] || metric;
}