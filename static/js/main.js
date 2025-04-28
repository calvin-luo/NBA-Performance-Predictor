/**
 * Stat Lemon - NBA Performance Predictor
 * Main JavaScript File
 */

// ======= Global Variables =======
// NBA Teams list
const NBA_TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
];

// Chart colors
const CHART_COLORS = {
    primary: 'rgb(54, 162, 235)',
    primaryLight: 'rgba(54, 162, 235, 0.2)',
    secondary: 'rgb(255, 99, 132)',
    secondaryLight: 'rgba(255, 99, 132, 0.2)',
    tertiary: 'rgb(75, 192, 192)',
    tertiaryLight: 'rgba(75, 192, 192, 0.2)',
    success: 'rgb(40, 167, 69)',
    successLight: 'rgba(40, 167, 69, 0.2)',
    warning: 'rgb(255, 193, 7)',
    warningLight: 'rgba(255, 193, 7, 0.2)',
    danger: 'rgb(220, 53, 69)',
    dangerLight: 'rgba(220, 53, 69, 0.2)',
    gray: 'rgb(108, 117, 125)',
    grayLight: 'rgba(108, 117, 125, 0.2)'
};

// Global chart defaults
if (typeof Chart !== 'undefined') {
    Chart.defaults.font.family = "'Roboto', 'Segoe UI', Arial, sans-serif";
    Chart.defaults.color = '#666';
    Chart.defaults.animation.duration = 1000;
}

// ======= Document Ready =======
$(document).ready(function() {
    // Initialize tooltips
    initTooltips();
    
    // Initialize player search autocomplete
    initPlayerSearchAutocomplete();
    
    // Initialize date pickers
    initDatePickers();
    
    // Add animated effects
    initAnimations();
    
    // Handle mobile menu
    initMobileMenu();
});

// ======= Initialization Functions =======

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize player search autocomplete
 */
function initPlayerSearchAutocomplete() {
    if ($('#player-search').length > 0) {
        $('#player-search').autocomplete({
            source: function(request, response) {
                // In production, this would be an API call
                $.ajax({
                    url: "/api/search_player",
                    data: { q: request.term },
                    dataType: "json",
                    success: function(data) {
                        response(data);
                    },
                    error: function() {
                        // Fallback to dummy data for MVP
                        const dummyPlayers = [
                            "LeBron James", "Stephen Curry", "Kevin Durant", 
                            "Giannis Antetokounmpo", "Luka Dončić", "Nikola Jokić",
                            "Joel Embiid", "Kawhi Leonard", "Jayson Tatum", "Damian Lillard"
                        ];
                        
                        response(dummyPlayers.filter(player => 
                            player.toLowerCase().includes(request.term.toLowerCase())
                        ));
                    }
                });
            },
            minLength: 2,
            select: function(event, ui) {
                window.location.href = "/player/" + encodeURIComponent(ui.item.value);
                return false;
            }
        });
    }
}

/**
 * Initialize date pickers across the app
 */
function initDatePickers() {
    // Initialize any date pickers
    if (typeof $.fn.datepicker !== 'undefined' && $('.datepicker').length > 0) {
        $('.datepicker').datepicker({
            format: 'yyyy-mm-dd',
            autoclose: true,
            todayHighlight: true
        });
    }
}

/**
 * Initialize scroll and fade animations
 */
function initAnimations() {
    // Animate elements when they come into view
    const animateOnScroll = function() {
        $('.animate-on-scroll').each(function() {
            const elementPos = $(this).offset().top;
            const topOfWindow = $(window).scrollTop();
            const windowHeight = $(window).height();
            
            if (elementPos < topOfWindow + windowHeight - 50) {
                $(this).addClass('animated');
            }
        });
    };
    
    // Run on page load and scroll
    animateOnScroll();
    $(window).scroll(animateOnScroll);
}

/**
 * Initialize mobile menu behavior
 */
function initMobileMenu() {
    // Close the mobile menu when a menu item is clicked
    $('.navbar-nav .nav-link').click(function() {
        if ($('.navbar-collapse').hasClass('show')) {
            $('.navbar-toggler').click();
        }
    });
}

// ======= UI Helper Functions =======

/**
 * Create HTML for a loading spinner
 * @param {string} size - Size of spinner (sm, md, lg)
 * @param {string} text - Text to display next to spinner
 * @returns {string} HTML for loading spinner
 */
function createLoadingSpinner(size = '', text = 'Loading...') {
    const sizeClass = size ? `spinner-border-${size}` : '';
    return `
        <div class="text-center py-3">
            <div class="spinner-border ${sizeClass} text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">${text}</p>
        </div>
    `;
}

/**
 * Format a date string for display
 * @param {string} dateString - Date string in format YYYY-MM-DD
 * @returns {string} Formatted date string
 */
function formatDate(dateString) {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        weekday: 'long',
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    });
}

/**
 * Get a date string offset from today
 * @param {number} dayOffset - Number of days to offset from today
 * @returns {string} Date string in format YYYY-MM-DD
 */
function getDateString(dayOffset) {
    const date = new Date();
    date.setDate(date.getDate() + dayOffset);
    return date.toISOString().split('T')[0];
}

/**
 * Get team logo URL
 * @param {string} teamName - Full team name
 * @returns {string} URL to team logo
 */
function getTeamLogoUrl(teamName) {
    // Remove spaces and convert to lowercase
    const formattedName = teamName.replace(/\s+/g, '').toLowerCase();
    return `/static/img/teams/${formattedName}.png`;
}

/**
 * Show a message alert
 * @param {string} message - Message to display
 * @param {string} type - Alert type (success, danger, warning, info)
 * @param {string} container - Selector for container to append alert to
 * @param {boolean} dismissible - Whether alert should be dismissible
 */
function showAlert(message, type = 'info', container = '#alerts-container', dismissible = true) {
    // Create alert HTML
    const dismissBtn = dismissible ? '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>' : '';
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            ${dismissBtn}
        </div>
    `;
    
    // Append to container
    $(container).append(alertHtml);
    
    // Auto-dismiss after 5 seconds
    if (dismissible) {
        setTimeout(() => {
            $('.alert').alert('close');
        }, 5000);
    }
}

// ======= API Interaction Functions =======

/**
 * Fetch player statistics from API
 * @param {string} playerName - Name of player
 * @param {boolean} forceRefresh - Whether to force refresh from API
 * @returns {Promise} Promise resolving to player stats
 */
function fetchPlayerStats(playerName, forceRefresh = false) {
    return new Promise((resolve, reject) => {
        // Build the API endpoint
        const endpoint = `/api/player_stats/${encodeURIComponent(playerName)}${forceRefresh ? '?refresh=true' : ''}`;
        
        // Make the request
        $.ajax({
            url: endpoint,
            method: 'GET',
            dataType: 'json',
            success: function(data) {
                resolve(data);
            },
            error: function(xhr, status, error) {
                console.error('Error fetching player stats:', error);
                reject(error);
            }
        });
    });
}

/**
 * Fetch player predictions from API
 * @param {string} playerName - Name of player
 * @param {string} opponent - Optional opponent team name
 * @returns {Promise} Promise resolving to player predictions
 */
function fetchPlayerPredictions(playerName, opponent = '') {
    return new Promise((resolve, reject) => {
        // Build the API endpoint
        const endpoint = `/api/player_prediction/${encodeURIComponent(playerName)}${opponent ? '?opponent=' + encodeURIComponent(opponent) : ''}`;
        
        // Make the request
        $.ajax({
            url: endpoint,
            method: 'GET',
            dataType: 'json',
            success: function(data) {
                resolve(data);
            },
            error: function(xhr, status, error) {
                console.error('Error fetching player predictions:', error);
                reject(error);
            }
        });
    });
}

/**
 * Fetch players comparison from API
 * @param {Array} playerNames - Array of player names to compare
 * @returns {Promise} Promise resolving to comparison data
 */
function fetchPlayersComparison(playerNames) {
    return new Promise((resolve, reject) => {
        // Build the API endpoint
        const endpoint = `/api/compare_players?players=${playerNames.join(',')}`;
        
        // Make the request
        $.ajax({
            url: endpoint,
            method: 'GET',
            dataType: 'json',
            success: function(data) {
                resolve(data);
            },
            error: function(xhr, status, error) {
                console.error('Error fetching players comparison:', error);
                reject(error);
            }
        });
    });
}

/**
 * Submit a lineup for prediction
 * @param {Array} playerNames - Array of player names in lineup
 * @param {string} opponent - Optional opponent team name
 * @returns {Promise} Promise resolving to lineup prediction
 */
function submitLineupPrediction(playerNames, opponent = '') {
    return new Promise((resolve, reject) => {
        // Build the API endpoint
        const endpoint = '/api/lineup_prediction';
        
        // Prepare data
        const data = {
            players: playerNames,
            opponent: opponent || null
        };
        
        // Make the request
        $.ajax({
            url: endpoint,
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            dataType: 'json',
            success: function(data) {
                resolve(data);
            },
            error: function(xhr, status, error) {
                console.error('Error submitting lineup prediction:', error);
                reject(error);
            }
        });
    });
}

// ======= Chart Helper Functions =======

/**
 * Create a radar chart for comparing player attributes
 * @param {string} canvasId - ID of canvas element
 * @param {Array} labels - Array of labels for radar axes
 * @param {Array} datasets - Array of dataset objects
 * @returns {Chart} Chart instance
 */
function createRadarChart(canvasId, labels, datasets) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (window[canvasId + 'Chart']) {
        window[canvasId + 'Chart'].destroy();
    }
    
    // Create new chart
    window[canvasId + 'Chart'] = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100,
                    ticks: {
                        stepSize: 20,
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
    
    return window[canvasId + 'Chart'];
}

/**
 * Create a line chart for time series data
 * @param {string} canvasId - ID of canvas element
 * @param {Array} labels - Array of labels for x-axis
 * @param {Array} datasets - Array of dataset objects
 * @param {string} title - Chart title
 * @returns {Chart} Chart instance
 */
function createLineChart(canvasId, labels, datasets, title = '') {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (window[canvasId + 'Chart']) {
        window[canvasId + 'Chart'].destroy();
    }
    
    // Create new chart
    window[canvasId + 'Chart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: !!title,
                    text: title
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
    
    return window[canvasId + 'Chart'];
}

/**
 * Create a bar chart for comparing values
 * @param {string} canvasId - ID of canvas element
 * @param {Array} labels - Array of labels for x-axis
 * @param {Array} datasets - Array of dataset objects
 * @param {string} title - Chart title
 * @returns {Chart} Chart instance
 */
function createBarChart(canvasId, labels, datasets, title = '') {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (window[canvasId + 'Chart']) {
        window[canvasId + 'Chart'].destroy();
    }
    
    // Create new chart
    window[canvasId + 'Chart'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: !!title,
                    text: title
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
    
    return window[canvasId + 'Chart'];
}

/**
 * Create a doughnut chart
 * @param {string} canvasId - ID of canvas element
 * @param {Array} labels - Array of labels for segments
 * @param {Array} data - Array of data values
 * @param {Array} colors - Array of colors for segments
 * @param {string} title - Chart title
 * @returns {Chart} Chart instance
 */
function createDoughnutChart(canvasId, labels, data, colors, title = '') {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (window[canvasId + 'Chart']) {
        window[canvasId + 'Chart'].destroy();
    }
    
    // Create new chart
    window[canvasId + 'Chart'] = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: !!title,
                    text: title
                },
                legend: {
                    position: 'bottom'
                }
            },
            cutout: '50%'
        }
    });
    
    return window[canvasId + 'Chart'];
}

// ======= Player Utility Functions =======

/**
 * Calculate average value for a metric from player stats
 * @param {Array} playerStats - Array of player game stats
 * @param {string} metric - Metric to calculate average for
 * @returns {number} Average value
 */
function calculateAverage(playerStats, metric) {
    if (!playerStats || playerStats.length === 0) return 0;
    
    const values = playerStats.map(game => parseFloat(game[metric])).filter(val => !isNaN(val));
    return values.reduce((sum, val) => sum + val, 0) / values.length;
}

/**
 * Get a metric label for display
 * @param {string} metric - Metric key
 * @returns {string} Display label for metric
 */
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
        'MINUTES_PLAYED': 'Minutes'
    };
    
    return labels[metric] || metric;
}

/**
 * Format a metric value for display
 * @param {number} value - Metric value
 * @param {string} metric - Metric key
 * @returns {string} Formatted value for display
 */
function formatMetricValue(value, metric) {
    if (value === undefined || value === null || isNaN(value)) return '-';
    
    // Format based on metric type
    if (metric === 'FG_PCT' || metric === 'TS_PCT') {
        // Percentage metrics
        return (value * 100).toFixed(1) + '%';
    } else if (metric === 'PTS_PER_MIN') {
        // Per minute metrics - show as per 36 minutes
        return (value * 36).toFixed(1);
    } else if (metric === 'PLUS_MINUS') {
        // Plus/minus can be negative
        return value > 0 ? '+' + value.toFixed(1) : value.toFixed(1);
    } else {
        // Default formatting
        return value.toFixed(1);
    }
}

// ======= Team Utility Functions =======

/**
 * Get NBA team abbreviation from full name
 * @param {string} teamName - Full team name
 * @returns {string} Team abbreviation
 */
function getTeamAbbreviation(teamName) {
    const teamAbbreviations = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS"
    };
    
    return teamAbbreviations[teamName] || '';
}

/**
 * Populate a team dropdown select
 * @param {string} selectId - ID of select element
 * @param {string} initialValue - Initial value to select
 */
function populateTeamDropdown(selectId, initialValue = '') {
    // Get the select element
    const $select = $('#' + selectId);
    
    // Check if it exists
    if ($select.length === 0) return;
    
    // Empty options list
    let options = '<option value="">Select Team</option>';
    
    // Add all teams
    NBA_TEAMS.forEach(team => {
        const selected = initialValue === team ? 'selected' : '';
        options += `<option value="${team}" ${selected}>${team}</option>`;
    });
    
    // Update select options
    $select.html(options);
}