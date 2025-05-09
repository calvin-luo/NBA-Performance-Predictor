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

// Metric categories and key metrics
let CATEGORIES = {
    "fantasy":    ["THREES", "TWOS", "FTM", "REB", "AST", "BLK", "STL", "TOV"],
    "volume":     ["MIN", "FGM", "FGA"],
    "efficiency": ["FG_PCT", "TS_PCT", "TOV", "THREE_PCT"],
};
let KEY_METRICS = [];
// Flatten and deduplicate metrics from all categories
for (const category in CATEGORIES) {
    CATEGORIES[category].forEach(metric => {
        if (!KEY_METRICS.includes(metric)) {
            KEY_METRICS.push(metric);
        }
    });
}
// Sort the metrics alphabetically
KEY_METRICS.sort();

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
    
    // Add animated effects
    initAnimations();
    
    // Handle mobile menu
    initMobileMenu();
    
    // Load today's games if the container exists
    if ($('#today-games').length > 0) {
        loadTodayGames();
    }
    
    // Load metric categories from API if needed
    if (typeof CATEGORIES_LOADED === 'undefined' || !CATEGORIES_LOADED) {
        loadMetricCategories();
    }
});

// ======= Handle Player Search =======

/**
 * Handle player search form submission
 * @param {Event} event - The form submission event
 */
function handleSearch(event) {
    event.preventDefault();
    const playerName = document.getElementById('player-search').value.trim();
    
    if (playerName) {
        window.location.href = "/player/" + encodeURIComponent(playerName);
    }
}

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
    if ($('#player-search').length > 0 && $.fn.autocomplete) {
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

/**
 * Load metric categories from API
 */
function loadMetricCategories() {
    $.ajax({
        url: '/api/metric_categories',
        method: 'GET',
        dataType: 'json',
        success: function(data) {
            if (data.categories && data.key_metrics) {
                CATEGORIES = data.categories;
                KEY_METRICS = data.key_metrics;
                window.CATEGORIES_LOADED = true;
                
                // Trigger an event that other code can listen for
                $(document).trigger('metrics-loaded');
            }
        },
        error: function(xhr, status, error) {
            console.error('Error loading metric categories:', error);
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

/**
 * Render category tabs for metrics visualization
 * @param {string} containerId - ID of container to render tabs in
 * @param {function} onTabChange - Callback function when tab changes
 */
function renderCategoryTabs(containerId, onTabChange) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Create tabs HTML
    let tabsHtml = `
        <ul class="nav nav-tabs" id="${containerId}-tabs" role="tablist">
    `;
    
    // Add a tab for each category
    Object.keys(CATEGORIES).forEach((category, index) => {
        const active = index === 0 ? 'active' : '';
        const selected = index === 0 ? 'true' : 'false';
        
        // Format category name for display (capitalize first letter)
        const displayName = category.charAt(0).toUpperCase() + category.slice(1);
        
        tabsHtml += `
            <li class="nav-item" role="presentation">
                <button class="nav-link ${active}" id="${containerId}-${category}-tab" 
                    data-bs-toggle="tab" data-bs-target="#${containerId}-${category}" 
                    type="button" role="tab" aria-controls="${category}" 
                    aria-selected="${selected}" data-category="${category}">
                    ${displayName}
                </button>
            </li>
        `;
    });
    
    tabsHtml += `</ul>`;
    
    // Create tab content HTML
    tabsHtml += `
        <div class="tab-content" id="${containerId}-tabContent">
    `;
    
    // Add content for each category
    Object.keys(CATEGORIES).forEach((category, index) => {
        const active = index === 0 ? 'show active' : '';
        
        tabsHtml += `
            <div class="tab-pane fade ${active}" id="${containerId}-${category}" 
                role="tabpanel" aria-labelledby="${containerId}-${category}-tab">
                <div class="p-3" id="${containerId}-${category}-content">
                    <!-- Content will be loaded dynamically -->
                </div>
            </div>
        `;
    });
    
    tabsHtml += `</div>`;
    
    // Set the HTML to the container
    container.innerHTML = tabsHtml;
    
    // Add event listener for tab changes
    if (typeof onTabChange === 'function') {
        $(`#${containerId}-tabs .nav-link`).on('shown.bs.tab', function (e) {
            const category = $(e.target).data('category');
            onTabChange(category);
        });
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
 * @returns {Promise} Promise resolving to player predictions
 */
function fetchPlayerPredictions(playerName) {
    return new Promise((resolve, reject) => {
        // Build the API endpoint
        const endpoint = `/api/player_prediction/${encodeURIComponent(playerName)}`;
        
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
 * Submit a lineup for prediction
 * @param {Array} playerNames - Array of player names in lineup
 * @returns {Promise} Promise resolving to lineup prediction
 */
function submitLineupPrediction(playerNames) {
    return new Promise((resolve, reject) => {
        // Build the API endpoint
        const endpoint = '/api/lineup_projection';
        
        // Prepare data
        const data = {
            players: playerNames
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

/**
 * Fetch today's games from API
 * @returns {Promise} Promise resolving to games data
 */
function fetchTodayGames() {
    return new Promise((resolve, reject) => {
        // Make the request
        $.ajax({
            url: '/api/today_games',
            method: 'GET',
            dataType: 'json',
            success: function(data) {
                resolve(data);
            },
            error: function(xhr, status, error) {
                console.error('Error fetching today\'s games:', error);
                reject(error);
            }
        });
    });
}

/**
 * Load and display today's games
 */
function loadTodayGames() {
    // Show loading state
    $('#today-games').html(`
        <tr>
            <td colspan="4" class="text-center py-3">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Loading today's games...</p>
            </td>
        </tr>
    `);
    
    // Fetch games
    fetchTodayGames()
        .then(function(data) {
            // Process games data
            let gamesHtml = '';
            
            // Check if we have games data
            if (data.games && data.games.length > 0) {
                // Process each game
                data.games.forEach(function(game) {
                    // Add row for this game
                    gamesHtml += `
                        <tr>
                            <td>${game.game_time || 'TBD'}</td>
                            <td>${game.away_team}</td>
                            <td>${game.home_team}</td>
                            <td>${game.venue || 'TBD'}</td>
                        </tr>
                    `;
                });
                
                // Update the games count
                $('#games-count').text(`${data.games.length} Games`);
            } else {
                // No games found
                gamesHtml = `
                    <tr>
                        <td colspan="4" class="text-center py-3">
                            <i class="fas fa-basketball-ball fa-3x text-muted mb-3"></i>
                            <h5>No games scheduled for today</h5>
                            <p class="text-muted">Check back later.</p>
                        </td>
                    </tr>
                `;
                
                // Update games count
                $('#games-count').text('0 Games');
            }
            
            // Update the table
            $('#today-games').html(gamesHtml);
        })
        .catch(function(error) {
            // Show error
            $('#today-games').html(`
                <tr>
                    <td colspan="4" class="text-center py-3">
                        <div class="text-danger">
                            <i class="fas fa-exclamation-circle fa-3x mb-3"></i>
                            <h5>Error loading games</h5>
                            <p>Please try again later.</p>
                        </div>
                    </td>
                </tr>
            `);
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
                        stepSize: 20
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
 * Create a grouped bar chart for category metrics
 * @param {string} canvasId - ID of canvas element
 * @param {string} category - Category name
 * @param {Array} datasets - Array of dataset objects with category metrics
 * @param {string} title - Chart title
 * @returns {Chart} Chart instance
 */
function createCategoryChart(canvasId, category, datasets, title = '') {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy existing chart if it exists
    if (window[canvasId + 'Chart']) {
        window[canvasId + 'Chart'].destroy();
    }
    
    // Get metrics for this category
    const metrics = CATEGORIES[category] || [];
    
    // Format labels for display
    const labels = metrics.map(metric => getMetricLabel(metric));
    
    // Extract data for each dataset
    const formattedDatasets = datasets.map(dataset => {
        const data = metrics.map(metric => {
            // Find the metric in the dataset
            return dataset.data[metric] || 0;
        });
        
        return {
            label: dataset.label,
            data: data,
            backgroundColor: dataset.backgroundColor || CHART_COLORS.primary,
            borderColor: dataset.borderColor || CHART_COLORS.primary,
            borderWidth: 1
        };
    });
    
    // Create new chart
    window[canvasId + 'Chart'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: formattedDatasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: !!title,
                    text: title || category.charAt(0).toUpperCase() + category.slice(1) + ' Metrics'
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

/**
 * Format a metric value for display
 * @param {number} value - Metric value
 * @param {string} metric - Metric key
 * @returns {string} Formatted value for display
 */
function formatMetricValue(value, metric) {
    if (value === undefined || value === null || isNaN(value)) return '-';
    
    // Format based on metric type
    if (metric === 'FG_PCT' || metric === 'TS_PCT' || metric === 'THREE_PCT') {
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