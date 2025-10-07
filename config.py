"""
⚙️ Configuration File - Advanced Diamond Analysis & Prediction System
====================================================================

Centralized configuration management for the diamond analysis system.
This file contains all customizable parameters, settings, and options.

Key Configuration Sections:
- Model Configuration: XGBoost parameters and training settings
- Feature Engineering: Feature creation and validation parameters
- Dashboard Settings: UI/UX customization options
- Performance Tuning: Memory, caching, and processing settings
- Business Logic: Thresholds and decision criteria

Usage:
- Modify parameters here instead of hardcoding in main files
- Ensures consistent settings across the application
- Easy to customize for different environments or requirements

Author: AI Assistant
Version: 1.0
"""

# Model Configuration
# XGBoost parameters and training settings for optimal performance
MODEL_CONFIG = {
    'algorithm': 'xgboost',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'early_stopping_rounds': 50,
    'n_jobs': -1,
    'test_size': 0.2,  # 20% for testing
    'validation_split': 0.8  # 80% for training
}

# Feature Engineering Configuration
# Parameters for creating and validating advanced features
FEATURE_CONFIG = {
    'min_category_size': 20,  # Minimum records per category for analysis
    'max_categories_analyze': 10,  # Maximum categories to analyze in detail
    'rolling_window_days': 5,  # Days for rolling statistics
    'momentum_window': 3,  # Days for momentum calculation
    'volatility_window': 5,  # Days for volatility calculation
}

# Quality Scoring Maps
# Standardized scoring systems for diamond quality assessment
QUALITY_MAPS = {
    'cut': {'3EX': 5, 'EX': 4, 'VG': 3, 'VGDN': 2, 'GD': 1},
    'color': {chr(ord('D') + i): 10 - i for i in range(11)},
    'clarity': {
        'FL': 12, 'IF': 11, 'VVS1': 10, 'VVS2': 9, 'VS1': 8, 'VS2': 7,
        'SI1': 6, 'SI2': 5, 'SI3': 4, 'I1': 3, 'I2': 2, 'I3': 1
    },
    'fluorescence': {'NONE': 4, 'FAINT': 3, 'MEDIUM': 2, 'STRONG': 1}
}

# Trend Analysis Configuration
# Statistical parameters for trend detection and demand forecasting
TREND_CONFIG = {
    'min_sales_for_trend': 5,  # Minimum sales for trend analysis
    'min_price_points': 3,  # Minimum price points for trend analysis
    'trend_threshold': 0.1,  # Minimum slope for trend classification
    'confidence_level': 0.95,  # Confidence level for predictions
}

# Dashboard Configuration
# UI/UX settings for the Streamlit web interface
DASHBOARD_CONFIG = {
    'page_title': "💎 Advanced Diamond Analysis & Prediction System",
    'page_icon': "💎",
    'layout': "wide",
    'initial_sidebar_state': "expanded",
    'max_rows_display': 1000,  # Maximum rows to display in data explorer
    'chart_height': 500,  # Default chart height
    'color_scheme': 'viridis',  # Default color scheme for charts
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'default_rap_rate': 1650,
    'default_carat': 0.55,
    'default_days_inventory': 30,
    'confidence_multiplier': 1.96,  # 95% confidence interval
    'min_prediction_confidence': 0.7,  # Minimum confidence for predictions
}

# File Configuration
FILE_CONFIG = {
    'supported_formats': ['.csv'],
    'max_file_size_mb': 100,
    'encoding': 'utf-8',
    'date_formats': ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y'],
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'enable_caching': True,
    'cache_ttl': 3600,  # Cache time-to-live in seconds
    'chunk_size': 10000,  # Process data in chunks
    'max_memory_usage_mb': 4000,  # Maximum memory usage
    'parallel_processing': True,
    'n_jobs': -1,  # Use all available cores
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'default_theme': 'plotly_white',
    'color_continuous_scale': 'viridis',
    'opacity': 0.7,
    'line_width': 2,
    'marker_size': 6,
    'font_size': 12,
    'title_font_size': 16,
}

# Business Logic Configuration
BUSINESS_CONFIG = {
    'high_demand_threshold': 0.7,  # Sales velocity threshold for high demand
    'low_demand_threshold': 0.3,  # Sales velocity threshold for low demand
    'price_increase_threshold': 2.0,  # Minimum price increase for "rising" trend
    'price_decrease_threshold': -2.0,  # Maximum price decrease for "falling" trend
    'inventory_pressure_threshold': 50,  # High inventory count threshold
}

# Export Configuration
EXPORT_CONFIG = {
    'default_format': 'csv',
    'include_metadata': True,
    'compression': None,  # 'gzip', 'bz2', 'zip', 'xz', or None
    'float_format': '%.2f',
    'date_format': '%Y-%m-%d',
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'diamond_analysis.log',
    'max_size_mb': 10,
    'backup_count': 5,
}

# Validation Configuration
VALIDATION_CONFIG = {
    'required_columns_in_out': [
        'ereport_no', 'in_date', 'shape', 'color', 'clarity', 
        'florecent', 'cut', 'size', 'cut_group'
    ],
    'required_columns_price': [
        'ereport_no', 'pdate', 'rap_rate', 'disc', 'rate'
    ],
    'min_data_points': 100,  # Minimum data points for analysis
    'max_missing_percentage': 0.1,  # Maximum 10% missing values
}

# Advanced Features Configuration
ADVANCED_CONFIG = {
    'enable_deep_learning': False,  # Enable LSTM/GRU models
    'enable_ensemble': False,  # Enable ensemble methods
    'enable_hyperparameter_tuning': False,  # Enable automatic tuning
    'enable_feature_selection': True,  # Enable automatic feature selection
    'enable_cross_validation': True,  # Enable cross-validation
    'cv_folds': 5,  # Number of cross-validation folds
}

# API Configuration (for future use)
API_CONFIG = {
    'enable_api': False,  # Enable REST API
    'api_host': 'localhost',
    'api_port': 8000,
    'api_debug': False,
    'cors_enabled': True,
    'rate_limit': 100,  # Requests per minute
}

# Security Configuration
SECURITY_CONFIG = {
    'enable_authentication': False,  # Enable user authentication
    'session_timeout': 3600,  # Session timeout in seconds
    'max_upload_size': 100,  # Maximum upload size in MB
    'allowed_file_types': ['.csv', '.xlsx', '.xls'],
    'sanitize_inputs': True,  # Sanitize user inputs
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    'enable_notifications': False,  # Enable email notifications
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email_from': 'noreply@diamondanalysis.com',
    'email_to': 'admin@diamondanalysis.com',
    'notification_events': ['model_training_complete', 'prediction_accuracy_low'],
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'enable_monitoring': False,  # Enable system monitoring
    'monitor_interval': 300,  # Monitoring interval in seconds
    'alert_thresholds': {
        'cpu_usage': 80,  # CPU usage percentage
        'memory_usage': 80,  # Memory usage percentage
        'disk_usage': 90,  # Disk usage percentage
        'response_time': 5,  # Response time in seconds
    },
    'log_metrics': True,  # Log performance metrics
}

# Customization Helpers
# Utility functions for easy configuration management
def get_model_params():
    """Get model parameters for easy customization."""
    return MODEL_CONFIG.copy()

def get_feature_params():
    """Get feature engineering parameters."""
    return FEATURE_CONFIG.copy()

def get_quality_maps():
    """Get quality scoring maps."""
    return QUALITY_MAPS.copy()

def update_config(section, key, value):
    """Update configuration values dynamically."""
    if section in globals():
        if hasattr(globals()[section], key):
            globals()[section][key] = value
            return True
    return False

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Validate model config
    if MODEL_CONFIG['test_size'] + MODEL_CONFIG['validation_split'] != 1.0:
        errors.append("Model test_size + validation_split must equal 1.0")
    
    if MODEL_CONFIG['learning_rate'] <= 0 or MODEL_CONFIG['learning_rate'] > 1:
        errors.append("Learning rate must be between 0 and 1")
    
    # Validate feature config
    if FEATURE_CONFIG['min_category_size'] < 1:
        errors.append("Minimum category size must be at least 1")
    
    # Validate trend config
    if TREND_CONFIG['confidence_level'] < 0 or TREND_CONFIG['confidence_level'] > 1:
        errors.append("Confidence level must be between 0 and 1")
    
    return errors

# Configuration validation on import
if __name__ == "__main__":
    errors = validate_config()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration is valid!")
else:
    # Validate configuration when imported
    config_errors = validate_config()
    if config_errors:
        import warnings
        for error in config_errors:
            warnings.warn(f"Configuration error: {error}")
