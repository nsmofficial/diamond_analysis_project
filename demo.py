#!/usr/bin/env python3
"""
🎯 Demo Script - Advanced Diamond Analysis & Prediction System
=============================================================

This script generates realistic sample data for testing and demonstration purposes.
Creates 1,000 fake diamonds with realistic properties, price history, and sales data.

Features:
- Generates realistic diamond inventory data
- Creates price change history with market patterns
- Simulates sales data with realistic time-to-sale patterns
- Provides comprehensive data summary and statistics

Usage:
- Run this script to generate demo data files
- Upload generated files to test the system
- Perfect for demonstrations and testing without real business data

Author: AI Assistant
Version: 1.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_demo_data():
    """
    Create sample data for demonstration purposes.
    Generates realistic diamond inventory and price history data.
    """
    print("🎯 Creating demo data for the Diamond Analysis System...")
    
    # Create sample IN-OUT data
    # Set random seed for reproducible results
    np.random.seed(42)
    n_diamonds = 1000
    
    # Generate diamond IDs
    diamond_ids = [f"DEMO{i:06d}" for i in range(n_diamonds)]
    
    # Generate sample data with realistic distributions
    # Diamond properties with market-realistic probability distributions
    shapes = ['ROUND'] * n_diamonds
    colors = np.random.choice(['D', 'E', 'F', 'G', 'H', 'I', 'J'], n_diamonds, p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1])
    clarities = np.random.choice(['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2'], n_diamonds, 
                                p=[0.05, 0.1, 0.15, 0.15, 0.2, 0.15, 0.15, 0.05])
    fluorescences = np.random.choice(['NONE', 'FAINT', 'MEDIUM', 'STRONG'], n_diamonds, 
                                   p=[0.6, 0.25, 0.1, 0.05])
    cuts = np.random.choice(['3EX', 'EX', 'VG', 'VGDN', 'GD'], n_diamonds, 
                          p=[0.3, 0.4, 0.2, 0.08, 0.02])
    sizes = np.random.choice(['0.200-0.239', '0.240-0.299', '0.300-0.329', '0.330-0.399', 
                            '0.500-0.599', '0.700-0.799'], n_diamonds, 
                           p=[0.2, 0.2, 0.15, 0.2, 0.15, 0.1])
    
    # Generate dates with realistic inventory patterns
    start_date = datetime(2024, 1, 1)
    in_dates = [start_date + timedelta(days=np.random.randint(0, 300)) for _ in range(n_diamonds)]
    
    # Simulate sales with realistic probability and timing
    sold_probability = 0.3  # 30% of diamonds are sold
    out_dates = []
    for i, in_date in enumerate(in_dates):
        if np.random.random() < sold_probability:
            days_in_inventory = np.random.randint(1, 180)  # 1-180 days to sell
            out_dates.append(in_date + timedelta(days=days_in_inventory))
        else:
            out_dates.append(None)  # Still in inventory
    
    # Create IN-OUT DataFrame
    in_out_data = {
        'ereport_no': diamond_ids,
        'in_date': in_dates,
        'shape': shapes,
        'color': colors,
        'clarity': clarities,
        'florecent': fluorescences,
        'cut': cuts,
        'symn': cuts,  # Simplified
        'polish': cuts,  # Simplified
        'size': sizes,
        'cut_group': cuts,
        'out_date': out_dates
    }
    
    in_out_df = pd.DataFrame(in_out_data)
    
    # Create price history data with realistic market patterns
    price_data = []
    
    for _, diamond in in_out_df.iterrows():
        # Generate 3-10 price changes per diamond (realistic frequency)
        n_changes = np.random.randint(3, 11)
        
        # Calculate base rap rate based on carat and quality
        carat_min = float(diamond['size'].split('-')[0])
        quality_multiplier = (10 - ord(diamond['color']) + ord('A')) / 10
        base_rap_rate = int(carat_min * 3000 * quality_multiplier)
        
        # Generate price history with realistic discount patterns
        current_discount = np.random.uniform(20, 60)  # Starting discount 20-60%
        
        for i in range(n_changes):
            # Price change date (realistic intervals)
            if i == 0:
                price_date = diamond['in_date']
            else:
                days_since_last = np.random.randint(1, 30)  # 1-30 days between changes
                price_date = price_data[-1]['pdate'] + timedelta(days=days_since_last)
            
            # Discount changes over time (generally increases for unsold diamonds)
            if i > 0:
                discount_change = np.random.uniform(-2, 5)  # Can decrease or increase
                current_discount = max(10, min(80, current_discount + discount_change))
            
            # Calculate final rate after discount
            final_rate = base_rap_rate * (1 - current_discount / 100)
            
            price_data.append({
                'ereport_no': diamond['ereport_no'],
                'pdate': price_date,
                'rap_rate': base_rap_rate,
                'disc': round(current_discount, 1),
                'rate': int(final_rate)
            })
    
    price_df = pd.DataFrame(price_data)
    
    # Save demo data
    in_out_df.to_csv('demo_in_out.csv', index=False)
    price_df.to_csv('demo_price.csv', index=False)
    
    print(f"✅ Created demo data:")
    print(f"   - {len(in_out_df)} diamonds in inventory")
    print(f"   - {len(price_df)} price change records")
    print(f"   - {in_out_df['out_date'].notna().sum()} sold diamonds")
    print(f"   - Date range: {in_out_df['in_date'].min().date()} to {in_out_df['in_date'].max().date()}")
    
    return in_out_df, price_df

def run_demo():
    """
    Run the complete demo.
    Generates sample data and provides usage instructions.
    """
    print("💎 Advanced Diamond Analysis & Prediction System - Demo")
    print("=" * 60)
    
    # Check if demo data exists, create if needed
    if not (os.path.exists('demo_in_out.csv') and os.path.exists('demo_price.csv')):
        create_demo_data()
    else:
        print("📊 Using existing demo data...")
    
    # Load and display demo data summary
    in_out_df = pd.read_csv('demo_in_out.csv')
    price_df = pd.read_csv('demo_price.csv')
    
    print("\n📈 Demo Data Summary:")
    print(f"   - Total diamonds: {len(in_out_df)}")
    print(f"   - Price records: {len(price_df)}")
    print(f"   - Sold diamonds: {in_out_df['out_date'].notna().sum()}")
    print(f"   - Unique categories: {in_out_df.groupby(['shape', 'size', 'color', 'clarity', 'cut_group', 'florecent']).ngroups}")
    
    print("\n🎯 Key Features Demonstrated:")
    print("   1. ✅ Advanced Feature Engineering (25+ features)")
    print("   2. ✅ Time-series aware modeling")
    print("   3. ✅ Category-specific analysis")
    print("   4. ✅ Trend analysis and forecasting")
    print("   5. ✅ Interactive prediction interface")
    print("   6. ✅ Professional dashboard with 5 tabs")
    print("   7. ✅ Real-time predictions with confidence intervals")
    print("   8. ✅ Comprehensive data visualization")
    
    print("\n🚀 To run the full system:")
    print("   streamlit run enhanced_main.py")
    print("\n📁 Upload these files in the dashboard:")
    print("   - demo_in_out.csv (IN-OUT data)")
    print("   - demo_price.csv (Price history data)")
    
    print("\n🎯 Interview Highlights:")
    print("   - Advanced ML with XGBoost and feature engineering")
    print("   - Time-series analysis and trend forecasting")
    print("   - Professional dashboard with Streamlit")
    print("   - Business intelligence and actionable insights")
    print("   - Production-ready code structure")
    print("   - Comprehensive documentation and README")

# Main execution block
if __name__ == "__main__":
    run_demo()
