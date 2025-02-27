import os
import time
import logging
import argparse
import schedule
from datetime import datetime
from src.analysis.okx_tb_10 import OKXMarketAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_tracker.log')
    ]
)
logger = logging.getLogger('market_tracker')

# Make sure the output directory exists
os.makedirs('reports', exist_ok=True)


def track_market(hours_ago=4):
    """Track market and save reports to file."""
    try:
        # Initialize analyzer
        analyzer = OKXMarketAnalyzer(min_volume=1000000)  # $1M minimum volume

        # Get current timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Get top gainers and losers
        gainers, losers = analyzer.get_top_gainers_losers()

        # Save to CSV
        gainers.to_csv(f'reports/top_gainers_{timestamp}.csv', index=False)
        losers.to_csv(f'reports/top_losers_{timestamp}.csv', index=False)

        # Get comparison if we have historical data
        try:
            gainers_comp, losers_comp = analyzer.compare_with_previous(hours_ago=hours_ago)
            if not gainers_comp.empty:
                gainers_comp.to_csv(f'reports/gainers_comparison_{timestamp}.csv', index=False)
            if not losers_comp.empty:
                losers_comp.to_csv(f'reports/losers_comparison_{timestamp}.csv', index=False)
        except Exception as e:
            logger.warning(f"Comparison not available yet: {str(e)}")

        # Print to console
        analyzer.print_top_tables()
        analyzer.print_comparison(hours_ago=hours_ago)

        logger.info(f"Market tracking completed and saved to reports directory at {timestamp}")

    except Exception as e:
        logger.error(f"Error tracking market: {str(e)}")


def main():
    """Main function to run the market tracker with scheduling."""
    parser = argparse.ArgumentParser(description='Track OKX market data periodically')
    parser.add_argument('--interval', type=float, default=4.0,
                        help='Interval in hours between tracking runs (default: 4)')
    parser.add_argument('--compare', type=float, default=4.0,
                        help='Compare with data from this many hours ago (default: 4)')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit without scheduling')

    args = parser.parse_args()

    # Log startup
    logger.info(f"Market tracker started with interval={args.interval}h, compare={args.compare}h")

    # Run once immediately
    logger.info("Running initial market tracking...")
    track_market(hours_ago=args.compare)

    # Exit if --once flag was provided
    if args.once:
        logger.info("Completed one-time run as requested")
        return

    # Schedule regular runs
    interval_minutes = int(args.interval * 60)
    schedule.every(interval_minutes).minutes.do(track_market, hours_ago=args.compare)

    logger.info(f"Scheduled to run every {interval_minutes} minutes")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Market tracker stopped by user")


if __name__ == "__main__":
    main()