"""
Test script for evaluating ML prediction strategies across multiple symbols.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import required components
from src.market_data.okx_client import OKXClient
from src.ml.price_predictor import MLModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_strategy_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ml_strategy_tester")


class MLStrategyTester:
    """Test ML prediction strategies across multiple symbols and models."""

    def __init__(self):
        """Initialize the strategy tester."""
        self.client = OKXClient()
        self.model_manager = MLModelManager()

        # Test parameters
        self.timeframe = "4h"
        self.prediction_periods = 5
        self.data_points = 2000
        self.hyperparameter_tuning = True
        self.model_types = ["random_forest", "gradient_boosting", "linear", "svr"]

        # Results storage
        self.results = []

    def get_top_symbols(self, limit=10):
        """Get top volume symbols from the exchange."""
        logger.info(f"Fetching top {limit} symbols by volume...")
        return self.client.get_top_volume_symbols(limit=limit)

    def fetch_data(self, symbol):
        """Fetch historical data for a symbol."""
        logger.info(f"Fetching data for {symbol} ({self.timeframe}, {self.data_points} points)...")

        try:
            klines = self.client.get_klines(
                symbol,
                interval=self.timeframe,
                limit=self.data_points
            )

            if not klines or len(klines) < 100:
                logger.warning(f"Insufficient data for {symbol} (got {len(klines) if klines else 0} points)")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])

            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def test_model(self, symbol, df, model_type):
        """Test a specific model type on a symbol."""
        logger.info(f"Testing {model_type} model for {symbol}...")

        try:
            # Get predictor
            predictor = self.model_manager.get_predictor(symbol, self.timeframe)

            # Train model
            train_results = predictor.train_model(
                df,
                symbol=symbol,
                timeframe=self.timeframe,
                model_type=model_type,
                hyperparameter_tuning=self.hyperparameter_tuning
            )

            # Evaluate model
            metrics, _ = predictor.evaluate_model(df, model_type=model_type)

            # Make predictions
            predictions = predictor.predict(
                df,
                model_type=model_type,
                prediction_periods=self.prediction_periods
            )

            # Calculate prediction trend
            first_price = predictions.iloc[0]['predicted_close']
            last_price = predictions.iloc[-1]['predicted_close']
            price_change_pct = ((last_price - first_price) / first_price) * 100

            # Calculate confidence interval width (normalized)
            avg_price = predictions['predicted_close'].mean()
            avg_interval_width = (predictions['upper_bound'] - predictions['lower_bound']).mean()
            normalized_interval_width = (avg_interval_width / avg_price) * 100

            # Get top features if available
            top_features = None
            if hasattr(predictor.models[model_type]['model'], 'feature_importances_'):
                feature_importance = predictor._get_feature_importance(
                    predictor.models[model_type]['model'],
                    predictor.models[model_type]['feature_names']
                )
                # Get top 5 features
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

            # Store results
            result = {
                'symbol': symbol,
                'model_type': model_type,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'mse': metrics['mse'],
                'price_change_pct': price_change_pct,
                'confidence_width_pct': normalized_interval_width,
                'prediction_direction': 'up' if price_change_pct > 0 else 'down',
                'top_features': top_features,
                'current_price': df['close'].iloc[-1],
                'predicted_prices': predictions['predicted_close'].tolist(),
                'lower_bounds': predictions['lower_bound'].tolist(),
                'upper_bounds': predictions['upper_bound'].tolist(),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.info(
                f"Test completed for {symbol} with {model_type}: R²={metrics['r2']:.4f}, Direction: {result['prediction_direction']}")
            return result

        except Exception as e:
            logger.error(f"Error testing {model_type} model for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def run_tests(self):
        """Run tests on all symbols and models."""
        # Get top symbols
        symbols = self.get_top_symbols(limit=10)

        if not symbols:
            logger.error("Failed to fetch symbols. Exiting.")
            return

        logger.info(f"Testing {len(symbols)} symbols with {len(self.model_types)} model types")
        logger.info(
            f"Parameters: Timeframe={self.timeframe}, Periods={self.prediction_periods}, Data Points={self.data_points}")

        # Process each symbol
        for symbol in tqdm(symbols, desc="Testing Symbols"):
            # Fetch data
            df = self.fetch_data(symbol)

            if df is None:
                continue

            # Test each model type
            for model_type in self.model_types:
                result = self.test_model(symbol, df, model_type)

                if result:
                    self.results.append(result)

        # Save results
        self.save_results()

        # Generate report
        self.generate_report()

    def save_results(self):
        """Save test results to CSV."""
        if not self.results:
            logger.warning("No results to save")
            return

        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)

        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ml_strategy_test_results_{timestamp}.csv"

        # Handle top_features column (convert list to string)
        if 'top_features' in results_df.columns:
            results_df['top_features'] = results_df['top_features'].apply(
                lambda x: ', '.join([f"{f}:{v:.4f}" for f, v in x]) if x else None
            )

        results_df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")

    def generate_report(self):
        """Generate a summary report of the test results."""
        if not self.results:
            logger.warning("No results to generate report")
            return

        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)

        # 1. Best overall model by R²
        best_overall = results_df.loc[results_df['r2'].idxmax()]

        # 2. Best model for each symbol
        best_by_symbol = results_df.loc[results_df.groupby('symbol')['r2'].idxmax()]

        # 3. Best model type overall
        model_type_performance = results_df.groupby('model_type')['r2'].mean().sort_values(ascending=False)

        # 4. Symbols with strongest upward predictions
        upward_predictions = results_df[results_df['prediction_direction'] == 'up'].sort_values('price_change_pct',
                                                                                                ascending=False)

        # 5. Symbols with most reliable predictions (highest R²)
        most_reliable = results_df.sort_values('r2', ascending=False)

        # Generate report
        report = [
            "=" * 80,
            "ML STRATEGY TEST REPORT",
            "=" * 80,
            f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Parameters: Timeframe={self.timeframe}, Periods={self.prediction_periods}, Data Points={self.data_points}, Hyperparameter Tuning={self.hyperparameter_tuning}",
            "=" * 80,
            "\nBEST OVERALL MODEL:",
            f"Symbol: {best_overall['symbol']}",
            f"Model Type: {best_overall['model_type']}",
            f"R²: {best_overall['r2']:.4f}",
            f"RMSE: {best_overall['rmse']:.4f}",
            f"Prediction Direction: {best_overall['prediction_direction']}",
            f"Price Change: {best_overall['price_change_pct']:.2f}%",
            "=" * 80,
            "\nBEST MODEL FOR EACH SYMBOL:",
        ]

        for _, row in best_by_symbol.iterrows():
            report.append(
                f"{row['symbol']}: {row['model_type']} (R²: {row['r2']:.4f}, Direction: {row['prediction_direction']}, Change: {row['price_change_pct']:.2f}%)")

        report.extend([
            "=" * 80,
            "\nMODEL TYPE PERFORMANCE (Average R²):",
        ])

        for model_type, r2 in model_type_performance.items():
            report.append(f"{model_type}: {r2:.4f}")

        report.extend([
            "=" * 80,
            "\nTOP 5 SYMBOLS WITH STRONGEST UPWARD PREDICTIONS:",
        ])

        for i, (_, row) in enumerate(upward_predictions.head(5).iterrows(), 1):
            report.append(
                f"{i}. {row['symbol']} ({row['model_type']}): +{row['price_change_pct']:.2f}% (R²: {row['r2']:.4f})")

        report.extend([
            "=" * 80,
            "\nTOP 5 MOST RELIABLE PREDICTIONS (Highest R²):",
        ])

        for i, (_, row) in enumerate(most_reliable.head(5).iterrows(), 1):
            report.append(
                f"{i}. {row['symbol']} ({row['model_type']}): R²={row['r2']:.4f}, Direction: {row['prediction_direction']}, Change: {row['price_change_pct']:.2f}%")

        report.append("=" * 80)

        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"ml_strategy_test_report_{timestamp}.txt"

        with open(report_filename, 'w') as f:
            f.write('\n'.join(report))

        logger.info(f"Report saved to {report_filename}")

        # Print report to console
        print('\n'.join(report))

        # Generate visualization
        self.visualize_results(results_df)

        def visualize_results(self, results_df):
            """Create visualizations of the test results."""
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Model performance by symbol
            performance_pivot = results_df.pivot_table(
                index='symbol',
                columns='model_type',
                values='r2',
                aggfunc='mean'
            ).sort_values(by='random_forest', ascending=False)

            performance_pivot.plot(
                kind='bar',
                ax=axes[0, 0],
                title='Model Performance by Symbol (R²)',
                ylabel='R² Score'
            )
            axes[0, 0].set_xlabel('Symbol')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].legend(title='Model Type')

            # 2. Prediction direction by model type
            direction_counts = results_df.groupby(['model_type', 'prediction_direction']).size().unstack()
            direction_counts.plot(
                kind='bar',
                stacked=True,
                ax=axes[0, 1],
                title='Prediction Direction by Model Type',
                ylabel='Count'
            )
            axes[0, 1].set_xlabel('Model Type')
            axes[0, 1].legend(title='Direction')

            # 3. Price change % vs R² (scatter plot)
            for model_type in results_df['model_type'].unique():
                model_data = results_df[results_df['model_type'] == model_type]
                axes[1, 0].scatter(
                    model_data['r2'],
                    model_data['price_change_pct'],
                    label=model_type,
                    alpha=0.7
                )

            axes[1, 0].set_title('Price Change % vs Model Reliability (R²)')
            axes[1, 0].set_xlabel('R² Score')
            axes[1, 0].set_ylabel('Predicted Price Change %')
            axes[1, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            axes[1, 0].legend(title='Model Type')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Confidence interval width vs R² (scatter plot)
            for model_type in results_df['model_type'].unique():
                model_data = results_df[results_df['model_type'] == model_type]
                axes[1, 1].scatter(
                    model_data['r2'],
                    model_data['confidence_width_pct'],
                    label=model_type,
                    alpha=0.7
                )

            axes[1, 1].set_title('Confidence Interval Width vs Model Reliability (R²)')
            axes[1, 1].set_xlabel('R² Score')
            axes[1, 1].set_ylabel('Confidence Interval Width %')
            axes[1, 1].legend(title='Model Type')
            axes[1, 1].grid(True, alpha=0.3)

            # Adjust layout and save
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f"ml_strategy_test_visualization_{timestamp}.png", dpi=300)
            logger.info(f"Visualization saved to ml_strategy_test_visualization_{timestamp}.png")

            # Create additional chart: Best trading opportunities
            self.visualize_trading_opportunities(results_df)

        def visualize_trading_opportunities(self, results_df):
            """Create a visualization of the best trading opportunities."""
            # Filter for reliable models (R² > 0.6) with upward predictions
            opportunities = results_df[
                (results_df['r2'] > 0.6) &
                (results_df['prediction_direction'] == 'up')
                ].sort_values('price_change_pct', ascending=False)

            if len(opportunities) == 0:
                logger.warning("No reliable upward predictions found for trading opportunities visualization")
                return

            # Take top 10 or fewer
            opportunities = opportunities.head(10)

            # Create figure
            plt.figure(figsize=(12, 8))

            # Create bar chart of price change %
            bars = plt.bar(
                opportunities['symbol'] + ' (' + opportunities['model_type'] + ')',
                opportunities['price_change_pct'],
                color='green',
                alpha=0.7
            )

            # Add R² as text on bars
            for bar, r2 in zip(bars, opportunities['r2']):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.3,
                    f'R²: {r2:.3f}',
                    ha='center',
                    va='bottom',
                    rotation=0,
                    fontsize=9
                )

            plt.title('Top Trading Opportunities (Reliable Upward Predictions)')
            plt.xlabel('Symbol (Model Type)')
            plt.ylabel('Predicted Price Change %')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f"ml_trading_opportunities_{timestamp}.png", dpi=300)
            logger.info(f"Trading opportunities visualization saved to ml_trading_opportunities_{timestamp}.png")

def main():
    """Run the ML strategy tester."""
    print("Starting ML Strategy Tester...")
    tester = MLStrategyTester()
    tester.run_tests()
    print("Testing completed. Check the generated report and visualizations.")

if __name__ == "__main__":
    main()
