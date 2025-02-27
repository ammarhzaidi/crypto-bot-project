import sys
import os
import pytest

# Add the src directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.hh_hl_strategy import detect_hh_hl, detect_lh_ll, analyze_price_action


def test_detect_uptrend():
    """Test that uptrend detection works properly."""
    # Clear uptrend with HH and HL pattern
    uptrend = [10, 8, 15, 12, 20, 17, 25, 22, 30]

    result = analyze_price_action(uptrend)

    assert result["uptrend_analysis"]["higher_highs"] == True
    assert result["uptrend_analysis"]["higher_lows"] == True


def test_detect_downtrend():
    """Test that downtrend detection works properly."""
    # Clear downtrend with LH and LL pattern
    downtrend = [30, 32, 25, 27, 20, 22, 15, 17, 10]

    result = analyze_price_action(downtrend)

    assert result["downtrend_analysis"]["lower_highs"] == True
    assert result["downtrend_analysis"]["lower_lows"] == True