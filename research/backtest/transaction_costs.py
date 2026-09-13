def calculate_turnover(previous_weights, current_weights):
    """
    Portfolio turnover measured as the sum of absolute
    changes in portfolio weights.
    """

    if previous_weights is None:
        return 1.0

    tickers = set(previous_weights.index).union(
        current_weights.index
    )

    previous = previous_weights.reindex(
        tickers,
        fill_value=0.0
    )

    current = current_weights.reindex(
        tickers,
        fill_value=0.0
    )

    return float(
        (current - previous).abs().sum()
    )


def calculate_transaction_cost(
    previous_weights,
    current_weights,
    cost_bps=10
):
    """
    cost_bps=10 means 10 basis points = 0.10%.
    """

    turnover = calculate_turnover(
        previous_weights,
        current_weights
    )

    cost = turnover * cost_bps / 10000.0

    return cost, turnover