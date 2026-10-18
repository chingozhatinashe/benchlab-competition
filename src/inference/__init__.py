"""Forecast orchestration: fetch inputs -> build features -> run model."""

FORECAST_HOURS = 72


def run_forecast(t0):
    # Return 72 hourly solar wind speeds (km/s) for t0+1h .. t0+72h.
    # Placeholder baseline until a real pipeline is wired in.
    return [420.0] * FORECAST_HOURS
