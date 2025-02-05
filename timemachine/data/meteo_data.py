import os, joblib
from meteostat import Hourly


def fetch(location : set, period='60d', reload=False):
    cache_name = f"{__file__[:-3]}_meteo__cache"
    data = {}

    if os.path.exists(cache_name) and not reload:
        data = joblib.load(cache_name)
        missing_tickers = tickers.difference(set(data.keys()))
        if missing_tickers:
            print(f"Fetching {len(missing_tickers)} non-cached tickers.")
            for tck in missing_tickers:
                try:
                    data[tck] = yf.download(tck, period=period, interval=interval)
                except:
                    print(f"Failed to download {tck} at interval:{interval} and period:{period}.")
            joblib.dump(data, cache_name)
    else:
        print("Making time series cache.")
        for tck in tickers:
            try:
                data[tck] = yf.download(tck, period=period, interval=interval)
            except:
                print(f"Failed to download {tck} at interval:{interval} and period:{period}.")
        joblib.dump(data, cache_name)

    return data