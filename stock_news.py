import requests


def stock_discovery(apikey):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=blockchain&apikey={apikey}"
    r = requests.get(url)
    data = r.json()["feed"]
    vals_blogs = []
    keys_stock = list()
    for i in data:
        stocks = i["ticker_sentiment"]
        for stock in stocks:
            if stock["ticker_sentiment_label"] == "Bullish":
                if "CRYPTO" not in stock["ticker"]:
                    vals_blogs.append(i["url"])
                    keys_stock.append(stock["ticker"])

    res = {keys_stock[i]: vals_blogs[i] for i in range(len(keys_stock))}
    print(res)
    return res


# Acording to the blogs make the AI decide ona limit order, place it and store the ticker, the limit, the current status of order
