import time
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.common import APIError
import datetime
from datetime import timedelta
import openai
import scraper
import stock_news
import yfinance as yf
from alpaca.trading.client import TradingClient
import os


class StockTradingBot:
    def __init__(self, alpaca_key, alpaca_secret, openai_key, apikey, openai_model="gpt-4o"):
        """
        Initializes the trading and OpenAI clients, and sets up prompts for analysis tasks.

        Args:
            alpaca_key (str): Your Alpaca API key used for authentication.
            alpaca_secret (str): Your Alpaca API secret key used for authentication.
            openai_key (str): Your OpenAI API key used for authentication.
            openai_model (str, optional): The model name for OpenAI API queries. Default is "gpt-4o".

        Attributes:
            alpaca_key (str): Stores the Alpaca API key.
            alpaca_secret (str): Stores the Alpaca API secret key.
            openai_key (str): Stores the OpenAI API key.
            trading_client (TradingClient): Instance of the TradingClient initialized with Alpaca API credentials.
            account (Account): The account information retrieved from the Alpaca API.
            openai_client (openai.OpenAI): The OpenAI client initialized with the provided API key.
            openai_model (str): The name of the OpenAI model to be used.
            num_of_stocks (int): A counter tracking the number of stocks, initialized to 0.
            limit_order_prompt (str): Prompt text for analyzing stock sell limit order prices.
            analyser_prompt (str): Prompt text for evaluating a company's growth potential as "evergreen".

        """
        self.alpaca_key = alpaca_key
        self.alpaca_secret = alpaca_secret
        self.openai_key = openai_key
        self.alphavantageapikey = apikey

        self.trading_client = TradingClient(
            self.alpaca_key, self.alpaca_secret, paper=True
        )
        self.account = self.trading_client.get_account()
        self.openai_client = openai.OpenAI(api_key=self.openai_key)
        self.openai_model = openai_model

        self.num_of_stocks = 0
        self.limit_order_prompt = (
            "Given historical price data, the current price of a stock, and the most recent news about the company whose stock you hold, "
            "your task is to analyze this information to determine an optimal sell limit order price. Please provide your decision as a numeric "
            "value representing the price at which you wish to set your sell limit order. You must reply in numbers only. Be as realistically optimistic as you can!"
        )
        self.analyser_prompt = (
            "You will be provided with a ticker symbol representing a company and a brief summary of recent news related to that company. "
            "Your task is to analyze this information and predict the company's growth potential over the next few years. Specifically, "
            "you are to determine if the company fits the definition of an 'evergreen' company, meaning it has strong growth prospects "
            "for the future, even if it is currently experiencing losses. "
            "- **Respond with 'Yes'** if, based on the information provided, you believe the company is evergreen and "
            "likely to grow in the coming years. "
            "- **Respond with 'No'** if, based on the information provided, you assess the company as not likely "
            "to achieve growth in the near future, considering its current situation. "
            "Remember, your analysis should take into account the specific details of the news provided and any known factors about the company's sector, operational model, or financial health that can influence its future growth potential. Strictly answer with a yes or no"
        )

    def get_current_price(self, ticker) -> float | None:
        """
        Gets the Current Price of a Ticker

        Args:
            ticker (str): Ticker Symbol

        Returns:
            str: The Current Price of A ticker
        """
        try:
            current_price = yf.Ticker(ticker).info["currentPrice"]
            return current_price
        except Exception as e:
            print(f"An error occurred when fetching current price for {ticker}: {e}")
            return None

    def round_to_two_decimals(self, price: float) -> float:
        """
        Rounds the decimal value to 2 places

        Args:
            price (float): The Price of the Stop Loss Limit, the Take Profit Limit, The Current Price etc

        Returns:
            float: Rounded off Value to Two Decimal PLaces
        """
        return round(price, 2)

    def get_historical_stock_data(
        self, symbol, start_date: datetime, end_date: datetime
    ) -> (list, float):
        """
        Fetches historical stock data and current price.

        This method uses the `yfinance` library to download the historical stock
        data for a given ticker symbol within a specified date range. It then extracts
        the closing price data, rounds the values, and returns them as a list along
        with the current stock price.

        Args:
            symbol (str): The ticker symbol of the stock.
            start_date (datetime): The start date for the historical data in 'YYYY-MM-DD' format.
            end_date (datetime): The end date for the historical data in 'YYYY-MM-DD' format.

        Returns:
            tuple: A tuple containing two elements:
                - list: A list of rounded closing prices between the start and end dates.
                - float: The current price of the stock.

        Raises:
            Exception: If an error occurs during the data retrieval, it is caught, printed, and the function returns None.
        """
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            data = stock_data["Close"].round().to_list()
            current_price = yf.Ticker(symbol).info["currentPrice"]
            return data, current_price
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def complete_chat_forward(self, blog, stock_symbol) -> str:
        """
        Generates a completion using the OpenAI API based on provided stock news and symbol.

        This method leverages the OpenAI API to analyze a given blog/news content related to a
        specific stock symbol and predict the company's growth potential. The response indicates
        if the company is considered 'evergreen' or not, as per the specified Analyzer Prompt.

        Args:
            blog (str): The news or blog content related to the company.
            stock_symbol (str): The ticker symbol of the stock.

        Returns:
            str: The completion result from OpenAI, indicating 'yes' or 'no' regarding the company's growth potential.
        """
        completion = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": self.analyser_prompt},
                {
                    "role": "user",
                    "content": f"Ticker Symbol: {stock_symbol} News: {blog}",
                },
            ],
            temperature=1,
            stream=False,
            seed=50,
        )
        return completion.choices[0].message.content.lower()

    def limit_order_predictor(self, data, news, curr_price) -> str:
        """
        Predicts an optimal sell limit order price using the OpenAI API.

        This method uses the OpenAI API to analyze provided historical stock data,
        current stock price, and recent news about the company. It generates a suggested
        sell limit order price based on the analysis.

        Args:
            data (str): A list of historical closing prices of the stock.
            news (str): Recent news about the company.
            curr_price (str): The current price of the stock.

        Returns:
            str: The predicted sell limit order price as a string. The completion result from
            OpenAI should be returned as a numeric value representing the price.
        """
        completion = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": self.limit_order_prompt},
                {
                    "role": "user",
                    "content": f"Historical data: {data} Current price:{curr_price} News: {news}",
                },
            ],
            temperature=0.7,
            stream=False,
            seed=50,
        )
        return completion.choices[0].message.content.lower()

    def order_management(
        self, ticker, quantity, take_profit_limit, stop_loss_price
    ) -> None:
        """
        Manages the placement of bracket orders including validation and submission.

        This method rounds the take profit and stop loss values to two decimal places,
        validates the order parameters, and places a bracket order using the Alpaca API.
        It ensures that the take profit limit is greater than the stop loss price and the
        order quantity is positive. If the validation passes, the order is submitted; otherwise,
        appropriate error messages are printed.

        Args:
            ticker (str): The ticker symbol of the stock.
            quantity (int): The number of shares to buy.
            take_profit_limit (float): The price at which to take profit.
            stop_loss_price (float): The price at which to trigger a stop loss.

        Returns:
            None

        Raises:
            APIError: If there is an issue submitting the order via the Alpaca API.
        """
        take_profit_limit = self.round_to_two_decimals(take_profit_limit)
        stop_loss_price = self.round_to_two_decimals(stop_loss_price)

        print(
            f"Placing order for {ticker}: Quantity = {quantity}, Take Profit Limit = {take_profit_limit}, Stop Loss = {stop_loss_price}"
        )

        if quantity <= 0:
            print(f"Invalid quantity ({quantity}) for {ticker}. Order skipped.")
            return

        if take_profit_limit <= stop_loss_price:
            print(
                f"Invalid order parameters for {ticker}: Take Profit Limit ({take_profit_limit}) must be greater than Stop Loss ({stop_loss_price}). Order skipped."
            )
            return

        bracket_order_data = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit_limit),
            stop_loss=StopLossRequest(stop_price=stop_loss_price),
        )

        try:
            self.trading_client.submit_order(order_data=bracket_order_data)
        except APIError as e:
            print(f"Failed to place order: {e}")

    def correct_allocate_percentages(self, input_dict) -> dict:
        """
        Adjusts the allocation percentages of a given dictionary to ensure certain constraints.

        This method takes a dictionary with allocation values, normalizes these values to
        percentages, and ensures that the highest percentage is at least 20% greater than the
        second highest. It then adjusts the remaining percentages to ensure the total sums
        up to 100%. The method returns a new dictionary with the keys mapped to their
        corrected percentage values.

        Args:
            input_dict (dict): A dictionary where keys are items to allocate and values
                               are their corresponding allocation values.

        Returns:
            dict: A dictionary where keys are the same items and values are their corrected
                  allocation percentages.
        """
        keys = list(input_dict.keys())
        values = list(input_dict.values())
        total_value = sum(values)

        indexed_values = sorted(
            ((value, index) for index, value in enumerate(values)), reverse=True
        )
        percentages = [(value / total_value) * 100 for value, _ in indexed_values]

        if len(percentages) > 1 and percentages[0] <= (percentages[1] * 1.2):
            percentages[0] = percentages[1] * 1.2

        extra_percent = sum(percentages) - 100
        for i in range(1, len(percentages)):
            percentages[i] -= (percentages[i] / sum(percentages[1:])) * extra_percent

        for i in range(len(percentages)):
            if percentages[i] < 0:
                percentages[i] = 0

        sum_percentages = sum(percentages)
        percentages = [p / sum_percentages * 100 for p in percentages]

        result = {}
        for (value, original_idx), percentage in zip(indexed_values, percentages):
            key = keys[original_idx]
            result[key] = percentage

        return result

    def logic_stock(self) -> None:
        """
        Core logic for stock analysis, decision-making, and order management.

        This method orchestrates the process of discovering stocks, analyzing their
        news, making buy/sell decisions, predicting limit sell order prices, and
        managing stock sell or hold orders. Specifically, it:
        - Retrieves stock symbols and corresponding news.
        - Analyzes the news for each stock to decide whether to buy.
        - Fetches historical data and current price for the stock if the decision is to buy.
        - Predicts the optimal limit sell order price based on historical data, current price, and news.
        - Collects the analysis results, including ticker symbol, limit order value, and spread.
        - Manages stock sell or hold orders based on the analysis results.

        The method also performs the necessary print operations for logging information about the decisions.

        Returns:
            None
        """
        output = []
        data = stock_news.stock_discovery(self.alphavantageapikey)
        for stock_symbol, news in data.items():
            blog = str(scraper.scrape_blog(news)).strip()
            decision = self.complete_chat_forward(blog, stock_symbol)
            if decision == "yes":
                print("BUY ", stock_symbol)
                start_date = datetime.datetime.today()
                end_date = start_date - timedelta(days=7)
                historical_prices, current_price = self.get_historical_stock_data(
                    stock_symbol, end_date, start_date
                )
                if historical_prices and current_price:
                    limit_order_val = self.limit_order_predictor(
                        str(historical_prices), blog, str(current_price)
                    )
                    print("The limit sell order: " + limit_order_val)
                    output.append(
                        {
                            "Ticker": stock_symbol,
                            "Limit": limit_order_val,
                            "Spread": float(limit_order_val) - current_price,
                        }
                    )
            else:
                print("DONT BUY ", stock_symbol)

        self.manage_stock_sell_or_hold(output)

    def extract_spreads(self, items):
        """
        Maps The Ticker Symbol to Its Spread of Predicted Limit Price and Current Price

        Args:
            items (list): The List Of Stocks, with their Features

        Returns:
            None
        """
        result = {item["Ticker"]: item["Spread"] for item in items}
        return result

    def calculate_stop_loss_price(self, current_price):
        """
        Calculates the Stop Loss Price by Taking the Stop Loss Price as 95% of the Current Price

        Args:
            current_price (float): The Current Price of a Particular Ticker

        Returns:
            float: Rounded off Value of the Stop Loss Price
        """
        stop_loss_percentage = 0.95  # 5% below the current price
        return self.round_to_two_decimals(current_price * stop_loss_percentage)

    def position_sizing(self, output):
        """
        Manages the allocation of cash for purchasing stocks based on their calculated spreads and predicted limits.

        This method performs the following steps:
        1. Extracts the available cash.
        2. Extracts and corrects the stock spreads from the output.
        3. Allocates available cash among stocks based on corrected allocation percentages.
        4. For each stock, fetches the current price and calculates the quantity to buy.
        5. Validates and defines take profit and stop loss prices, then places the order if valid.

        Args:
            output (list): A list of dictionaries containing stock information, with each dictionary
                           having keys 'Ticker', 'Limit', and 'Spread'.

        Returns:
            int: The number of positions currently held after performing the position sizing.
        """
        cash = int(float(self.account.cash))
        stock_spread = self.extract_spreads(output)
        stocks = self.correct_allocate_percentages(stock_spread)

        print(f"Total available cash: {cash}")
        print(f"Stock spreads: {stock_spread}")
        print(f"Stocks allocation percentages: {stocks}")

        for index, stock in enumerate(stocks):
            current_price = self.get_current_price(stock)
            if current_price is None:
                print(f"Skipping {stock} due to error in fetching current price.")
                continue

            current_price = self.round_to_two_decimals(current_price)
            print(f"Current price of {stock}: {current_price}")

            allotted_cash = (stocks[stock] / 100) * cash
            quantity = round(allotted_cash / current_price)
            print(f"Allotted cash for {stock}: {allotted_cash}, Quantity: {quantity}")

            if quantity > 0:
                take_profit_limit = self.round_to_two_decimals(
                    float(output[index]["Limit"])
                )
                stop_loss_price = self.calculate_stop_loss_price(current_price)
                if take_profit_limit > stop_loss_price:
                    self.order_management(
                        stock, quantity, take_profit_limit, stop_loss_price
                    )
                else:
                    print(
                        f"Invalid limits for {stock}: Take profit {take_profit_limit} must be greater than stop loss {stop_loss_price}."
                    )
            else:
                print(f"Skipping {stock} due to zero or negative quantity: {quantity}")

        return len(self.get_positions())

    def get_positions(self):
        """
        Gets All The Current Positions in Portfolio

        Returns:
            list: The List of The Positions
        """
        return self.trading_client.get_all_positions()

    def manage_stock_sell_or_hold(self, output):
        """
        Controller Logic of The Programme
        """
        self.num_of_stocks = self.position_sizing(output)
        while True:
            positions = self.get_positions()
            if len(positions) < self.num_of_stocks:
                self.logic_stock()
            else:
                print("Auto Trader is Going To Sleep, and will get Activated Tomorrow again to analyse your Portfolio")
                time.sleep(86400)

    def __call__(self):
        self.logic_stock()


bot = StockTradingBot(
    alpaca_key=os.getenv("ALPACA_API_KEY"),
    alpaca_secret=os.getenv("ALPACA_SECRET"),
    openai_key=os.getenv("OPENAI_API_KEY"),
    apikey=os.getenv("ALPHA_VANTAGE_API_KEY")
)
bot()
