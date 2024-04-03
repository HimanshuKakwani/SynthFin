import streamlit as st
import yfinance as yf
import plotly.graph_objs as go

def home():
    st.title("SynthFin")
    st.subheader("Your Personal Fund Manager")

    # Fetching data
    nifty_data = yf.Ticker('^NSEI').history(period='10y')
    gold_data = yf.Ticker('GC=F').history(period='10y')
    sp500_data = yf.Ticker('^GSPC').history(period='10y')
    usdinr_data = yf.Ticker('INR=X').history(period='10y')

    # Displaying cards in a 2x2 grid
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Nifty 50")
        if len(nifty_data) >= 2:
            st.write("Current Level:", nifty_data['Close'].iloc[-1])
            st.write("Change:", nifty_data['Close'].iloc[-1] - nifty_data['Close'].iloc[-2])
        else:
            st.write("Insufficient data for Nifty 50")

        st.markdown("### Gold")
        if len(gold_data) >= 2:
            st.write("Current Level:", gold_data['Close'].iloc[-1])
            st.write("Change:", gold_data['Close'].iloc[-1] - gold_data['Close'].iloc[-2])
        else:
            st.write("Insufficient data for Gold")

    with col2:
        st.markdown("### S&P 500")
        if len(sp500_data) >= 2:
            st.write("Current Level:", sp500_data['Close'].iloc[-1])
            st.write("Change:", sp500_data['Close'].iloc[-1] - sp500_data['Close'].iloc[-2])
        else:
            st.write("Insufficient data for S&P 500")

        st.markdown("### USD/INR")
        if len(usdinr_data) >= 2:
            st.write("Current Level:", usdinr_data['Close'].iloc[-1])
            st.write("Change:", usdinr_data['Close'].iloc[-1] - usdinr_data['Close'].iloc[-2])
        else:
            st.write("Insufficient data for USD-INR")
        # Chart selector
    chart_type = st.selectbox("Select Chart", ["Nifty 50", "Gold", "S&P 500", "USD/INR"])
    timeframe = st.selectbox("Select Timeframe", ["1y", "1d","10y", "max"])  # Adjusted timeframe options
    chart_kind = "line"  # Default to line chart
    if st.checkbox("Show Candlestick Chart"):
        chart_kind = "candlestick"

    # Map chart type to corresponding ticker symbol
    chart_type_map = {"Nifty 50": "^NSEI", "Gold": "GC=F", "S&P 500": "^GSPC", "USD/INR": "INR=X"}
    selected_ticker = chart_type_map.get(chart_type)

    if selected_ticker:
        selected_data = yf.Ticker(selected_ticker).history(period=timeframe)
        st.subheader(chart_type)
        st.plotly_chart(create_chart(selected_data, chart_kind))

def create_chart(data, chart_kind):
    fig = go.Figure()
    

    if chart_kind == "line":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    elif chart_kind == "candlestick":
        fig.add_trace(go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candlestick'))
    fig.update_layout(xaxis_title='Date', yaxis_title='Price')
    return fig
    
# def get_score():
#     st.title("Get Score Page")
#     import streamlit as st

def get_score():
    st.title("Get Score Page")
    st.subheader("Please fill in the following details to calculate your risk score:")

    with st.form("risk_form"):
        age = st.number_input("Age", min_value=0)
        dependants = st.number_input("Number of Dependents", min_value=0)
        investment_amount = st.number_input("Investment Amount", min_value=0)
        drawdown = st.number_input("Percentage Loss you can handle", min_value=0)
        time_period = st.number_input("Time Period of Investment (in years)", min_value=0)

        submitted = st.form_submit_button("Calculate Risk Score")

    if submitted:
        risk_score = 0
        
        if drawdown <= 10:
            risk_score += 1
        elif 10 < drawdown <= 30:
            risk_score += 2
        elif drawdown > 30:
            risk_score += 3
        
        if investment_amount <= 1200000:
            risk_score += 1
        elif 1200000 < investment_amount <= 3600000:
            risk_score += 2
        elif investment_amount > 36000000:
            risk_score += 3
        
        if dependants <= 2:
            risk_score += 3
        elif 2 < dependants <= 5:
            risk_score += 2
        
        if age <= 40:
            risk_score += 3
        elif 40 < age <= 60:
            risk_score += 2
        
        st.write("Your risk score is:", risk_score)

        # Store risk score, time period, and investment amount for later use
        st.session_state.risk_score = risk_score
        st.session_state.time_period = time_period
        st.session_state.investment_amount = investment_amount

    # Add content for Get Score page here

def portfolio():
    import streamlit as st
    st.title("Portfolio Page")
    import pandas as pd
    import random
    from datetime import datetime, timedelta
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    # import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import Callback
    import yfinance as yf
    import ta

    def fetch_stock_data(symbol, start_date, end_date):
        data = yf.download(symbol, start=start_date, end=end_date)
        return data

    def calculate_technical_indicators(data):
        data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        return data

    def preprocess_data(data):
        data.dropna(inplace=True)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        sequence_length = 20
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length, 3])
        X_train, y_train = np.array(X), np.array(y)
        return X_train, y_train, scaler

    def create_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        return model

    def generate_portfolio(stock_list, bonds_list, risk_score):
        stock_list = pd.read_excel("Stock List.xlsx")
        stock_list = list(stock_list["Ticker"])[:10]
        bonds_list = pd.read_excel("Bonds List.xlsx")
        bonds_list = list(bonds_list["Ticker"])[:10]
        gold_funds = ["BSLGOLDETF.NS","TATAGOLD.NS"]
        
        if risk_score >= 10:
            assets = list(random.sample(stock_list, 6))
        elif 6 < risk_score < 10:
            debt_assets = list(random.sample(bonds_list, 2))
            equity_assets = list(random.sample(stock_list, 4))
            assets = equity_assets + debt_assets
        else:
            gold_assets = list(random.sample(gold_funds, 1))
            debt_assets = list(random.sample(bonds_list, 2))
            equity_assets = list(random.sample(stock_list, 3))
            assets = equity_assets + debt_assets + gold_assets

        weights = [round(1 / len(assets) * 100, 2)] * len(assets)
        portfolio = pd.DataFrame({"Assets": assets, "Weights": weights})
        return portfolio

    def predict_prices_for_tickers(tickers, start_date, end_date, stock_list, bonds_list, risk_score):
        predictions_df = pd.DataFrame()
        for ticker in tickers:
            try:
                # Fetch data
                stock_data = fetch_stock_data(ticker, start_date, end_date)
                # Calculate technical indicators
                stock_data = calculate_technical_indicators(stock_data)
                # Preprocess the data
                X_train, y_train, scaler = preprocess_data(stock_data)
                # Reshape input data for LSTM
                input_shape = (X_train.shape[1], X_train.shape[2])
                # Create and train the LSTM model
                model = create_lstm_model(input_shape)
                epoch_logger = EpochLogger()
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[epoch_logger])
                # Make predictions for the next month
                future_dates = pd.date_range(end_date, periods=30, freq='D')[1:]  # Exclude the end date itself
                future_data = fetch_stock_data(ticker, end_date, future_dates[-1])
                future_data = calculate_technical_indicators(future_data)
                X_future, _, _ = preprocess_data(future_data)
                predictions = model.predict(X_future)
                # Rescale predictions to original scale
                predictions = scaler.inverse_transform(np.concatenate((np.zeros((len(predictions), X_train.shape[1]-1)), predictions), axis=1))[:, -1]
                # Create DataFrame for predictions
                prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions})
                prediction_df['Ticker'] = ticker
                predictions_df = pd.concat([predictions_df, prediction_df], ignore_index=True)
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
        portfolio = generate_portfolio(stock_list, bonds_list, risk_score)
        return predictions_df, portfolio

    def portfolio1():
        st.title("Portfolio Page")
        st.subheader("Your Dynamically Allocated Portfolio")
        
        # Load risk score, time period, and investment amount from session state
        risk_score = st.session_state.risk_score
        
        # Determine start date based on time period (e.g., 10 years before current date)
        time_period = st.session_state.time_period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period * 365)

        # Fetch stock and bonds list from Excel files
        stock_list = pd.read_excel("Stock List.xlsx")
        stock_list = list(stock_list["Ticker"])
        bonds_list = pd.read_excel("Bonds List.xlsx")
        bonds_list = list(bonds_list["Ticker"])

        # Concatenate stock and bonds lists
        tickers = stock_list + bonds_list

        # Predict prices and generate portfolio
        predictions, portfolio = predict_prices_for_tickers(tickers, start_date, end_date, stock_list, bonds_list, risk_score)

        # # Display predictions and portfolio
        # st.write("Predicted Prices for Selected Tickers:")
        # st.write(predictions)
        st.write("Dynamically Allocated Portfolio:")
        st.write(portfolio)

    portfolio_df = portfolio1()
    # portfolio_df= pd.DataFrame(portfolio_df)
    # # print(portfolio_df.type)
    # st.session_state.portfolio_df = portfolio_df

    
def analysis():    
    st.title('Analysis')
    # Add content for Analysis page here

def wishlist():
    st.title("Wishlist Page")
    # Add content for Wishlist page here

def rebalancing():
    st.title("Rebalancing Page")
    # Add content for Rebalancing page here

def about_us():
    st.title("About Us Page")
    # Add content for About Us page here
