import streamlit as st
import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import sqlite3
import hashlib
from io import StringIO
import chardet
import base64


# import plotly.graph_objects as go

# Function to encode the local image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Detect file encoding
def detect_encoding(file):
    raw_data = file.read(10000)
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    file.seek(0)
    return encoding


class SalesData:
    def __init__(self, data):
        self.data = data

    def show_data(self):
        """Display the first few rows of the dataset."""
        return self.data.head()

    def best_selling_product(self, top_n):
        best_selling_prods = self.data.groupby('Product Name')['Quantity'].sum().sort_values(ascending=False).head(
            top_n)
        return best_selling_prods
    def best_selling_products_plot(self, top_n):

        best_selling_prods = self.data.groupby('Product Name')['Quantity'].sum().sort_values(ascending=False).head(
            top_n)

        # Convert the data into a DataFrame
        n_df = pd.DataFrame(best_selling_prods)

        # Reset index to make the 'Product Name' a column for plotting
        n_df.reset_index(inplace=True)

        # Create figure and axes using Seaborn style
        plt.figure(figsize=(12, 6))

        # Use Seaborn's barplot to create a nicer visual
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x='Product Name', y='Quantity', data=n_df, palette="crest")
        # Customize the plot
        ax.set_title('Best Selling Products vs Quantity Sold', fontsize=12)
        ax.set_xlabel('Product Name', fontsize=10)
        ax.set_ylabel('Quantity Sold', fontsize=10)

        # Rotate x-tick labels for better visibility
        plt.xticks(rotation=45, ha='right')

        # Set a Seaborn whitegrid background
        sns.despine()

        # Render the plot in Streamlit
        st.pyplot(plt)

    def top_revenue_generated_cities_plot(self, top_n):
        top_cities = self.data.groupby('City')['Profit'].sum().sort_values(ascending=False).head(top_n)

        # Create a DataFrame for the top cities
        top_cities_df = pd.DataFrame(top_cities).reset_index()

        # Create a Plotly line plot
        fig = px.line(
            top_cities_df,
            x='City',
            y='Profit',
            markers=True,  # This adds markers to the line
            title='Top Revenue Generating Cities vs Profit',
            line_shape='linear'  # Can also be 'spline' for smoother curves
        )

        # Customize the layout and appearance
        fig.update_traces(line=dict(color='#664343', width=3, dash='dashdot'))  # Custom line color and style
        fig.update_layout(
            title={
                'text': 'Top Revenue Generating Cities vs Profit',
                'y': 0.9,  # Adjust the vertical alignment of the title (0 is the bottom, 1 is the top)
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Ensure the title is anchored in the center
                'yanchor': 'top'  # Align it from the top
            },
            title_font_size=16,
            title_font_color='#333',
            xaxis_title='City',
            yaxis_title='Profit',
            plot_bgcolor='#B7B7B7',  # Background color
            paper_bgcolor='white',  # Axes background color
        )

        # Rotate the x-axis labels for better readability
        fig.update_xaxes(tickangle=90)

        # Render the plot in Streamlit
        st.plotly_chart(fig)

    def golden_customer(self, top_n):
        """Return the top N customers with the highest total quantity purchased."""
        golden_customers = self.data.groupby(['Customer ID', 'Customer Name'])['Quantity'].sum().sort_values(
            ascending=False).head(top_n)
        return golden_customers
    def golden_customer_plot(self, top_n):

        # Group by Customer ID and Customer Name, sum the quantities
        golden_customers = self.data.groupby(['Customer ID', 'Customer Name'])['Quantity'].sum().sort_values(
            ascending=False).head(top_n)

        # Convert data to DataFrame
        n_df_2 = pd.DataFrame(golden_customers).reset_index(level='Customer ID')

        # Create figure and axes for donut chart
        fig, ax = plt.subplots(figsize=(4, 3), facecolor='#EDDFE0')

        # Plot pie chart with a hole in the center for the donut effect
        wedges, texts, autotexts = ax.pie(n_df_2['Quantity'], labels=n_df_2.index, autopct='%0.1f%%',
                                          explode=[0.1] + [0] * (top_n - 1), shadow=True, startangle=90,
                                          textprops={'fontsize': 8}, colors=sns.color_palette("pastel"))

        # Create a white circle in the middle for the donut effect
        center_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(center_circle)

        # Customize the title
        ax.set_title('Golden Customers & Purchase Rates', fontsize=12)

        # Display the plot in Streamlit
        st.pyplot(fig)

    def next_investment_sector(self, top_n):
        categories = self.data.groupby('Category')['Profit'].sum().sort_values(ascending=False).head(top_n)
        return categories

    def next_investment_sector_plot(self, top_n):
        category = self.data.groupby('Category')['Profit'].sum().sort_values(ascending=False).head(top_n)

        # Convert the data into a DataFrame
        n_df_2 = pd.DataFrame(category).reset_index()

        # Create a figure using Seaborn
        plt.figure(figsize=(10, 5))  # Adjust the figure size if necessary

        # Set Seaborn theme for a clean background
        sns.set_theme(style="whitegrid")

        # Create horizontal bar plot
        ax = sns.barplot(x='Profit', y='Category', data=n_df_2, palette='crest')

        # Customize plot appearance
        ax.set_title('Next Investment Sector and Generated Profits', fontsize=12)
        ax.set_xlabel('Generated Profits', fontsize=10)
        ax.set_ylabel('Investment Sectors', fontsize=10)

        # Set the background color for axes (if you want to retain that)
        ax.set_facecolor('#F5F5F7')

        # Invert the y-axis to show the highest bar on top
        #ax.invert_yaxis()

        # Render the plot in Streamlit
        st.pyplot(plt)
    def top_revenue_generated_cities(self, top_n):
        top_cities = self.data.groupby('City')['Profit'].sum().sort_values(ascending=False).head(top_n)
        return top_cities

    def top_sold_product_without_discount(self, top_n):
        prod_without_discount = self.data[self.data['Discount'] != 0.0]
        top_prods = prod_without_discount.groupby('Product Name')['Quantity'].sum().sort_values(ascending=False).head(
            top_n)
        return top_prods

    def top_sold_product_after_discount(self, top_n):
        prod_with_discount = self.data[self.data['Discount'] == 0.0]
        top_prods = prod_with_discount.groupby('Product Name')['Quantity'].sum().sort_values(ascending=False).head(
            top_n)
        return top_prods

    def prophet_sales_forecast(self, days):
        # Prepare the data for Prophet
        self.data['Order Date'] = pd.to_datetime(self.data['Order Date'])
        new_data = self.data.groupby('Order Date')['Sales'].sum().sort_index(ascending=False).reset_index()

        # Prepare the data for Prophet
        new_data.columns = ['ds', 'y']

        # Fit the Prophet model
        m = Prophet()
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model = m.fit(new_data)

        # Make future predictions
        future = m.make_future_dataframe(periods=days, freq='D')
        forecast = m.predict(future)

        # Extract forecast data and aggregate it by week
        forecast['ds'] = pd.to_datetime(forecast['ds'])  # Convert 'ds' column to datetime
        forecast.set_index('ds', inplace=True)

        # Aggregate forecast by week
        forecast_weekly = forecast['yhat'].resample('W').sum()

        # Separate historical and predicted data
        last_date = new_data['ds'].max()  # Last date of the historical data

        # Historical forecasted sales (before last date in original data)
        historical_forecast = forecast_weekly[forecast_weekly.index <= last_date]

        # Predicted future sales (after last date in original data)
        predicted_forecast = forecast_weekly[forecast_weekly.index > last_date]

        # Prepare data for Plotly Express
        historical_data = pd.DataFrame(
            {'Date': historical_forecast.index, 'Sales': historical_forecast.values, 'Type': 'Historical Data'})
        predicted_data = pd.DataFrame(
            {'Date': predicted_forecast.index, 'Sales': predicted_forecast.values, 'Type': 'Predicted Data'})

        plot_data = pd.concat([historical_data, predicted_data])

        # Create bar chart using Plotly Express
        fig = px.bar(plot_data, x='Date', y='Sales', color='Type',
                     title='Forecasted Sales (Historical vs Predicted)',
                     labels={'Date': 'Date', 'Sales': 'Sales'},
                     color_discrete_map={'Historical Data': '#629584', 'Predicted Data': '#387478'})

        # Update the layout for better visualization
        fig.update_layout(
            title={
                'text': 'Sales Forecast',
                'y': 0.9,  # Adjust the vertical alignment of the title (0 is the bottom, 1 is the top)
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Ensure the title is anchored in the center
                'yanchor': 'top'  # Align it from the top
            },
            plot_bgcolor='#EDDFE0',
            paper_bgcolor='white',
            xaxis_tickangle=45,
            font=dict(size=10),
            bargap=0.1  # Gap between bars
        )

        # Show the plot using Streamlit
        st.plotly_chart(fig)
    def prophet_profit_forecast(self, days):
        # Prepare the data for Prophet
        self.data['Order Date'] = pd.to_datetime(self.data['Order Date'])
        new_data = self.data.groupby('Order Date')['Profit'].sum().sort_index(ascending=False).reset_index()

        # Prepare the data for Prophet
        new_data.columns = ['ds', 'y']

        # Fit the Prophet model
        m = Prophet()
        m.add_seasonality(name='weekly', period=7, fourier_order=5)
        model = m.fit(new_data)

        # Make future predictions
        future = m.make_future_dataframe(periods=days, freq='D')
        forecast = m.predict(future)

        fig = m.plot(forecast)
        fig.set_size_inches(6, 2)
        # m.plot_components(forecast)
        # plt.show()
        st.pyplot(fig)
    def sales_anomaly_detection(self, days):
        self.data['Order Date'] = pd.to_datetime(self.data['Order Date'])
        new_data = self.data.groupby('Order Date')['Sales'].sum().sort_index(ascending=False).reset_index()

        # Prepare the data for Prophet
        new_data.columns = ['ds', 'y']

        # Fit the Prophet model
        m = Prophet()
        m.add_seasonality(name='weekly', period=7, fourier_order=5)
        model = m.fit(new_data)

        # Make future predictions
        future = m.make_future_dataframe(periods=days, freq='D')
        forecast = m.predict(future)

        # Extract forecast data and keep 'ds' and 'yhat' columns
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        df_forecast = pd.DataFrame(forecast)
        df_forecast_1 = df_forecast[['ds', 'yhat']]

        # Filter out historical data (keep only forecasted rows)
        forecasted_data = df_forecast_1[df_forecast_1['ds'] > new_data['ds'].max()].copy()

        # Use IsolationForest to detect anomalies in forecasted data only
        clf = IsolationForest(contamination=0.01, random_state=42)
        clf.fit(forecasted_data['yhat'].values.reshape(-1, 1))

        # Predict anomalies (outliers)
        predictions = clf.predict(forecasted_data['yhat'].values.reshape(-1, 1))

        # Add the predictions to the dataframe
        forecasted_data['anomaly'] = predictions

        # Extract the anomalous dates and sales forecast
        anomalous_data = forecasted_data[forecasted_data['anomaly'] < 0].copy()
        mean_yhat = forecasted_data['yhat'].mean()

        # Classify anomalies as 'spike' or 'drop' based on yhat values
        anomalous_data['type'] = np.where(anomalous_data['yhat'] > mean_yhat, 'spike', 'drop')

        # Display results in Streamlit
        st.markdown(
            '<p style="color:black;font-size:22px;"><strong>Dear User,</strong></p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="color:black;font-size:22px;"><strong>Based on our analysis and the sales forecasting model, we’ve detected a few anomalies in the predicted sales for the upcoming period. Here are the dates where these anomalies are expected:</strong></p>',
            unsafe_allow_html=True)

        for index, row in anomalous_data.iterrows():
            anomaly_type = "spike" if row['type'] == 'spike' else "drop"
            st.write(f"**On {row['ds'].date()}, there is a significant {anomaly_type} in sales.**")

        # Plot the sales data and highlight anomalies using Plotly Express
        forecasted_data['anomaly_label'] = np.where(forecasted_data['anomaly'] < 0, 'Anomaly', 'Forecast')

        # Create the line plot
        fig = px.line(forecasted_data, x='ds', y='yhat', title='Sales Forecast with Anomalies Detected',
                      labels={'ds': 'Date', 'yhat': 'Sales Forecast'},
                      color_discrete_sequence=['#705C53'])

        # Add the anomalies as scatter points
        fig.add_scatter(x=anomalous_data['ds'], y=anomalous_data['yhat'], mode='markers', name='Anomalies',
                        marker=dict(color='#FD8B51', size=10))

        # Update the layout for background colors and format
        fig.update_layout(
            title={
                'text': 'Sales Forecast with Anomaly Detection',
                'y': 0.9,  # Adjust the vertical alignment of the title (0 is the bottom, 1 is the top)
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',  # Ensure the title is anchored in the center
                'yanchor': 'top'  # Align it from the top
            },
            plot_bgcolor='#F5F5F7',
            paper_bgcolor='white',
            xaxis_title='Date',
            yaxis_title='Sales',
            font=dict(size=10)
        )

        # Display the plot using Streamlit
        st.plotly_chart(fig)

        # Suggestions
        #st.write('**Suggestions:**')
        st.markdown(
            '<p style="color:black;font-size:22px;"><strong>Suggestions</strong></p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="color:black;font-size:22px;"><strong>1. For the predicted drops in sales, we recommend reviewing your marketing and promotional strategies during this period. Additionally, check for any operational issues that could impact performance.</strong></p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="color:black;font-size:22px;"><strong>2. For the expected spike, it would be beneficial to ensure adequate stock levels and prepare your sales channels to take advantage of this potential increase in demand.</strong></p>',
            unsafe_allow_html=True)

        st.markdown(
            '<p style="color:black;font-size:22px;"><strong>Best Regards.</strong></p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p style="color:black;font-size:22px;"><strong>Data Dynamo.</strong></p>',
            unsafe_allow_html=True)

    def button_create(self):
        st.write(['Order ID', 'Order Date',
                  'Customer ID', 'Customer Name', 'City',
                  'Product ID', 'Category', 'Sub-Category',
                  'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit'])


def main():
    st.set_page_config(page_title="Sales Dashboard", page_icon="📈", layout="wide")

    # Adding the background image via CSS
    img_path = "Background.jpg"  # Replace with your actual image name
    base64_img = get_base64_of_bin_file(img_path)

    st.markdown(
        f"""
        <style>
        .main {{
            background-color: #F5F5F7;  /* Set main background to black */
        }}
        [data-testid="stSidebar"] {{
            background-color: #B7B7B7;
        }}
        .stButton>button {{
            background-color: #705C53;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 12px;
            border: none;
            margin: 10px;
            cursor: pointer;

        }}
        .stButton>button:hover {{
            background-color: #705C53;
            color: #6df2b7;
         }}
        # input {{
        #     background-color: #2c2c2c;
        #     color: #f0f2f5;
        #     #padding: 12px;
        #     border-radius: 10px;
        #     border: 1px solid #444;
        #     width: 250px;
        # }}
        input {{
            background-color: #2c2c2c;
            color: #f0f2f5;
            border-radius: 10px;
            border: 1px solid #444;
            width: 30px;  /* Change this to adjust the input box length */
            margin-bottom: 5px;  /* Adjust this to reduce space between input boxes */
        }}
        h1 {{
            color: #705C53;
            text-align: left;
            font-family: "Times New Roman", Times, serif;
            font-size: 65px;
            font-weight: 600;
        }}
        h2, h3 {{
            color: #705C53;
        }}
        label {{
            color: #dadfe6;
        }}
        .no-background {{
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    conn = sqlite3.connect('user_db.sqlite')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT,
                    business_name TEXT,
                    email TEXT,
                    mobile_number TEXT)''')
    conn.commit()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        st.title("Data Dynamo")
        option = st.sidebar.selectbox("Choose an option", ["Sign Up", "Log In"])

        if option == "Sign Up":
            st.subheader("Sign Up")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            business_name = st.text_input("Business Name")
            email = st.text_input("Email")
            mobile_number = st.text_input("Mobile Number")

            if st.button("Sign Up"):
                if not (username and password and business_name and email and mobile_number):
                    st.error("All fields are required.")
                else:
                    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
                    c.execute("SELECT * FROM users WHERE username=?", (username,))
                    if c.fetchone():
                        st.error("Username already exists.")
                    else:
                        c.execute(
                            "INSERT INTO users (username, password, business_name, email, mobile_number) VALUES (?, ?, ?, ?, ?)",
                            (username, hashed_pw, business_name, email, mobile_number))
                        conn.commit()
                        st.success("Account created successfully!")

        elif option == "Log In":
            st.subheader("Log In")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            if st.button("Log In"):
                hashed_pw = hashlib.sha256(password.encode()).hexdigest()
                c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_pw))
                if c.fetchone():
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid username or password.")

    else:
        st.title("Data Dynamo's Dashboard")

        st.subheader("Upload CSV or Excel File")
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

        if uploaded_file is not None:
            # Handling CSV files
            if uploaded_file.type == "text/csv":
                encoding = detect_encoding(uploaded_file)
                data = pd.read_csv(uploaded_file, encoding=encoding)
            # Handling Excel files
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                data = pd.read_excel(uploaded_file)

            # Initialize the SalesData class with the uploaded data
            sales_data = SalesData(data)

            # Display sales data
            st.subheader("Sales Data")
            st.write(sales_data.show_data())

            # Sidebar for analysis options
            st.sidebar.title("Analyze Data")
            analysis_option = st.sidebar.radio("Select an analysis",
                                               ["Select an option", "Best Selling Product", "Golden Customer",
                                                "Next Investment Sector", "Top Revenue Generated City",
                                                "Top Sold Products Without Discount",
                                                "Top Sold Products With Discount"])

            # Handle each analysis option separately
            if analysis_option != "Select an option":
                if analysis_option == "Best Selling Product":
                    top_n = st.sidebar.number_input("How many top products?", min_value=1, max_value=30, value=5)
                    st.subheader(f"Top {top_n} Best Selling Products")
                    best_selling_prods = sales_data.best_selling_product(top_n)
                    st.write(best_selling_prods)
                    sales_data.best_selling_products_plot(top_n)

                elif analysis_option == "Golden Customer":
                    top_n = st.sidebar.number_input("How many top customers?", min_value=1, max_value=30, value=3)
                    st.subheader("Golden Customer")
                    golden_customers = sales_data.golden_customer(top_n)
                    st.write(golden_customers)
                    sales_data.golden_customer_plot(top_n)

                elif analysis_option == "Next Investment Sector":
                    top_n = st.sidebar.number_input("How many top categories?", min_value=1, max_value=30, value=3)
                    st.subheader("Next Investment Sector")
                    categories = sales_data.next_investment_sector(top_n)
                    st.write(categories)
                    sales_data.next_investment_sector_plot(top_n)

                elif analysis_option == "Top Revenue Generated City":
                    top_n = st.sidebar.number_input("How many top cities?", min_value=1, max_value=30, value=1)
                    st.subheader(f"Top {top_n} Revenue Cities")
                    top_cities = sales_data.top_revenue_generated_cities(top_n)
                    st.write(top_cities)
                    sales_data.top_revenue_generated_cities_plot(top_n)

                elif analysis_option == "Top Sold Products Without Discount":
                    top_n = st.sidebar.number_input("How many products?", min_value=1, max_value=30, value=1)
                    st.subheader("Top Sold Products Without Discount")
                    top_prods_no_discount = sales_data.top_sold_product_without_discount(top_n)
                    st.write(top_prods_no_discount)

                elif analysis_option == "Top Sold Products With Discount":
                    top_n = st.sidebar.number_input("How many products?", min_value=1, max_value=30, value=1)
                    st.subheader("Top Sold Products With Discount")
                    top_prods_with_discount = sales_data.top_sold_product_after_discount(top_n)
                    st.write(top_prods_with_discount)

            # sidebar for catarories analysis
            st.sidebar.title("Catagory Wise Analysis")
            option = st.sidebar.radio('Select an option for analysis:',
                                      ['Select an Option', 'Best Selling Product', 'Golden Customer',
                                       'Top Revenue Generated City'])

            if option != 'Select an Option':
                if option == 'Best Selling Product':
                    # Group data by category and product name
                    n_data = data.groupby(['Category', 'Product Name'])['Quantity'].sum().reset_index()
                    n_data = n_data.sort_values(['Category', 'Quantity'], ascending=[False, False])

                    # Display categories once "Best Selling Products" is selected
                    categories = n_data['Category'].unique()

                    # Streamlit UI for category selection
                    selected_category = st.selectbox('Select a category for analysis:', categories)

                    # Streamlit number input for selecting the number of products
                    num_products = st.sidebar.number_input('How many top products do you want to analyze?', min_value=1,
                                                           max_value=30, value=3)

                    # Filter and display the best-selling products based on the selected category
                    if selected_category:
                        n_data2 = n_data.groupby('Category')
                        x = n_data2.get_group(selected_category)
                        y = x.drop('Category', axis='columns')
                        y.set_index('Product Name', inplace=True)
                        best_sold_products = y.head(int(num_products))

                        # Display the results
                        st.subheader(f'Top {num_products} Best Selling Products in {selected_category} Category:')
                        st.dataframe(best_sold_products)
                        plt.figure(figsize=(12, 6))

                        # Use Seaborn's barplot to create a nicer visual
                        sns.set_theme(style="whitegrid")
                        ax = sns.barplot(x='Product Name', y='Quantity', data=best_sold_products, palette="crest")
                        # Customize the plot
                        ax.set_title('Best Selling Products vs Quantity Sold', fontsize=12)
                        ax.set_xlabel('Product Name', fontsize=10)
                        ax.set_ylabel('Quantity Sold', fontsize=10)

                        # Rotate x-tick labels for better visibility
                        plt.xticks(rotation=45, ha='right')

                        # Set a Seaborn whitegrid background
                        sns.despine()

                        # Render the plot in Streamlit
                        st.pyplot(plt)



                elif option == 'Golden Customer':
                    # Group data by category, customer ID, and customer name
                    n_data = data.groupby(['Category', 'Customer ID', 'Customer Name'])['Quantity'].sum().reset_index()
                    n_data = n_data.sort_values(['Category', 'Quantity'], ascending=[False, False])

                    # Get unique categories for the dropdown menu
                    categories = n_data['Category'].unique()

                    # Streamlit UI for category selection
                    selected_category = st.selectbox('Select a category for analysis:', categories)

                    # Streamlit number input for selecting the number of customers
                    num_customers = st.sidebar.number_input('How many top customers do you want to analyze?',
                                                            min_value=1, max_value=30, value=3)

                    # Filter and display the golden customers based on the selected category
                    if selected_category:
                        n_data2 = n_data.groupby('Category')
                        x = n_data2.get_group(selected_category)
                        y = x.drop('Category', axis='columns')

                        # Set Customer ID as the index
                        y.set_index('Customer ID', inplace=True)

                        # Get top N golden customers
                        golden_customers = y.head(num_customers)

                        # Display the results
                        st.subheader(f'Top {num_customers} Golden Customers in {selected_category} Category:')
                        st.dataframe(golden_customers)
                        fig, ax = plt.subplots(figsize=(4, 3), facecolor='#EDDFE0')

                        # Plot pie chart with a hole in the center for the donut effect
                        wedges, texts, autotexts = ax.pie(golden_customers['Quantity'], labels=golden_customers['Customer Name'], autopct='%0.1f%%',
                                                          explode=[0.1] + [0] * (num_customers - 1), shadow=True, startangle=90,
                                                          textprops={'fontsize': 8}, colors=sns.color_palette("pastel"))

                        # Create a white circle in the middle for the donut effect
                        center_circle = plt.Circle((0, 0), 0.70, fc='white')
                        fig.gca().add_artist(center_circle)

                        # Customize the title
                        ax.set_title('Golden Customers & Purchase Rates', fontsize=12)

                        # Display the plot in Streamlit
                        st.pyplot(fig)


                elif option == 'Top Revenue Generated City':
                    # Group data by category and city, summing the profit
                    n_data = data.groupby(['Category', 'City'])['Profit'].sum().sort_values(
                        ascending=False).reset_index()
                    n_data = n_data.sort_values(['Category', 'Profit'], ascending=[False, False])

                    # Get unique categories for the dropdown menu
                    categories = n_data['Category'].unique()

                    # Streamlit UI for category selection
                    selected_category = st.selectbox('Select a category for analysis:', categories)

                    # Streamlit number input for selecting the number of cities
                    num_cities = st.sidebar.number_input('How many top cities do you want to analyze?', min_value=1,
                                                         max_value=30, value=3)

                    # Filter and display the top revenue-generating cities based on the selected category
                    if selected_category:
                        n_data2 = n_data.groupby('Category')
                        x = n_data2.get_group(selected_category)
                        y = x.drop('Category', axis='columns')

                        # Set City as the index
                        y.set_index('City', inplace=True)

                        # Get top N cities generating revenue
                        top_cities = y.head(num_cities)

                        # Display the results
                        st.subheader(f'Top {num_cities} Revenue Generating Cities in {selected_category} Category:')
                        st.dataframe(top_cities)

                        fig = px.line(
                            top_cities,
                            x=top_cities.index,
                            y='Profit',
                            markers=True,  # This adds markers to the line
                            title='Top Revenue Generating Cities vs Profit',
                            line_shape='linear'  # Can also be 'spline' for smoother curves
                        )

                        # Customize the layout and appearance
                        fig.update_traces(
                            line=dict(color='#664343', width=3, dash='dashdot'))  # Custom line color and style
                        fig.update_layout(
                            title={
                                'text': 'Top Revenue Generating Cities vs Profit',
                                'y': 0.9,  # Adjust the vertical alignment of the title (0 is the bottom, 1 is the top)
                                'x': 0.5,  # Center the title horizontally
                                'xanchor': 'center',  # Ensure the title is anchored in the center
                                'yanchor': 'top'  # Align it from the top
                            },
                            title_font_size=16,
                            title_font_color='#333',
                            xaxis_title='City',
                            yaxis_title='Profit',
                            plot_bgcolor='#B7B7B7',  # Background color
                            paper_bgcolor='white',  # Axes background color
                        )

                        # Rotate the x-axis labels for better readability
                        fig.update_xaxes(tickangle=90)

                        # Render the plot in Streamlit
                        st.plotly_chart(fig)

            # Sidebar for prediction options
            st.sidebar.title("Predict Data")
            prediction_option = st.sidebar.radio("Select a prediction",
                                                 ["Select an option", "Future Sales Prediction",
                                                  "Profit Forecast", "Anomalies Forecast"])

            # Handle each prediction option separately
            if prediction_option != "Select an option":
                if prediction_option == "Future Sales Prediction":
                    days = st.sidebar.number_input("How many days do you want to predict?", min_value=1,
                                                   max_value=20000, value=365)
                    st.subheader("Future Sales Prediction")
                    predicted = sales_data.prophet_sales_forecast(int(days))
                    st.write(predicted)


                elif prediction_option == "Profit Forecast":
                    days = st.sidebar.number_input("How many days do you want to predict?", min_value=1,
                                                   max_value=20000, value=365)
                    st.subheader("Future Profit Prediction")
                    predicted_2 = sales_data.prophet_profit_forecast(int(days))
                    st.write(predicted_2)

                elif prediction_option == "Anomalies Forecast":
                    days = st.sidebar.number_input("How many days do you want to predict?", min_value=1,
                                                   max_value=20000, value=365)
                    st.subheader("Future Anomaly Prediction")
                    predicted_3 = sales_data.sales_anomaly_detection(int(days))
                    st.write(predicted_3)

            st.sidebar.title("Profit Calculator")
            option1 = st.sidebar.radio('Select an option for analysis:', ['Select an Option', 'Profit Calculation'])

            if option1 != 'Select an Option':
                if option1 == 'Profit Calculation':
                    selected_category = st.selectbox('Select a duration for analysis:', ['Yearly', 'Monthly'])
                    if selected_category == 'Yearly':
                        data1=data
                        data1['Order Date'] = pd.to_datetime(data1['Order Date'], infer_datetime_format=True)

                        # Extract the year from 'Order Date'
                        data1['Year'] = data1['Order Date'].dt.year

                        # Streamlit calendar widget to select a date (default is today)
                        selected_date = st.date_input("Select a date (Only Year will be used)", datetime.date.today())

                        # Extract the year from the selected date
                        selected_year = selected_date.year

                        # Filter the data for the selected year
                        filtered_data = data1[data1['Year'] == selected_year]

                        # Calculate total profit for the selected year
                        total_profit = filtered_data['Profit'].sum()

                        # Display the result
                        st.write(f"Total Profit for the year {selected_year}: ${total_profit:.2f}")

                    elif selected_category == 'Monthly':

                        # Convert 'Order Date' to datetime
                        data1=data
                        data1['Order Date'] = pd.to_datetime(data1['Order Date'], infer_datetime_format=True)
                        # Extract year and month from 'Order Date'
                        data1['Year-Month'] = data1['Order Date'].dt.to_period('M')

                        # Streamlit calendar widget to select a date
                        selected_date = st.date_input("Select a date (Only Month will be used)", datetime.date.today())

                        # Extract the year and month from the selected date
                        selected_year_month = selected_date.strftime('%Y-%m')

                        # Filter data for the selected month and year
                        filtered_data = data1[data1['Year-Month'] == selected_year_month]

                        # Calculate total profit for the selected month and year
                        total_profit = filtered_data['Profit'].sum()

                        # Display the result
                        st.write(f"Total Profit for {selected_year_month}: ${total_profit:.2f}")

            if st.sidebar.button('Help'):
                st.write("Must Required Columns in the dataset:")
                help_me = sales_data.button_create()
                st.write(help_me)

        if st.sidebar.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.stop()


    conn.close()


if __name__ == "__main__":
    main()