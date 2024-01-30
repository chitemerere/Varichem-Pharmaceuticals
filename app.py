#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from io import BytesIO
from datetime import datetime
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime as dt
from kneed import KneeLocator
import warnings
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk
import squarify

# Download NLTK vader_lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize the VADER SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# Function to save plot
def save_plot(fig, filename):
    """Saves a matplotlib figure as an image file."""
    fig.savefig(filename)
    return filename

# Function to plot data
def plot_data(selected_towns, data):
    if selected_towns:
        filtered_data = data[data['TOWN'].isin(selected_towns)]
    else:
        # If no towns are selected, display data for all towns
        filtered_data = data

    town_dispensing = filtered_data.groupby('TOWN').sum(numeric_only=True)
    total_dispensed_per_town = town_dispensing.sum(axis=1).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    total_dispensed_per_town.plot(kind='bar', color='teal')
    plt.title('Total Quantity Sold by Selected Towns')
    plt.xlabel('Town')
    plt.ylabel('Total Quantity Sold')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

# Function to plot data
def plot_product_distribution_by_towns(selected_products, selected_towns, data):
    for selected_product in selected_products:
        # Filtering data based on the selected product and towns
        filtered_data = data[data['DISCRIPTION'] == selected_product]
        if selected_towns:
            filtered_data = filtered_data[filtered_data['TOWN'].isin(selected_towns)]

        town_distribution = filtered_data.groupby('TOWN').sum().sum(axis=1).sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        town_distribution.plot(kind='bar', color='teal')
        plt.title(f'Distribution of {selected_product} in Selected Towns')
        plt.xlabel('Town')
        plt.ylabel('Total Quantity Sold')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)
        
# Define a function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Attempt to convert 'Invoice Date' to datetime format
        try:
            data['Invoice Date'] = pd.to_datetime(data['Invoice Date'], format='%d/%m/%Y')
        except Exception as e:
            st.error(f"Error in converting 'Invoice Date' to datetime: {e}")
            return None
        return data
    else:
        return None
    
def cluster_segments (rfm):
    if(rfm.RFM_SCORE>=450):
        return 'High Value Customers'
    if(rfm.RFM_SCORE>=340 and rfm.RFM_SCORE<450):
        return 'Potential Loyalist'
    if(rfm.RFM_SCORE>=280 and rfm.RFM_SCORE<340):
        return 'At Risk'
    else:
        return 'Sleelpig Customers'
    
# Function to calculate NPS
def calculate_nps(scores):
    promoters = len([score for score in scores if score >= 9])
    detractors = len([score for score in scores if score <= 6])
    total_responses = len(scores)

    nps = ((promoters - detractors) / total_responses) * 100
    return nps

# Streamlit application layout
st.image("logo.png", width=200)
st.markdown("<h1 style='font-size:30px;'>Varichem Pharmaceuticals Sales Analysis Dashboard</h1>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select an Analysis:', 
                           ['Trend Analysis','Geographical Analysis','Product Performance', 'Pharmacy Performance', 'Alerts','Sales Forecasting', 'Market Segmentation', 'Sentiment Analysis', 'Net Promoter Score'])

# Password input
password_guess = st.text_input('What is the Password?', type ="password").strip()

# Check if password is entered and incorrect
if password_guess and password_guess != st.secrets["password"]:
    st.error("Incorrect password. Please try again.")
    st.stop()

# Proceed only if the password is correct
if password_guess == st.secrets["password"]:
    st.success("Password is correct")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if options == 'Trend Analysis':
            st.header('Trend Analysis')
            st.subheader('Trend Anaysis Total Units')
            # Include your product performance analysis code here
            # Preparing data for trend analysis
            # Summing up the quantity for each month across all products and pharmacies

            # User input for product selection using multiselect
            products = sorted(data['DISCRIPTION'].unique().tolist())
            selected_products = st.multiselect('Select Products (leave blank for all products)', products)

            # Sorting the selected products in ascending order
            selected_products.sort()

            # Filtering data and plotting trends based on product selection
            if selected_products:
                for product in selected_products:
                    # Filtering data for the selected product
                    filtered_data = data[data['DISCRIPTION'] == product]

                    # Preparing data for trend analysis
                    monthly_totals = filtered_data.iloc[:, 5:].sum()

                    # Plotting the trend for each selected product
                    fig, ax = plt.subplots(figsize=(12, 6))
                    monthly_totals.plot(kind='bar', color='skyblue', ax=ax)
                    plt.title(f'Monthly Sales Trend for {product} (Nov 2022 - Nov 2023)')
                    plt.xlabel('Month')
                    plt.ylabel('Total Quantity Sold')
                    plt.xticks(rotation=45)
                    plt.grid(axis='y')

                    # Display each plot in Streamlit
                    st.pyplot(fig)

                    # Save the plot to a BytesIO object and create a download button
                    buffer = BytesIO()
                    fig.savefig(buffer, format='png')
                    buffer.seek(0)
                    st.download_button(
                        label=f"Download {product} Trend Image",
                        data=buffer,
                        file_name=f"{product}_trend.png",
                        mime="image/png"
                    )
            else:
                # Preparing data for trend analysis for all products
                monthly_totals = data.iloc[:, 5:].sum()

                # Plotting the trend for all products
                fig, ax = plt.subplots(figsize=(12, 6))
                monthly_totals.plot(kind='bar', color='skyblue', ax=ax)
                plt.title('Monthly Sales Trend for All Products (Nov 2022 - Nov 2023)')
                plt.xlabel('Month')
                plt.ylabel('Total Quantity Sold')
                plt.xticks(rotation=45)
                plt.grid(axis='y')

                # Display the plot in Streamlit
                st.pyplot(fig)

                # Save the plot to a BytesIO object and create a download button
                buffer = BytesIO()
                fig.savefig(buffer, format='png')
                buffer.seek(0)
                st.download_button(
                    label="Download All Products Trend Image",
                    data=buffer,
                    file_name="all_products_trend.png",
                    mime="image/png"
                )


            # Analyzing the trend for each product brand (DISCRIPTION)

            # Grouping the data by 'DISCRIPTION' and summing up the quantities for each month
            product_trends = data.groupby('DISCRIPTION').sum(numeric_only=True)

            # Transposing the dataframe for easier plotting
            product_trends_transposed = product_trends.T

            # Plotting the trends for the top 6 products based on total quantity dispensed
            st.title("Product Sales Trend Analysis")
            st.subheader('Trend Analysis for Selected Products')

            # User input for selecting products using multiselect
            all_products = product_trends.index.tolist()
            selected_products = st.multiselect('Select Products (leave blank for top 6 products)', all_products)

            # Create a matplotlib figure
            fig, ax = plt.subplots(figsize=(15, 8))

            # Determine the products to plot
            if not selected_products:
                st.subheader('Trend Analysis for Top 6 Products')
                products_to_plot = product_trends.sum(axis=1).nlargest(6).index
                plot_title = 'Monthly Sales Trend for Top 6 Products (Nov 2022 - Nov 2023)'
            else:
                products_to_plot = selected_products
                plot_title = 'Monthly Sales Trend for Selected Products (Nov 2022 - Nov 2023)'

            # Plotting the trend
            for product in products_to_plot:
                product_trends_transposed[product].plot(kind='line', label=product, ax=ax)
            plt.title(plot_title)
            plt.xlabel('Month')
            plt.ylabel('Quantity Sold')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Save the plot to a BytesIO object and create a download button
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Trend Image",
                data=buffer,
                file_name="trend_analysis.png",
                mime="image/png"
            )


            # User input for number of top products
            st.subheader('Trend Anaysis Top N Products')

            num_products = st.number_input('Select number of top products to analyze', min_value=1, value=5, step=1)

            # Plotting the trends for the top N products based on total quantity dispensed
            top_products = product_trends.sum(axis=1).nlargest(num_products).index
            fig, ax = plt.subplots(figsize=(15, 8))
            for product in top_products:
                product_trends_transposed[product].plot(kind='line', label=product, ax=ax)
            plt.title(f'Monthly Sales Trend for Top {num_products} Products (Nov 2022 - Nov 2023)')
            plt.xlabel('Month')
            plt.ylabel('Quantity Sold')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.grid(axis='y')

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Save the plot to a BytesIO object and create a download button
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Trend Image",
                data=buffer,
                file_name=f"Top_{num_products}_Products_Trend.png",
                mime="image/png"
            )


            # User input for number of product/products
            st.subheader('Trend Analysis for Selected Product(s)')

            # User input for selecting products
            all_products = product_trends_transposed.columns.tolist()
            selected_products = st.multiselect('Select Products', all_products, default=all_products[0])

            # Create a matplotlib figure for the plot
            fig, ax = plt.subplots(figsize=(15, 8))

            # Plotting the trends for the selected products
            for product in selected_products:
                product_trends_transposed[product].plot(kind='line', label=product, ax=ax)
            plt.title('Monthly Sales Trend for Selected Products (Nov 2022 - Nov 2023)')
            plt.xlabel('Month')
            plt.ylabel('Quantity Sold')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.grid(axis='y')

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Save the plot to a BytesIO object and create a download button
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download Trend Image",
                data=buffer,
                file_name="selected_products_trend.png",
                mime="image/png"
            )

        elif options == 'Geographical Analysis':
            st.header('Geographical Analysis')
            st.subheader('Overall Unit Sales by Town')

            # Convert all town names to strings and filter out any NaN or missing values
            towns = data['TOWN'].dropna().astype(str).unique()
            towns = sorted(towns)
            selected_towns = st.multiselect('Select Towns', towns)

            # Display plot
            plot_data(selected_towns, data)

            # Town and Product selection
            st.subheader('Unit Sales by Town and Product')
            towns = data['TOWN'].dropna().astype(str).unique()
            towns = sorted(towns)
            products = data['DISCRIPTION'].dropna().astype(str).unique()
            products = sorted(products)
            selected_towns = st.multiselect('Select Towns', towns, key='select_towns')
            selected_products = st.multiselect('Select Products', products, key='select_products')

            # Display plot
            if selected_products:
                plot_product_distribution_by_towns(selected_products, selected_towns, data)
            else:
                st.write("Please select one or more products to view their distribution.")

        elif options == 'Product Performance':
            st.header('Product Performance')

            # Recreating the total_dispensed_per_town variable
            st.header('Total sold per town')

            # User input for number of top towns and products
            num_towns = st.number_input('Select number of top towns to analyze', min_value=1, value=10, step=1)
            num_products = st.number_input('Select number of top products to analyze', min_value=1, value=10, step=1)

            # Grouping and summarizing data
            town_dispensing = data.groupby('TOWN').sum(numeric_only=True)
            product_trends = data.groupby('DISCRIPTION').sum(numeric_only=True)
            total_dispensed_per_town = town_dispensing.sum(axis=1, numeric_only=True).sort_values(ascending=False)

            # Selecting top N towns and products
            top_towns = total_dispensed_per_town.nlargest(num_towns).index
            top_products = product_trends.sum(axis=1, numeric_only=True).nlargest(num_products).index

            # Filtering and aggregating data
            filtered_data = data[data['TOWN'].isin(top_towns) & data['DISCRIPTION'].isin(top_products)]
            town_brand_aggregated = filtered_data.groupby(['TOWN', 'DISCRIPTION']).sum(numeric_only=True).reset_index()

            # Creating summary tables for each town and saving them as CSV
            for town in top_towns:
                town_data = town_brand_aggregated[town_brand_aggregated['TOWN'] == town]

                # Adjusting the slicing to include all relevant monthly columns
                monthly_columns = town_data.columns[0:]  # Modify this according to your DataFrame
                summary = town_data.pivot(index='DISCRIPTION', columns='TOWN', values=monthly_columns.tolist()).fillna(0)

                # Displaying the summary table for the town
                st.subheader(f"Summary for {town}:")
                st.table(summary)

                # Converting the DataFrame to CSV and encoding to bytes
                csv = summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download data for {town}",
                    data=csv,
                    file_name=f"{town}_summary.csv",
                    mime='text/csv'
                )

            # Calculating the total quantity dispensed for each product
            total_quantity_by_product = data.groupby('DISCRIPTION').sum(numeric_only=True).sum(axis=1).sort_values(ascending=False)
            st.subheader("Total Quantity Sold for Each Product:")
            st.table(total_quantity_by_product.head(10))  # Displaying top 10

            # Identifying the top 10 products based on total quantity dispensed
            top_10_products = total_quantity_by_product.nlargest(10)

            # Analyzing monthly trends for these top 10 products
            monthly_trends_top_10 = data[data['DISCRIPTION'].isin(top_10_products.index)].groupby('DISCRIPTION').sum(numeric_only=True)
            st.subheader("Monthly Trends for Top 10 Products:")
            st.table(monthly_trends_top_10)

            # Converting the DataFrame to CSV and encoding to bytes
            csv = monthly_trends_top_10.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Download Monthly Trends CSV",
                data=csv,
                file_name="monthly_trends_top_10.csv",
                mime='text/csv'
            )

            # Analyzing distribution across towns for these top 10 products
            distribution_across_towns_top_10 = data[data['DISCRIPTION'].isin(top_10_products.index)].groupby(['DISCRIPTION', 'TOWN']).sum(numeric_only=True).sum(axis=1).unstack(fill_value=0)
            st.subheader("Distribution Across Towns for Top 10 Products:")
            st.table(distribution_across_towns_top_10)

                # Converting the DataFrame to CSV and encoding to bytes
            csv = distribution_across_towns_top_10.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Download Distribution CSV",
                data=csv,
                file_name="distribution_across_towns_top_10.csv",
                mime='text/csv'
            )

            # Product Performance by Pharmacy filtered by month-year

            # Melting the dataset to convert it from wide format to long format
            # This helps in plotting time series data more easily
            # Unique months in the dataset for the dropdown
            # Melting the DataFrame
            df_melted = data.melt(id_vars=['C-CODE', 'NAME', 'TOWN', 'P-CODE', 'DISCRIPTION'], 
                                var_name='Month', value_name='Sales')

            st.subheader('Top N Product Performance by Pharmacy')

            months = df_melted['Month'].unique()
            selected_month = st.selectbox('Select a Month', months)

            # User input for number of top products
            num_products = st.number_input('Select number of top products', min_value=1, value=5, step=1)

            # Filtering the data based on the selected month
            filtered_data = df_melted[df_melted['Month'] == selected_month]

            # Aggregating sales data by 'NAME' and 'DISCRIPTION'
            aggregated_data = filtered_data.groupby(['NAME', 'DISCRIPTION'])['Sales'].sum().reset_index()

            # Sorting by sales and getting the top N products
            top_n_products = aggregated_data.sort_values(by='Sales', ascending=False).head(num_products)

            # Displaying the aggregated data
            st.table(top_n_products)

            # Converting the DataFrame to CSV and encoding to bytes
            csv = top_n_products.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Top N Products CSV",
                data=csv,
                file_name="top_n_products.csv",
                mime='text/csv'
            )


            # Product Returns
            # User input for the number of top return alerts
            st.subheader("Top Product Returns")

            # User input for town selection
            towns = data['TOWN'].astype(str).unique().tolist()  # Convert all entries to strings
            towns.sort()  # Sort the towns list
            towns.insert(0, 'All Towns')  # Adding an option to select all towns
            selected_town = st.selectbox('Select Town (or choose "All Towns" for all)', towns)

            # Filtering data by selected town if not 'All Towns'
            if selected_town != 'All Towns':
                data = data[data['TOWN'] == selected_town]

            num_alerts = st.number_input('Select number of top return alerts to display', min_value=1, value=20, step=1)

            # Common function to generate alerts
            def generate_alerts(data):
                alerts = []
                grouped_data = data.groupby(['NAME', 'DISCRIPTION'])

                for (pharmacy, product), group in grouped_data:
                    monthly_sales = group.iloc[:, 5:]  # includes monthly last column
                    last_month_returns = monthly_sales.iloc[:, -1]  # Focusing on the last month

                    if last_month_returns.values[0] < 0:
                        returns = abs(last_month_returns.values[0])
                        alert_message = f"Alert: {product} at {pharmacy} - Returns last month: {returns} units"
                        alerts.append((alert_message, pharmacy, product, returns))

                return alerts

            alerts = generate_alerts(data)

            # Handling no alerts case
            if not alerts:
                st.subheader(f"Top {num_alerts} Product Returns Last Month Alerts")
                st.write("No return for the selected period and/or town")
            else:
                # Displaying top N alerts for narration
                st.subheader(f"Top {num_alerts} Product Returns Last Month Alerts")
                for alert in sorted(alerts, key=lambda x: x[3], reverse=True)[:num_alerts]:
                    st.write(alert[0])  # Displaying the alert message

                # Creating and displaying DataFrame
                alerts_df = pd.DataFrame(alerts, columns=["Alert", "Pharmacy", "Product", "Returns"])
                top_alerts_df = alerts_df.sort_values(by='Returns', ascending=False).head(num_alerts)
                st.subheader(f"Top {num_alerts} Product Returns Last Month Alerts Table")
                st.table(top_alerts_df.drop(columns="Alert"))

                # Converting the DataFrame to CSV and encoding to bytes
                csv = top_alerts_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Top Product Returns CSV",
                    data=csv,
                    file_name="top_product_returns_last_month.csv",
                    mime='text/csv'
                )

            # Repeat the adjusted logic for the month before last month section
            # ... [Repeat the logic for the month before last month section, similar to the last month section] ...
            # Common function to generate alerts
            def generate_alerts(data):
                alerts = []
                grouped_data = data.groupby(['NAME', 'DISCRIPTION'])

                for (pharmacy, product), group in grouped_data:
                    monthly_sales = group.iloc[:, 5:]  # includes monthly last column
                    last_month_returns = monthly_sales.iloc[:, -2]  # Focusing on the month beforelast month

                    if last_month_returns.values[0] < 0:
                        returns = abs(last_month_returns.values[0])
                        alert_message = f"Alert: {product} at {pharmacy} - Returns month before last month: {returns} units"
                        alerts.append((alert_message, pharmacy, product, returns))

                return alerts

            alerts = generate_alerts(data)

            # Handling no alerts case
            if not alerts:
                st.subheader(f"Top {num_alerts} Product Returns For Month Before Last Month Alerts")           
                st.write("No return for the selected period and/or town")
            else:
                # Displaying top N alerts for narration
                st.subheader(f"Top {num_alerts} Product Returns Month Before Last Month Alerts")
                for alert in sorted(alerts, key=lambda x: x[3], reverse=True)[:num_alerts]:
                    st.write(alert[0])  # Displaying the alert message

                # Creating and displaying DataFrame
                alerts_df = pd.DataFrame(alerts, columns=["Alert", "Pharmacy", "Product", "Returns"])
                top_alerts_df = alerts_df.sort_values(by='Returns', ascending=False).head(num_alerts)
                st.subheader(f"Top {num_alerts} Product Returns Month Before Last Month Alerts Table")
                st.table(top_alerts_df.drop(columns="Alert"))

                # Converting the DataFrame to CSV and encoding to bytes
                csv = top_alerts_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Top Product Returns CSV",
                    data=csv,
                    file_name="top_product_returns_month_before_last_month.csv",
                    mime='text/csv'
                )

        elif options == 'Pharmacy Performance':
            st.header('Pharmacy Performance')

            # Let the user select the number of top pharmacies to analyze
            num_pharmacies = st.number_input('Select number of top pharmacies to analyze', min_value=1, value=10, step=1)

            # Extract unique product descriptions and convert them to a list
            unique_products = data['DISCRIPTION'].unique().tolist()

            # Sort the list of unique products
            unique_products.sort()

            # Insert 'All Products' at the beginning of the list
            unique_products.insert(0, 'All Products')

            # Use the sorted list in the select box
            selected_product = st.selectbox('Select a Product (or choose "All Products" for all)', unique_products)

            if selected_product != 'All Products':
                data = data[data['DISCRIPTION'] == selected_product]

            # Analysis code
            pharmacy_performance = data.groupby('NAME').sum(numeric_only=True)
            total_dispensed_by_pharmacy = pharmacy_performance.sum(axis=1).sort_values(ascending=False)
            top_n_pharmacies = total_dispensed_by_pharmacy.nlargest(num_pharmacies)
            monthly_trends_top_n_pharmacies = data[data['NAME'].isin(top_n_pharmacies.index)].groupby('NAME').sum(numeric_only=True)

            st.subheader(f'Top {num_pharmacies} Pharmacies by Unit Sales for {selected_product}')
            st.table(top_n_pharmacies)

            # CSV Download for Unit Sales
            csv_unit_sales = top_n_pharmacies.to_csv().encode('utf-8')
            st.download_button(
                label="Download Top Pharmacies by Unit Sales",
                data=csv_unit_sales,
                file_name="top_pharmacies_unit_sales.csv",
                mime='text/csv'
            )

            st.subheader(f"Top {num_pharmacies} Monthly Trends by Pharmacy for {selected_product}")
            st.table(monthly_trends_top_n_pharmacies)

            # Month Trends for top pharmacies
            csv_monthly_trends = monthly_trends_top_n_pharmacies.to_csv().encode('utf-8')
            st.download_button(
                label="Download Monthly Trends for Top Pharmacies",
                data=csv_monthly_trends,
                file_name="monthly_trends_top_pharmacies.csv",
                mime='text/csv'
            )

            # Top N Pharmacies Data
            top_n_pharmacies_data = data[data['NAME'].isin(top_n_pharmacies.index)]
            top_n_pharmacies_product_performance = top_n_pharmacies_data.groupby(['NAME', 'DISCRIPTION']).sum(numeric_only=True).sum(axis=1).unstack(fill_value=0)

            st.subheader(f'Top {num_pharmacies} Pharmacies by Product')
            st.table(top_n_pharmacies_product_performance)

            # CSV Download for Pharmacies by Product
            csv_pharmacies_product = top_n_pharmacies_product_performance.to_csv().encode('utf-8')
            st.download_button(
                label="Download Top Pharmacies by Product",
                data=csv_pharmacies_product,
                file_name="top_pharmacies_by_product.csv",
                mime='text/csv'
            )

            # Pharmacies with at least a product sales
            # Function to process data
            months = data.columns[5:]
            pharmacies_with_sales = data.groupby('NAME').filter(lambda x: all(x[month].sum() > 0 for month in months))


            # Calculate total and monthly sales for each of these pharmacies
            total_and_monthly_sales = pharmacies_with_sales.groupby('NAME')[months].sum()
            total_and_monthly_sales['Total Sales'] = total_and_monthly_sales.sum(axis=1)

             # User input for top N pharmacies
            top_n = st.number_input('Enter number of Top Pharmacies to display', min_value=1, value=5)

            # Filter by Top N pharmacies based on total sales within the eligible pharmacies
            top_pharmacies = total_and_monthly_sales.sort_values(by='Total Sales', ascending=False).head(top_n)

            # Display the top pharmacies with their monthly and total sales
            st.subheader(f"Top {top_n} Pharmacies by Total Unit Sales (with at least one sale per month)")
            st.write(top_pharmacies)

            # Convert the data to a CSV for download
            csv = top_pharmacies.to_csv().encode('utf-8')
            st.download_button(
                label="Download Top Pharmacies Sales Data as CSV",
                data=csv,
                file_name='top_pharmacies_sales_data.csv',
                mime='text/csv',
            )

            # Define month columns
            months = data.columns[5:]

            # Identify pharmacies with at least one product sale per month
            pharmacies_with_sales = data.groupby('NAME').filter(lambda x: all(x[month].sum() > 0 for month in months))

            # Calculate total and monthly sales for each of these pharmacies
            total_and_monthly_sales = pharmacies_with_sales.groupby('NAME')[months].sum()
            total_and_monthly_sales['Total Sales'] = total_and_monthly_sales.sum(axis=1)

            # Determine the top N pharmacies based on total sales
            top_pharmacies = total_and_monthly_sales.sort_values(by='Total Sales', ascending=False).head(top_n)

            # Extract product sales data for these top pharmacies
            top_pharmacy_names = top_pharmacies.index.tolist()
            top_pharmacies_product_sales = pharmacies_with_sales[pharmacies_with_sales['NAME'].isin(top_pharmacy_names)]
            st.subheader(f"Top {top_n} Pharmacies by Unit Sales")
            st.write(top_pharmacies_product_sales)

            # Convert the data to a CSV for download
            csv = top_pharmacies_product_sales.to_csv().encode('utf-8')
            st.download_button(
                label = "Download Top Pharmacies Unit Sales as CSV",
                data = csv,
                file_name='top_pharmacies_unit_sales.csv',
                mime='text/csv',
            )

        elif options == 'Alerts':
            st.header('Pharmacy - Products Alerts')

            # User input for town selection
            towns = data['TOWN'].astype(str).unique().tolist()  # Convert all elements to strings
            towns.sort()  # Sort the list of towns
            towns.insert(0, 'All Towns')  # Add 'All Towns' option
            selected_town = st.selectbox('Select Town (or choose "All Towns" for all)', towns)

            # Filtering data by selected town if not 'All Towns'
            if selected_town != 'All Towns':
                data = data[data['TOWN'] == selected_town]

            # User input for pharmacy selection
            pharmacies = data['NAME'].unique().tolist()  # Assuming 'NAME' column contains Pharmacy names
            pharmacies.sort()  # Sort the list of pharmacies
            pharmacies.insert(0, 'All Pharmacies')  # Adding an option to select all pharmacies
            selected_pharmacy = st.selectbox('Select Pharmacy (or choose "All Pharmacies" for all)', pharmacies)

            # Filtering data by selected pharmacy if not 'All Pharmacies'
            if selected_pharmacy != 'All Pharmacies':
                data = data[data['NAME'] == selected_pharmacy]

            # User input for the number of top alerts
            num_alerts = st.number_input('Select number of top alerts to display', min_value=1, value=20, step=1)

            # Function for generating alerts
            def alert_sales_dip(data, num_alerts, period_desc):
                alerts = []
                grouped_data = data.groupby(['NAME', 'DISCRIPTION'])

                for (pharmacy, product), group in grouped_data:
                    monthly_sales = group.iloc[:, 5:] # includes monthly last column
                    average_sales = monthly_sales.mean(axis=1)

                    last_month_sales = monthly_sales.iloc[:, -1] if period_desc == "the last month" else monthly_sales.iloc[:, -2]
                    sales_dip = average_sales.values[0] - last_month_sales.values[0]
                    if last_month_sales.values[0] < average_sales.values[0]:
                        sales_dip_rounded = round(sales_dip)
                        alert_msg = f"Alert: {product} at {pharmacy} - Sales in {period_desc} for {selected_town} for {selected_pharmacy} are below the average by {sales_dip_rounded} units"
                        alerts.append((alert_msg, sales_dip_rounded))

                if not alerts:
                    return [f"No alerts for {selected_town} in {period_desc}"], []

                # Sorting and limiting alerts
                top_alerts = sorted(alerts, key=lambda x: x[1], reverse=True)[:num_alerts]
                alert_messages = [alert[0] for alert in top_alerts]
                alert_data = [{"Pharmacy": alert[0].split(' at ')[1].split(' - ')[0], "Product": alert[0].split('Alert: ')[1].split(' at ')[0], "Sales Dip": alert[1]} for alert in top_alerts]
                return alert_messages, alert_data

            # Narration and dataframe for last month
            st.header(f"Top {num_alerts} Alerts for {selected_town} for {selected_pharmacy} Narration for last month")
            sales_alerts, alert_data = alert_sales_dip(data, num_alerts, "the last month")
            for alert in sales_alerts:
                st.write(alert)  # Displaying the top N alerts

            alerts_df = pd.DataFrame(alert_data)

            if alerts_df.empty:
                st.write(f"No alerts for {selected_town} in the last month")
            else:
                st.subheader(f"Top {num_alerts} Alerts for {selected_town} for {selected_pharmacy} Table for last month")
                st.table(alerts_df)

                # CSV download button for last month
                csv_last_month = alerts_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Top Alerts for Last Month CSV",
                    data=csv_last_month,
                    file_name="top_sales_alerts_last_month.csv",
                    mime='text/csv'
                )

            # Narration and dataframe for the month before last month
            st.header(f"Top {num_alerts} Alerts for {selected_town} for {selected_pharmacy} Narration for month before last")
            sales_alerts, alert_data = alert_sales_dip(data, num_alerts, "the month before last")
            for alert in sales_alerts:
                st.write(alert)  # Displaying the top N alerts

            alerts_df = pd.DataFrame(alert_data)

            if alerts_df.empty:
                st.write(f"No alerts for {selected_town} in the month before last")
            else:
                st.subheader(f"Top {num_alerts} Alerts for {selected_town} for {selected_pharmacy} Table for month before last")
                st.table(alerts_df)

                # CSV download button for the month before last month
                csv_month_before_last = alerts_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Top Alerts for Month Before Last CSV",
                    data=csv_month_before_last,
                    file_name="top_sales_alerts_month_before_last.csv",
                    mime='text/csv'
                )

            # Data validation for product alets
            # Load data frame
            df = pd.DataFrame(data)

            # Streamlit UI
            st.subheader('Pharmacy Product Sales Data')

            # Dropdown for selecting Pharmacy
            pharmacy_list = df['NAME'].unique().tolist()  # Extract unique pharmacy names
            pharmacy_list.sort()  # Sort the pharmacy list in ascending order
            selected_pharmacy = st.selectbox('Select Pharmacy', pharmacy_list)

            # Multiselect for selecting Products
            product_list = sorted(df['DISCRIPTION'].unique().tolist())  # Sort unique product descriptions
            product_list.insert(0, 'All Products')  # Add 'All Products' as the first option
            # Set default to 'All Products'
            selected_products = st.multiselect('Select Products', product_list, default=['All Products'])

            # Logic to handle the "All Products" selection
            if 'All Products' in selected_products and len(selected_products) > 1:
                selected_products = ['All Products']

            # Filter DataFrame based on selections
            filtered_df = df[df['NAME'] == selected_pharmacy]

            # Check if 'All Products' is selected
            if 'All Products' in selected_products:
                # Do not filter by products
                filtered_df = filtered_df
            else:
                # Filter by selected products
                filtered_df = filtered_df[filtered_df['DISCRIPTION'].isin(selected_products)]

            # Display the filtered DataFrame
            st.write('Filtered Data:', filtered_df)

            # Convert DataFrame to CSV. Index=False to exclude DataFrame index from the CSV
            csv = filtered_df.to_csv(index=False).encode('utf-8')

            # Create a download button and provide the CSV file to download
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='alerts_data.csv',
                mime='text/csv',
            )

            # Product sales two or more consecutive drops of a variable or more
            # Streamlit User Interface
            st.subheader("Two or More Consecutive Drop in Unit Sales")

            # Ensure tqdm is patched into pandas
            tqdm.pandas()

            # User input for the percentage drop
            percentage_drop = st.number_input("Enter the percentage drop for alert (e.g., 25 for 25%)", min_value=10, value=25, max_value=30)
            drop_threshold = -percentage_drop / 100

            # Convert all sales columns to numeric, coercing errors to NaN
            sales_cols = df.columns[df.columns.str.contains('-')]
            df[sales_cols] = df[sales_cols].apply(pd.to_numeric, errors='coerce')

            # Define a function to check for two or more consecutive drops in sales
            def check_consecutive_drops(row):
                # Calculate the percentage change between months
                changes = row[sales_cols].pct_change().fillna(0)
                # Check for two or more consecutive drops of specified percentage or more
                drops = (changes <= drop_threshold)
                return drops.sum() >= 2

            # Dropdown for product selection
            product_list = sorted(df['DISCRIPTION'].unique())  # Extract unique product descriptions and sort
            selected_product = st.selectbox("Select a Product", product_list)

            # Dropdown for town selection
            town_list = df['TOWN'].dropna().astype(str).unique().tolist()  # Convert to string and drop NaNs
            town_list.sort()  # Sort the list of towns
            selected_town = st.selectbox("Select a Town", town_list, key="town_selection_key")

            # Filter the DataFrame by 'DISCRIPTION' and 'TOWN'
            filtered_df = df[(df['DISCRIPTION'] == selected_product) & (df['TOWN'] == selected_town)]

            # Apply the function across the rows
            consecutive_drops = filtered_df.progress_apply(check_consecutive_drops, axis=1)

            # Filter the DataFrame based on the condition
            final_filtered_df = filtered_df[consecutive_drops]

            # Handling empty DataFrame
            if final_filtered_df.empty:
                st.write("No results found for the selected criteria.")
            else:
                try:
                    final_filtered_df = final_filtered_df.drop(columns=['P-CODE', 'C-CODE'])
                except KeyError as e:
                    print("Columns not found in DataFrame: ", e)

                # Display the filtered DataFrame
                st.write(final_filtered_df)

                # Convert DataFrame to CSV. Index=False to exclude DataFrame index from the CSV
                csv = final_filtered_df.to_csv(index=False).encode('utf-8')

                # Create a download button and provide the CSV file to download
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='product_alerts_data.csv',
                    mime='text/csv',
                )

        elif options == 'Sales Forecasting':
            st.subheader("Sales Forecasting")
            # User selects a product
            # Extracting all product names
            all_products = sorted(data['DISCRIPTION'].unique().tolist())
            selected_product = st.selectbox('Select a Product for Forecasting', all_products)

            # Filtering data for a specific product
            product_data = data[data['DISCRIPTION'] == selected_product]

            # Summing up monthly sales data for the selected product
            monthly_sales = product_data.iloc[:, 6:].sum()

            # Converting the data into a time series format
            monthly_sales.index = pd.to_datetime(monthly_sales.index, format='%b-%y')

            # Sort index in case it's out of order
            monthly_sales = monthly_sales.sort_index()

            # Auto ARIMA model (assuming no seasonality)
            auto_model = auto_arima(monthly_sales, seasonal=False, trace=True)

            # Auto ARIMA model (assuming no seasonality)
            if 'auto_model' not in st.session_state or st.session_state.selected_product != selected_product:
                # Update the session state
                st.session_state.auto_model = auto_arima(monthly_sales, seasonal=False, trace=True)
                st.session_state.selected_product = selected_product
                
            # Forecasting
            forecast_periods = st.slider('Select number of months to forecast', 1, 12, 3)
            forecast = auto_model.predict(n_periods=forecast_periods)

            # Format the forecast to whole numbers
            forecast_rounded = np.round(forecast).astype(int)
            
            # Create a DataFrame from the forecast data
            data = {
                "Month": [f"Month {i+1}" for i in range(forecast_periods)],
                "Forecast (Rounded)": forecast_rounded,
                "Original Forecast": [f"{forecast[i]:.2f}" for i in range(forecast_periods)]
            }
            df = pd.DataFrame(data)

            # Convert the DataFrame to CSV
            csv = df.to_csv(index=False)

            # Display the forecast
            st.write(f'Forecast for the next {forecast_periods} months:')
            st.write(df)
            
            # Create a download button and offer the CSV for download
            st.download_button(
                label="Download forecast data as CSV",
                data=csv,
                file_name='forecast_data.csv',
                mime='text/csv',
            )
            
            st.subheader("Model Evaluation")
            # Check for NaN values in monthly_sales
            if monthly_sales.isna().any():
                st.write("Handling NaN values in monthly sales data.")
                monthly_sales.fillna(method='ffill', inplace=True)  # Forward fill as an example

#             # Assuming monthly_sales is a Pandas Series or a list containing your entire time series data
#             # Slice the last 'forecast_periods' months from monthly_sales
#             actual_values = np.array(monthly_sales[-forecast_periods:])

#             # Ensure forecast is a numpy array
#             forecast_values = np.array(forecast)  # Your forecasted sales data
            
#             if len(actual_values) != len(forecast_values):
#             # Option 1: Truncate the longer array (example)
#             min_length = min(len(actual_values), len(forecast_values))
#             actual_values = actual_values[:min_length]
#             forecast_values = forecast_values[:min_length]

            # Ensure forecast_periods is an integer and not larger than the length of monthly_sales
            forecast_periods = min(forecast_periods, len(monthly_sales))

            # Slice the last 'forecast_periods' months from monthly_sales
            actual_values = np.array(monthly_sales[-forecast_periods:])

            # Ensure forecast is a numpy array and has the same length as forecast_periods
            forecast_values = np.array(forecast[:forecast_periods])

            # Check if lengths match
            if len(actual_values) != len(forecast_values):
                raise ValueError(f"Length mismatch: actual_values has {len(actual_values)} elements, "
                                 f"forecast_values has {len(forecast_values)} elements.")

            # Calculate the metrics
            mae = mean_absolute_error(actual_values, forecast_values)
            mse = mean_squared_error(actual_values, forecast_values)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual_values - forecast_values) / (actual_values + np.finfo(float).eps))) * 100

            # Display the metrics in Streamlit
            st.metric("Mean Absolute Error", f"{mae:.2f}")
            st.metric("Mean Squared Error", f"{mse:.2f}")
            st.metric("Root Mean Squared Error", f"{rmse:.2f}")
            st.metric("Mean Absolute Percentage Error", f"{mape:.2f}%")
               
        elif options == 'Market Segmentation':
            st.subheader("Market Segmentation")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="dataload")
            
            if uploaded_file is not None:
                data = load_data(uploaded_file)
                data.columns = data.columns.str.strip()
                if data is not None and 'Invoice Date' in data.columns:
                    # Now that we have confirmed the data is loaded and the column exists,
                    # we can proceed with the rest of the code that uses 'Invoice Date'
                    try:
                        # ... Perform further operations with 'data' ...
                        # RFM Analysis
                        latest_date = data['Invoice Date'].max() + dt.timedelta(days=1)
                        rfm = data.groupby('Name').agg({
                            'Invoice Date': lambda x: (latest_date - x.max()).days,
                            'Name': 'count',
                            'Units Sold': 'sum'
                        }).rename(columns={'Invoice Date': 'Recency', 
                                        'Name': 'Frequency', 
                                        'Units Sold': 'MonetaryValue'})

                        # Scale the data
                        scaler = StandardScaler()
                        rfm_scaled = scaler.fit_transform(rfm)

                        # Determine the optimal number of clusters
                        inertia = []
                        for k in range(1, 11):
                            kmeans = KMeans(n_clusters=k, random_state=0)
                            kmeans.fit(rfm_scaled)
                            inertia.append(kmeans.inertia_)

                        # Plotting the Elbow Method
                        st.write("Elbow Method to Determine Optimal Clusters:")
                        plt.figure(figsize=(10, 6))
                        # Create a figure and axis object
                        fig, ax = plt.subplots()
                        ax.scatter([1, 2, 3], [1, 2, 3])
                        plt.plot(range(1, 11), inertia, marker='o')
                        plt.title('The Elbow Method')
                        plt.xlabel('Number of clusters')
                        plt.ylabel('Inertia')
                        st.pyplot(fig)

                        # Finding the elbow point
                        kl = KneeLocator(range(1, 11), inertia, curve='convex', direction='decreasing')
                        optimal_clusters = kl.elbow
                        st.write(f"Optimal Number of Clusters: {optimal_clusters}")
                        
                        st.title('Customer Segment Grouping')
                        
                       # K-Means clustering with optimal clusters
                        kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
                        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
                        rfm = rfm.sort_values(by='Cluster')
                        
                        rfm['RFM_SCORE'] = rfm['Recency'].astype('int') + rfm['Frequency'].astype('int') + rfm['MonetaryValue'].astype('int') 
                        rfm['RFM Customer Segments'] = rfm.apply(cluster_segments,axis=1) 
                        rfm['RFM Customer Segments'] 
                        
                        # Reset the index to turn 'Name' from an index into a column
                        rfm = rfm.reset_index()
                        
                        #
                        # Check the rfm DataFrame
                        st.write(rfm)  # Inspect the first few rows of the rfm DataFrame


                        # Creating a new DataFrame with only 'Name' and 'RFM Customer Segments'
                        rfm_segments = rfm[['Name', 'RFM_SCORE', 'RFM Customer Segments']]
                        
                        #
                        # Inspect the rfm_segments DataFrame
                        st.write(rfm_segments)  # Inspect the first few rows of the rfm_segments DataFrame


                        # Function to convert DataFrame to CSV
                        def convert_df_to_csv(df):
                            return df.to_csv(index=False).encode('utf-8')

                        # Provide a download button for the CSV file
                        csv = convert_df_to_csv(rfm_segments)  # Convert the new DataFrame to CSV
                        st.download_button(
                            label="Download RFM Customer Segments as CSV",
                            data=csv,
                            file_name='rfm_customer_segments.csv',
                            mime='text/csv',
                        )
                    
                        # Show the resulting segments in a table
                        segment_table = rfm.groupby('RFM Customer Segments').agg({
                            'Recency': 'mean',
                            'Frequency': 'mean',
                            'MonetaryValue': ['mean', 'count']
                        }).sort_values(by=('MonetaryValue', 'count'), ascending=False)

                        segment_table.reset_index(inplace=True)
                        segment_table.columns = ['Segment', 'Average Recency', 'Average Frequency', 'Average Monetary Value', 'Customer Count']
                        
                        # Calculate the total number of customers
                        total_customers = segment_table['Customer Count'].sum()

                        # Calculate the percentage of customers in each segment
                        segment_table['% Customer Count'] = (segment_table['Customer Count'] / total_customers) * 100
                        
                        st.title("Market Segments")
                        st.write(segment_table)
                        
                        # Converting the DataFrame to CSV and encoding to bytes
                        csv = segment_table.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Segment Table CSV",
                            data=csv,
                            file_name="Segment_Table.csv",
                            mime='text/csv'
                        )

                        
                        # Create a bar plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(data=segment_table, x='Customer Count', y='Segment', ax=ax)

                        # Add labels and title
                        ax.set_xlabel('Customer Count')
                        ax.set_ylabel('Segment')
                        ax.set_title('Customer Count per Segment')
                        
                        # Display the plot in Streamlit
                        st.pyplot(fig)
                                                                                             
                        # Create a treemap chart
                        fig = px.treemap(segment_table, path=['Segment'], values='Customer Count')

                        # Set chart title
                        fig.update_layout(title='Customer Count per Segment (Treemap)')

                        # Display the treemap chart in Streamlit
                        st.plotly_chart(fig)
                                            
                        # Create a new figure for 3D plotting
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')

                        # Plot data
                        # Assuming 'MonetaryValue' is your third dimension
                        sc = ax.scatter(rfm['Recency'], rfm['Frequency'], rfm['MonetaryValue'], c=rfm['Cluster'], cmap='viridis')

                        # Set labels and title
                        ax.set_xlabel('Recency')
                        ax.set_ylabel('Frequency')
                        ax.set_zlabel('Monetary Value')
                        ax.set_title('RFM Customer Segments 3D')
                        
                        # Adjust legend
                        # Create a legend and place it outside the plot area
                        legend = ax.legend(*sc.legend_elements(), title='Cluster', loc='center left', bbox_to_anchor=(1.2, 0.5))

                        # Show plot
                        st.pyplot(fig)
                        
                        
                    except TypeError as e:
                        st.error(f"TypeError encountered: {e}")
                else:
                    st.error("The uploaded file does not contain an 'Invoice Date' column or failed to load correctly.")
            else:
                st.write("Please upload a CSV file.")
                
        elif options == 'Sentiment Analysis':
                        
            filename = st.file_uploader("Upload reviews data:", type=("csv"))
            
            st.markdown("<h1 style='font-size:30px;'>Voice of the Customer: Sentiment Analysis</h1>", unsafe_allow_html=True)

            if filename is not None:
                data = pd.read_csv(filename, encoding="utf-8")  # Assuming CSV file format
                data["body"] = data["body"].astype("str")

                # Apply VADER sentiment analysis
                data["scores"] = data["body"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
                data["sentiment"] = data["scores"].apply(lambda x: "Positive" if x >= 0.05 else ("Neutral" if x > -0.05 else "Negative"))

                data = data[['brand', 'body', 'sentiment', 'scores', 'date']]
                data['date'] = pd.to_datetime(data['date'])
                data['quarter'] = pd.PeriodIndex(data.date, freq='Q')

                # Create a dynamic list of unique brands from the data
                brands = data['brand'].unique().tolist()

                # Set the default selected brand to be the first brand in the list
                default_brand = brands[0]

                st.sidebar.write("Select Brands for Sentiment Analysis:")
                selected_brands = st.sidebar.multiselect("Select brands:", brands, default=default_brand)

                # Define colors for sentiments
                sentiment_colors = {"Positive": "green", "Neutral": "blue", "Negative": "red"}

                st.subheader("Sentiment Analysis for Selected Brands")

                brand_sentiments = []

                for brand in selected_brands:
                    st.markdown(f"### {brand}")
                    brand_data = data[data['brand'] == brand]

                    # Processing data for sentiment analysis
                    sentiment_count = brand_data['sentiment'].value_counts().reset_index()
                    sentiment_count.columns = ['Sentiment', 'Count']

                    # Displaying review counts
                    st.sidebar.write(f"Reviews count by brand ({brand}):")
                    st.sidebar.write(sentiment_count)

                    # Plotting sentiment distribution with specified colors
                    st.subheader(f"Brand Reviews Sentiment Distribution ({brand})")
                    fig = px.pie(sentiment_count, values='Count', names='Sentiment', title=f'Sentiment distribution for {brand}', color='Sentiment', color_discrete_map=sentiment_colors)
                    st.plotly_chart(fig, use_container_width=True)

                    # Trend analysis
                    st.subheader(f"Trend Analysis of Sentiments for {brand}")

                    # First, ensure the 'date' column is in datetime format
                    brand_data['date'] = pd.to_datetime(brand_data['date'])

                    # Create a 'Month-Year' column before grouping
                    brand_data['Month-Year'] = brand_data['date'].dt.strftime('%m-%Y')

                    # Group by the original 'date' column and 'sentiment', then create 'Month-Year' for display
                    trend_data = brand_data.groupby([pd.Grouper(key='date', freq='M'), 'sentiment']).size().reset_index(name='Count')
                    trend_data['Month-Year'] = trend_data['date'].dt.strftime('%m-%Y')

                    # No need to sort by 'Month-Year' as it's already sorted by the 'date' column
                    fig2 = px.line(trend_data, x="Month-Year", y="Count", color='sentiment', title=f'Monthly Sentiment Trends for {brand}', color_discrete_map=sentiment_colors)
                    st.plotly_chart(fig2, use_container_width=True)


                    # Word clouds for each sentiment
                    st.subheader(f"Word Clouds for Reviews Sentiment ({brand})")
                    col1, col2, col3 = st.columns(3)
                    for sentiment, col in zip(['Positive', 'Neutral', 'Negative'], [col1, col2, col3]):
                        sentiment_data = brand_data[brand_data['sentiment'] == sentiment]
                        words = ' '.join(sentiment_data['body'])
                        if words:
                            wordcloud = WordCloud(width=300, height=150).generate(words)
                            col.image(wordcloud.to_array(), caption=f'{sentiment} Reviews Word Cloud', use_column_width=True)

                    # Display top 5 reviews for each sentiment
                    for sentiment in ['Positive', 'Neutral', 'Negative']:
                        st.subheader(f"Top 5 {sentiment} Reviews for {brand}:")
                        sentiment_reviews = brand_data[brand_data['sentiment'] == sentiment].nlargest(5, 'scores')
                        if not sentiment_reviews.empty:
                            for index, row in sentiment_reviews.iterrows():
                                st.write(f"{row['body']} - Score: {row['scores']}")
                        else:
                            st.write(f"No {sentiment.lower()} reviews found for this brand.")

                    brand_sentiments.append(brand_data)

                # Sentiment comparison between selected brands
                st.subheader("Sentiment Comparison Between Brands")
                combined_data = pd.concat(brand_sentiments)
                fig3 = px.bar(combined_data, x="brand", y="scores", color="sentiment", barmode="group", title='Sentiment Comparison Between Brands', color_discrete_map=sentiment_colors)
                st.plotly_chart(fig3, use_container_width=True)
                
        elif options == 'Net Promoter Score':
            # Streamlit app layout
            st.subheader("Net Promoter Score Calculator")

            # File uploader for periodic data
            uploaded_file = st.file_uploader("Upload your CSV file with periodic data", type="csv")
            if uploaded_file is not None:
                # Read data
                data = pd.read_csv(uploaded_file)

                # Check if the CSV has the expected columns: 'Period' and 'Score'
                if 'Period' in data.columns and 'Score' in data.columns:
                    # Calculate NPS for each period
                    period_nps = data.groupby('Period')['Score'].apply(calculate_nps).reset_index(name='NPS')
                    st.write("Net Promoter Score for each Period:")
                    st.write(period_nps)
                else:
                    st.error("CSV file must have columns named 'Period' and 'Score'")

    else:
        st.warning('Please upload a CSV file to proceed.')


# In[ ]:




