import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Define the UI
st.title("Customer Segmentation")

st.header("Input Customer Data")

# Input fields for customer data
customer_name = st.text_input('Name')
customer_age = st.number_input('Age', min_value=0, max_value=100, value=30)
customer_income = st.number_input('Annual Income (k$)', min_value=0, max_value=200, value=50)
customer_spending = st.number_input('Spending Score (1-100)', min_value=1, max_value=100, value=50)

# Collect data into a DataFrame
customer_data = {
    'Name': customer_name,
    'Age': customer_age,
    'Annual Income (k$)': customer_income,
    'Spending Score (1-100)': customer_spending
}

data = pd.DataFrame([customer_data])

# Initialize session state data
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['Name', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

# Add new customer data to session state data
if st.button("Add Customer"):
    if customer_name:  # Ensure the name is not empty
        st.session_state['data'] = pd.concat([st.session_state['data'], data], ignore_index=True)
        st.write(f"Customer {customer_name} added.")
    else:
        st.write("Please enter the customer's name.")
    st.write(st.session_state['data'])

# Clustering
if st.button("Cluster Customers"):
    if st.session_state['data'].empty:
        st.write("No customer data to cluster.")
    else:
        num_clusters = st.slider("Select number of clusters", 1, 10, 3)
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(st.session_state['data'][['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
        st.session_state['data']['Cluster'] = kmeans.labels_

        st.write("Clustered Data:")
        st.write(st.session_state['data'])

        # Plotting the clusters
        fig, ax = plt.subplots()
        scatter = ax.scatter(st.session_state['data']['Age'], st.session_state['data']['Annual Income (k$)'],
                             c=st.session_state['data']['Cluster'], cmap='viridis')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        st.pyplot(fig)
