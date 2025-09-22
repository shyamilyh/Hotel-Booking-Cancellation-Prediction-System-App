
import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# ----------------------------
# Load model and column order
# ----------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("best_hotel_cancellation_model.pkl")
    with open("model_columns.json") as f:
        cols = json.load(f)
    return model, cols

model, model_columns = load_resources()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Hotel Booking Cancellation Prediction", page_icon="🏨")
st.title("🏨 Hotel Booking Cancellation Prediction")

st.markdown("Enter booking details below to predict whether a booking will be **Canceled** or **Not Canceled**.")

# Example input fields (adjust to match your dataset and feature engineering)
# Refer to the 'model_columns' list for the expected feature names

lead_time = st.slider("Lead Time (days)", 0, 700, 100)
previous_cancellations = st.number_input("Previous Cancellations", 0, 26, 0)
previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", 0, 72, 0)
booking_changes = st.number_input("Booking Changes", 0, 21, 0)
days_in_waiting_list = st.number_input("Days in Waiting List", 0, 400, 0)
# Use pre-calculated min, max, and mean for ADR slider
adr = st.slider("Average Daily Rate (ADR)", -6.38, 5400.0, 101.83) # Using values from df.describe()
required_car_parking_spaces = st.number_input("Required Car Parking Spaces", 0, 8, 0)
total_of_special_requests = st.number_input("Total of Special Requests", 0, 5, 0)
total_guests = st.number_input("Total Guests", 0, 55, 2) # Adjust max based on your data
arrival_month = st.slider("Arrival Month (1=Jan, 12=Dec)", 1, 12, 7)
arrival_day_of_week = st.slider("Arrival Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
total_nights_stay = st.number_input("Total Nights Stay", 0, 60, 3) # Adjust max based on your data
is_weekend_booking = st.selectbox("Is Weekend Booking", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
# For lead_time_x_hotel_type, you might need an input for hotel type first
# Let's add a simple hotel type input for now and calculate the interaction term
hotel_type = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
lead_time_x_hotel_type = lead_time * (1 if hotel_type == 'City Hotel' else 0)

# For is_popular_month, you need to know which months are popular based on your EDA
# You can hardcode the popular months or save them during feature engineering
# For this example, let's assume you identified them and hardcode (replace with your actual popular months)
popular_months = [8, 7, 5] # Example: August, July, May
is_popular_month = 1 if arrival_month in popular_months else 0


st.subheader("Deposit Type (Select only one as True)")
deposit_type_No_Deposit = st.checkbox('No Deposit', value=True)
deposit_type_Non_Refund = st.checkbox('Non Refund')
deposit_type_Refundable = st.checkbox('Refundable')

# Ensure only one deposit type is selected as True
if sum([deposit_type_No_Deposit, deposit_type_Non_Refund, deposit_type_Refundable]) > 1:
    st.warning("Please select only one Deposit Type as True.")
    st.stop()


# Create a dictionary with the input values
input_data = {
    'lead_time': lead_time,
    'previous_cancellations': previous_cancellations,
    'previous_bookings_not_canceled': previous_bookings_not_canceled,
    'booking_changes': booking_changes,
    'days_in_waiting_list': days_in_waiting_list,
    'adr': adr,
    'required_car_parking_spaces': required_car_parking_spaces,
    'total_of_special_requests': total_of_special_requests,
    'total_guests': total_guests,
    'arrival_month': arrival_month,
    'arrival_day_of_week': arrival_day_of_week,
    'total_nights_stay': total_nights_stay,
    'is_weekend_booking': is_weekend_booking,
    'lead_time_x_hotel_type': lead_time_x_hotel_type,
    'is_popular_month': is_popular_month,
    'deposit_type_No Deposit': deposit_type_No_Deposit,
    'deposit_type_Non Refund': deposit_type_Non_Refund,
    'deposit_type_Refundable': deposit_type_Refundable,
}

# Convert input data to a DataFrame and reindex to match training columns
input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=model_columns, fill_value=0)


# Make prediction
if st.button('Predict Cancellation'):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"❌ This booking is likely to be **Canceled** (Probability: {prediction_proba[1]:.2f})")
    else:
        st.success(f"✅ This booking is likely **Not Canceled** (Probability: {prediction_proba[0]:.2f})")

