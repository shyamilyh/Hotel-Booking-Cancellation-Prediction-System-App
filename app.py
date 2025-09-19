import streamlit as st
import pandas as pd
import joblib

# Load the trained model
joblib.dump(best_model, "best_hotel_cancellation_model.pkl")

st.title('Hotel Booking Cancellation Prediction')

st.write("""
Enter the details of the hotel booking to predict if it will be canceled.
""")

# Create input fields for the features used in the model
# You will need to adjust these based on the exact features in your X_train_smote
# Refer to X_train_smote.columns to get the list of expected features

# Example input fields (replace with your actual features)
lead_time = st.slider('Lead Time (days)', 0, 700, 100)
previous_cancellations = st.number_input('Previous Cancellations', 0, 26, 0)
previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', 0, 72, 0)
booking_changes = st.number_input('Booking Changes', 0, 21, 0)
days_in_waiting_list = st.number_input('Days in Waiting List', 0, 400, 0)
adr = st.slider('Average Daily Rate (ADR)', -10.0, 300.0, 100.0) # Adjust range based on your data
required_car_parking_spaces = st.number_input('Required Car Parking Spaces', 0, 3, 0) # Max is 8, but usually 0 or 1
total_of_special_requests = st.number_input('Total of Special Requests', 0, 5, 0)
total_guests = st.number_input('Total Guests', 0, 20, 2) # Adjust max based on your data
arrival_month = st.slider('Arrival Month (1=Jan, 12=Dec)', 1, 12, 7)
arrival_day_of_week = st.slider('Arrival Day of Week (0=Mon, 6=Sun)', 0, 6, 2)
total_nights_stay = st.number_input('Total Nights Stay', 0, 60, 3) # Adjust max based on your data
is_weekend_booking = st.selectbox('Is Weekend Booking', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
lead_time_x_hotel_type = st.number_input('Lead Time x City Hotel (0 if Resort, Lead Time if City)', 0, 700, 0) # Need to handle this based on hotel type input
is_popular_month = st.selectbox('Is Popular Month (Top 3 booking months)', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Assuming you have one-hot encoded features for deposit_type
deposit_type_No_Deposit = st.selectbox('Deposit Type: No Deposit', [False, True])
deposit_type_Non_Refund = st.selectbox('Deposit Type: Non Refund', [False, True])
deposit_type_Refundable = st.selectbox('Deposit Type: Refundable', [False, True])


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

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the order of columns in the input DataFrame matches the training data
# This is crucial for correct prediction
# You might need to load a sample of your training data columns to get the exact order
# For now, let's assume the order is as defined above, but this is a potential point of failure
# A more robust approach would be to save the column order during training and load it here.

# Make prediction
if st.button('Predict Cancellation'):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error('This booking is likely to be **Canceled**')
    else:
        st.success('This booking is likely **Not Canceled**')
