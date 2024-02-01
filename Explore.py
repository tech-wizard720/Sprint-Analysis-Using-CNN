import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math as math
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import pickle


interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

def show_explore_page():
    # Set the title and page layout
    st.set_page_config(
        page_title="Performance Analysis App for Athletes",
        page_icon="🏋️",
        layout="centered"
    )

    # Create a folder for storing uploaded videos
    video_folder = "uploaded_videos"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    # Define custom CSS for styling

    # Header and introduction
    st.title("Performance Analysis App for Athletes")
    st.image("logo.png", use_column_width=False, width=150)  # Add your logo image
    st.write("Upload a video and enter your height and weight for analysis.")

    # Apply the custom style to the main content
    st.markdown('<div class="stFrame">', unsafe_allow_html=True)

    # Upload video
    uploaded_file = st.file_uploader("Upload a video:", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        st.success("Video Uploaded Successfully!")

        # Get user's height and weight
        user_height = st.number_input("Enter your height (cm):", min_value=1)
        user_weight = st.number_input("Enter your weight (kg):", min_value=1)
        user_gender = st.radio("Select your gender:", ("Male", "Female"))

        # Add a submit button
        if st.button("Submit"):
            # You can perform actions with the submitted data here
            st.subheader("Video Information:")
            st.write(f"File Name: {uploaded_file.name}")
            st.write("Height:", user_height, "cm")
            st.write("Weight:", user_weight, "kg")
            st.write("Gender:", user_gender)

            # Save the video to the project folder
            video_path = os.path.join(video_folder, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            st.write("Video saved to the project folder.")

            # Display video details
            # Load and display the uploaded video
            process_video(video_path, user_height, user_weight, user_gender)

    # Additional information
    st.markdown('<p class="stInfo">Please ensure your video format is one of the supported types (mp4, avi, mov).</p>', unsafe_allow_html=True)

    # Close the custom style
    st.markdown('</div>', unsafe_allow_html=True)

   
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def process_video(path,height,weight,gender):
    print("processed video successfully")
    left_elbow_index = 7
    left_shoulder_index = 5
    left_hip_index = 11
    right_elbow_index = 8
    right_shoulder_index = 6
    right_hip_index = 12
    left_ankle_x = []
    left_knee_angles = []
    right_knee_angles = []
    right_ankle_x = []
    last_left_ankle_x = None
    stride_lengths = []
    left_knee_index=13
    right_knee_index=13
    left_ankle_x_positions = []
    pelvic_tilt_angles = []
    left_elbow_flaring_angles = []
    right_elbow_flaring_angles = []
    vertical_upright_angles = []
    keypoints_positions_array = [] 
    left_ankle_index= 15
    right_ankle_index= 15

    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()
        data = load_model()
        model=data['model']
        if not ret:
            break

        # Reshape image
        img = frame.copy()
        input_image = tf.image.resize(np.expand_dims(img, axis=0), [192, 192])
        input_image = tf.cast(input_image, dtype=tf.float32)


        # Setup input and output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        keypoints_positions_array.append(keypoints_with_scores[0, :, :, :2])  # Store keypoint positions

        # Calculate stride length
        left_ankle_x = keypoints_with_scores[0, 0, 15, 0]
        right_ankle_x = keypoints_with_scores[0, 0, 16, 0]
        left_ankle_x_positions.append(left_ankle_x)

        # Calculate pelvic tilt
        current_pelvic_tilt = calculate_pelvic_tilt(keypoints_with_scores)
        pelvic_tilt_angles.append(current_pelvic_tilt)

        # Calculate elbow flaring angles
        elbow_flaring = calculate_elbow_flaring(keypoints_with_scores, left_elbow_index, left_shoulder_index, left_hip_index, right_elbow_index, right_shoulder_index, right_hip_index)
        left_elbow_flaring_angles.append(elbow_flaring[0])
        right_elbow_flaring_angles.append(elbow_flaring[1])

        current_vertical_upright_angle = calculate_vertical_upright_angle(
            keypoints_with_scores, left_shoulder_index, right_shoulder_index
        )
        vertical_upright_angles.append(current_vertical_upright_angle)

        # Calculate stride length
        if left_ankle_x is not None:
            if last_left_ankle_x is not None:
                stride_length = abs(right_ankle_x - last_left_ankle_x)
                stride_lengths.append(stride_length)
            last_left_ankle_x = left_ankle_x
        #calculte knee angle 
        left_knee_angle, right_knee_angle = calculate_knee_angles(keypoints_with_scores)
        left_knee_angles.append(left_knee_angle)
        right_knee_angles.append(right_knee_angle)

        # Rendering
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores, 0.4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame with keypoints
        cv2.imshow('MoveNet Lightning', frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    stride=visualize_stride_pattern(stride_lengths, left_ankle_x_positions)
    vertical_angle= visualize_vertical_upright_angle(vertical_upright_angles)
    print(vertical_angle)
    pelvic_tilt=visualize_pelvic_tilt(pelvic_tilt_angles)
    visualize_knee_angles(left_knee_angles, right_knee_angles)
    left, right=visualize_elbow_flaring(left_elbow_flaring_angles, right_elbow_flaring_angles)
    actual_values=np.empty((0,4))
    column_names = ['Mean_deviation_stride', 'Vertical_upright_angle', 'Left_elbow', 'Right_elbow']
    actual_values = np.append(actual_values, [[stride, vertical_angle, left, right]], axis=0)
    actual_values = pd.DataFrame(actual_values, columns=column_names)
    print(actual_values)
    predicted_values = train_multivariate_regression_model(model,height, weight,gender)
    predicted_values = pd.DataFrame(predicted_values, columns=column_names)
    give_feedback(actual_values,predicted_values)

def calculate_accuracy(actual, predicted):
    # Calculate accuracy as a percentage of how close actual is to predicted
    accuracy = 100 * (1 - abs(actual - predicted) / predicted)
    if(accuracy<50):
        accuracy=100-50
    # Ensure accuracy is non-negative and add 100 if it's negative
    accuracy = max(accuracy, 0) + (100 if accuracy < 0 else 0)
    
    return accuracy

def calculate_accuracy_angle(actual, predicted):
    if(actual>predicted):
        k=actual
        actual=predicted
        predicted=k
    accuracy = 100 * (actual / predicted)
    return accuracy

def give_feedback(actual_values, predicted_values):
    # Check if the input DataFrames have the same structure
    if not actual_values.columns.equals(predicted_values.columns):
        st.error("Input DataFrames must have the same columns.")
        return

    # Create a Streamlit figure to display the pie chart
    st.title("Feedback")

    # Display accuracy for each column
    st.subheader("Performance Metrics Feedback")
    sum=0
    for column in actual_values.columns:
        actual = actual_values[column].iloc[0]
        predicted = predicted_values[column].iloc[0]

        # Create a subheader with the column name
        st.subheader(column)

        # Display the accuracy
        # if column == 'Vertical_upright_angle':
        accuracy = calculate_accuracy_angle(actual, predicted)
        # else:
            # accuracy = calculate_accuracy(actual, predicted)

        st.text(f"Accuracy: {accuracy:.2f}%")

        # Provide instructions for each metric
        if column == 'Vertical_upright_angle':
            st.write("Vertical Upright Angle:")
            st.write("Instructions for Improvement:")
            if accuracy < 90:
                st.write("Work on maintaining a more upright posture.")
            else:
                st.write("Your posture is good; continue to keep it upright.")

        elif column == 'Stride':
            st.write("Stride Length:")
            st.write("Instructions for Improvement:")
            if accuracy < 90:
                st.write("Work on improving your stride length.")
            else:
                st.write("Your stride length is good; maintain it.")
        sum+=accuracy
    sum=sum/4
    st.text(f"OverAll Accuracy is: {sum}%")
    st.write("Actual values are:")
    st.dataframe(actual_values)
    st.write("Ideal values are:")
    st.dataframe(predicted_values)
    print(sum)

        # ... (similar blocks for other columns)

        # Create a figure and plot the pie chart with a title
        # plt.figure(figsize=(6, 6))
        # plt.pie([accuracy, 100 - accuracy], labels=['Accuracy', 'Error'], autopct='%1.1f%%')
        # plt.title(f"{column} Accuracy")
        # st.pyplot(plt.gcf())


def train_multivariate_regression_model(model,height, weight,gender):
    # Read the dataset
    df = pd.read_csv('datasetathlete.csv')
    # Normalize the input features (Height, Weight)
    scaler = MinMaxScaler()
    df[['Height', 'Weight']] = scaler.fit_transform(df[['Height', 'Weight']])
    if(gender =='Male'):
        gender=1
    else:
        gender=0
    #Define a function to predict values
    def predict_values(height_cm, weight_kg, gender):
        # Normalize the input data
        input_data = pd.DataFrame({'Height': [height_cm], 'Weight': [weight_kg], 'Gender': [gender]})
        input_data[['Height', 'Weight']] = scaler.transform(input_data[['Height', 'Weight']])

        # Predict the values
        predicted_values = model.predict(input_data)
        print(predicted_values)
        return predicted_values
    predicted_values = predict_values(height, weight,gender)
    return predicted_values



actual_values=[]
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)





def calculate_pelvic_tilt(keypoints_with_scores):
    # Define the indices for the relevant keypoints (left hip, right hip, and spine)
    left_hip_index = 11  # Adjust the keypoint indices based on your model
    right_hip_index = 12
    spine_index = 8

    # Extract the y-coordinates of the left hip, right hip, and spine keypoints
    left_hip_y = keypoints_with_scores[0,0, left_hip_index, 0]  
    right_hip_y = keypoints_with_scores[0,0, right_hip_index, 0] 
    spine_y = keypoints_with_scores[0,0, spine_index, 0] 

    if None not in [left_hip_y, right_hip_y, spine_y]:
        # Calculate pelvic tilt angle (in degrees)
        pelvic_tilt_angle = np.arctan2(right_hip_y - left_hip_y, spine_y - 0.5) * (180.0 / np.pi)
        return pelvic_tilt_angle
    else:
        return None


def calculate_vertical_upright_angle(keypoints, left_shoulder_index, right_shoulder_index):
    # Extract the y-coordinates of the left and right shoulder keypoints
    left_shoulder_y = keypoints[0, 0, left_shoulder_index, 0]
    right_shoulder_y = keypoints[0, 0, right_shoulder_index, 0]

    # Calculate the vertical upright angle (in degrees)
    vertical_upright_angle = np.arctan2(right_shoulder_y - left_shoulder_y, 1)  # Assuming the vertical axis is (0,1)
    vertical_upright_angle = np.degrees(vertical_upright_angle)

    return vertical_upright_angle

def visualize_vertical_upright_angle(vertical_upright_angles):
    time_axis = np.arange(len(vertical_upright_angles))

    # Create a Streamlit figure to display the graph
    st.pyplot(plt.figure(figsize=(10, 4)))

    # Plot the vertical upright angles over time
    plt.plot(time_axis, vertical_upright_angles)
    plt.xlabel('Frame')
    plt.ylabel('Vertical Upright Angle (degrees)')
    plt.title('Vertical Upright Angle Over Time')

    # Calculate and display the maximum vertical upright angle
    max_upright_angle = max(vertical_upright_angles)
    max_frame = vertical_upright_angles.index(max_upright_angle)

    plt.axvline(max_frame, color='red', linestyle='--', label=f'Max Angle: {max_upright_angle:.2f} degrees at frame {max_frame}')

    plt.legend()
    st.pyplot(plt)
    return max_upright_angle


def visualize_stride_pattern(stride_lengths, left_ankle_x_positions):
    # Calculate statistics
    avg_stride = np.mean(stride_lengths)
    # Calculate the mean deviation
    mean_deviation = np.mean(np.abs(stride_lengths - avg_stride))

    time_axis = np.arange(len(left_ankle_x_positions))

    # Create a Streamlit figure to display the graph
    st.pyplot(plt.figure(figsize=(10, 4)))

    # Plot the left ankle x-coordinate positions (stride pattern)
    plt.plot(time_axis, left_ankle_x_positions)
    plt.xlabel('Frame')
    plt.ylabel('X-coordinate of Left Ankle')
    plt.title('Left Ankle Stride Pattern')

    st.pyplot(plt)

    return mean_deviation
    


def visualize_pelvic_tilt(pelvic_tilt_angles):
    time_axis = range(len(pelvic_tilt_angles))
    
    # Create a Streamlit figure to display the graph
    st.pyplot(plt.figure(figsize=(10, 4)))

    # Plot pelvic tilt angles over time
    plt.plot(time_axis, pelvic_tilt_angles, label='Pelvic Tilt Angle')
    plt.xlabel('Frame')
    plt.ylabel('Pelvic Tilt Angle (degrees)')
    plt.title('Pelvic Tilt Over Time')

    # Calculate and display the average pelvic tilt angle
    avg_pelvic_tilt = sum(pelvic_tilt_angles) / len(pelvic_tilt_angles)
    plt.axhline(avg_pelvic_tilt, color='red', linestyle='--', label=f'Avg Pelvic Tilt: {avg_pelvic_tilt:.2f} degrees')

    plt.legend()
    st.pyplot(plt)

    return avg_pelvic_tilt


def calculate_angles(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cosine_theta = dot_product / magnitude_product
    angle_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg



def calculate_knee_angles(keypoints):
    # Define keypoint indices for hip, knee, and ankle
    left_hip_idx = 11
    left_knee_idx = 13
    left_ankle_idx = 15
    right_hip_idx = 12
    right_knee_idx = 14
    right_ankle_idx = 16

    # Extract keypoint coordinates and confidence scores
    keypoints = keypoints[0, 0]  # Assuming a single image and single person

    # Extract coordinates (yx) of the keypoints
    left_hip = keypoints[left_hip_idx, :2]
    left_knee = keypoints[left_knee_idx, :2]
    left_ankle = keypoints[left_ankle_idx, :2]
    right_hip = keypoints[right_hip_idx, :2]
    right_knee = keypoints[right_knee_idx, :2]
    right_ankle = keypoints[right_ankle_idx, :2]

    # Calculate the knee angles using the arctan2 function
    left_knee_angle = math.degrees(math.atan2(left_knee[1] - left_hip[1], left_knee[0] - left_hip[0]) - math.atan2(left_ankle[1] - left_knee[1], left_ankle[0] - left_knee[0]))
    right_knee_angle = math.degrees(math.atan2(right_knee[1] - right_hip[1], right_knee[0] - right_hip[0]) - math.atan2(right_ankle[1] - right_knee[1], right_ankle[0] - right_knee[0]))

    return left_knee_angle, right_knee_angle



def visualize_knee_angles(left_knee_angles, right_knee_angles):
    time = range(len(left_knee_angles))

    # Create a Streamlit figure to display the graphs
    st.pyplot(plt.figure(figsize=(10, 5)))

    # Plot left knee angles
    plt.subplot(211)
    plt.plot(time, left_knee_angles, label='Left Knee Angle')
    plt.xlabel('Time')
    plt.ylabel('Knee Angle (degrees)')
    plt.title('Left Knee Angle Over Time')
    plt.grid(True)
    plt.legend()

    # Plot right knee angles
    plt.subplot(212)
    plt.plot(time, right_knee_angles, label='Right Knee Angle', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Knee Angle (degrees)')
    plt.title('Right Knee Angle Over Time')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt)


def calculate_elbow_flaring(keypoints, left_elbow_index, left_shoulder_index, left_hip_index, right_elbow_index, right_shoulder_index, right_hip_index):
    # Extract the yx coordinates of the keypoints
    left_elbow = keypoints[0, 0, left_elbow_index, :2]
    left_shoulder = keypoints[0, 0, left_shoulder_index, :2]
    left_hip = keypoints[0, 0, left_hip_index, :2]
    right_elbow = keypoints[0, 0, right_elbow_index, :2]
    right_shoulder = keypoints[0, 0, right_shoulder_index, :2]
    right_hip = keypoints[0, 0, right_hip_index, :2]

    # Calculate the vectors representing the upper arms
    left_upper_arm_vector = left_shoulder - left_elbow
    right_upper_arm_vector = right_shoulder - right_elbow

    # Calculate the angles between the upper arms and the vertical axis for both elbows
    left_elbow_flare_angle = calculateangle(left_upper_arm_vector, np.array([0, -1]))
    right_elbow_flare_angle = calculateangle(right_upper_arm_vector, np.array([0, -1]))

    return left_elbow_flare_angle, right_elbow_flare_angle

def calculateangle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cosine_theta = dot_product / magnitude_product
    angle_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def visualize_elbow_flaring(left_angles, right_angles):
    time_axis = range(len(left_angles))

    # Create a Streamlit figure to display the graph for left elbow flaring angle
    st.pyplot(plt.figure(figsize=(10, 4)))

    # Plot the left elbow flaring angle over time
    plt.plot(time_axis, left_angles, label='Left Elbow Flaring Angle')
    plt.xlabel('Frame')
    plt.ylabel('Elbow Flaring Angle (degrees)')
    plt.title('Left Elbow Flaring Angle Over Time')
    plt.legend()

    # Calculate and display the average left elbow flaring angle
    avg_left_angle = np.mean(left_angles)
    st.write(f"Average Left Elbow Flaring Angle: {avg_left_angle:.2f} degrees")

    # Create a Streamlit figure to display the graph for right elbow flaring angle
    st.pyplot(plt.figure(figsize=(10, 4)))

    # Plot the right elbow flaring angle over time
    plt.plot(time_axis, right_angles, label='Right Elbow Flaring Angle')
    plt.xlabel('Frame')
    plt.ylabel('Elbow Flaring Angle (degrees)')
    plt.title('Right Elbow Flaring Angle Over Time')
    plt.legend()

    # Calculate and display the average right elbow flaring angle
    avg_right_angle = np.mean(right_angles)
    st.write(f"Average Right Elbow Flaring Angle: {avg_right_angle:.2f} degrees")

    return avg_left_angle, avg_right_angle

def plot_vertical_upright_angles(vertical_upright_angles):
    time_axis = np.arange(len(vertical_upright_angles))

    # Create a Streamlit figure to display the graph
    st.pyplot(plt.figure(figsize=(10, 4)))

    # Plot the vertical upright angles over time
    plt.plot(time_axis, vertical_upright_angles)
    plt.xlabel('Frame')
    plt.ylabel('Vertical Upright Angle (degrees)')
    plt.title('Vertical Upright Angle Over Time')
