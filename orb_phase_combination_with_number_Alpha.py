

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import time





def initialize_kalman_filter():
    dt = 1  # Time step (assuming consistent frame rate)

    # Create a KalmanFilter instance
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State Transition Matrix (F)
    kf.F = np.array([[1., 0., dt, 0.],
                     [0., 1., 0., dt],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]], dtype=float)

    # Measurement Function (H)
    kf.H = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.]], dtype=float)

    # Initial State (x)
    kf.x = np.array([0., 0., 0., 0.], dtype=float)

    # Covariance Matrix (P)
    kf.P *= 1000.0  # Ensure P is float

    # Process Noise Covariance (Q)
    kf.Q = np.eye(4, dtype=float) * 0.1

    # Measurement Noise Covariance (R)
    kf.R = np.eye(2, dtype=float) * 5.0

    # Adaptive filter parameter
    kf.alpha_R = 0.95  # Smoothing factor for R adaptation (0 < alpha_R < 1)

    return kf

def bound_covariance(cov_matrix, min_value=1e-4, max_value=1e4):
    cov_matrix = np.clip(cov_matrix, min_value, max_value)
    return cov_matrix

def adaptive_kalman_update(kf, z):
    """
    Perform the Kalman filter predict and update steps,
    and adaptively adjust the measurement noise covariance R.
    """
    # Store prior state for process noise adaptation
    kf.x_prior = kf.x.copy()
    kf.P_prior = kf.P.copy()

    # Predict step
    kf.predict()

    # Compute innovation (residual)
    y = z - np.dot(kf.H, kf.x)

    # Innovation covariance
    S = np.dot(kf.H, np.dot(kf.P, kf.H.T)) + kf.R

    # Kalman Gain
    K = np.dot(np.dot(kf.P, kf.H.T), np.linalg.inv(S))

    # Update state estimate and covariance
    kf.x += np.dot(K, y)
    kf.P = kf.P - np.dot(K, np.dot(kf.H, kf.P))

    # Adaptively update R using the smoothing factor
    innovation_cov = np.outer(y, y)
    kf.R = kf.alpha_R * kf.R + (1 - kf.alpha_R) * innovation_cov
    kf.R = bound_covariance(kf.R)

def estimate_shift_phase_correlation(img1, img2):
    # Ensure images are float32
    img1_float = np.float32(img1)
    img2_float = np.float32(img2)

    # Use phase correlation
    (shift_x, shift_y), _ = cv2.phaseCorrelate(img1_float, img2_float)

    # Invert shift direction (as per OpenCV convention)
    return -shift_x, -shift_y

def estimate_shift_orb(img1, img2):
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=500)

    # Detect keypoints and compute descriptors in both images
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Check if descriptors are found
    if des1 is None or des2 is None:
        return 0.0, 0.0, 0  # Return zero shift and zero matches

    # Create BFMatcher object with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    num_matches = len(matches)
    if num_matches == 0:
        return 0.0, 0.0, 0  # No matches found

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute shifts
    shifts = dst_pts - src_pts

    # Use median to reduce effect of outliers
    shift_x = np.median(shifts[:, 0])
    shift_y = np.median(shifts[:, 1])

    return shift_x, shift_y, num_matches

def estimate_shift_combined(img1, img2):
    # Estimate shifts using phase correlation
    shift_x_pc, shift_y_pc = estimate_shift_phase_correlation(img1, img2)

    # Estimate shifts using ORB
    shift_x_orb, shift_y_orb, num_matches = estimate_shift_orb(img1, img2)

    # Compute weights based on confidence
    # For example, weight based on number of matches in ORB
    # Adjust the weight scaling as needed
    weight_orb = num_matches
    weight_pc = 3  # Since phase correlation does not provide a confidence measure

    total_weight = weight_orb + weight_pc

    # Avoid division by zero
    if total_weight == 0:
        return 0.0, 0.0

    # Combine the shifts using weighted average
    shift_x_combined = (shift_x_pc * weight_pc + shift_x_orb * weight_orb) / total_weight
    shift_y_combined = (shift_y_pc * weight_pc + shift_y_orb * weight_orb) / total_weight

    return shift_x_combined, shift_y_combined

def real_time_processing(video_file, start_frame=None, end_frame=None):
    # Initialize video capture from file
    cap = cv2.VideoCapture(video_file)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return

    # Initialize Kalman Filter
    kf = initialize_kalman_filter()

    # Read the first frame
    ret, previous_frame = cap.read()
    if not ret:
        print("Failed to read from the video file.")
        cap.release()
        return

    # Convert to grayscale
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # Initialize lists to store data for plotting and timing
    measurements = []
    predictions = []
    mse_list = []
    prediction_times = []

    frame_count = 0

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break  # End of video

        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Start timing the prediction
        start_time = time.time()

        # Estimate shift between frames using combined method
        shift_x, shift_y = estimate_shift_combined(previous_frame_gray, current_frame_gray)

        # Measurement vector (z)
        z = np.array([shift_x, shift_y], dtype=float)

        # Adaptive Kalman filter update
        adaptive_kalman_update(kf, z)

        # Get the estimated position
        estimated_x, estimated_y = kf.x[0], kf.x[1]

        # End timing the prediction
        end_time = time.time()
        prediction_time = end_time - start_time
        prediction_times.append(prediction_time)

        # Store measurements and predictions for plotting
        measurements.append(z)
        predictions.append([estimated_x, estimated_y])

        # Compute MSE
        mse = np.mean((z - [estimated_x, estimated_y]) ** 2)
        mse_list.append(mse)

        # For visualization (optional)
        # Draw the measurement point in red
        meas_point = (int(300 + z[0]), int(300 + z[1]))
        cv2.circle(current_frame, meas_point, 5, (0, 0, 255), -1)

        # Draw the predicted point in green
        pred_point = (int(300 + estimated_x), int(300 + estimated_y))
        cv2.circle(current_frame, pred_point, 5, (0, 255, 0), -1)

        # Show the frame (optional)
        cv2.imshow('Frame', current_frame)

        # Prepare for next iteration
        previous_frame_gray = current_frame_gray.copy()

        frame_count += 1

        # Exit condition (press 'q' to exit early)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Convert lists to numpy arrays for plotting
    measurements = np.array(measurements)
    predictions = np.array(predictions)
    mse_list = np.array(mse_list)
    prediction_times = np.array(prediction_times)

    # Plot the results
    plot_results(measurements, predictions, mse_list, prediction_times, start_frame, end_frame)

    # Return the collected data
    return measurements, predictions, mse_list, prediction_times

def plot_results(measurements, predictions, mse_list, prediction_times, start_frame=None, end_frame=None):
    # Create a time axis based on the number of frames
    total_frames = len(measurements)
    time_axis = np.arange(total_frames)

    # Set default values for start_frame and end_frame if not provided
    if start_frame is None:
        start_frame = 0
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    # Ensure start_frame and end_frame are within valid range
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)

    # Slice the data arrays based on the specified frame range
    measurements_slice = measurements[start_frame:end_frame]
    predictions_slice = predictions[start_frame:end_frame]
    mse_list_slice = mse_list[start_frame:end_frame]
    prediction_times_slice = prediction_times[start_frame:end_frame]
    time_axis_slice = time_axis[start_frame:end_frame]

    # Plot measurements and predictions
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_axis_slice, measurements_slice[:, 0], 'r', label='Measurement X')
    plt.plot(time_axis_slice, predictions_slice[:, 0], 'g', label='Prediction X')
    plt.xlabel('Frame')
    plt.ylabel('Shift in X (pixels)')
    plt.title(f'Measurements vs Predictions - X (Frames {start_frame} to {end_frame})')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_axis_slice, measurements_slice[:, 1], 'r', label='Measurement Y')
    plt.plot(time_axis_slice, predictions_slice[:, 1], 'g', label='Prediction Y')
    plt.xlabel('Frame')
    plt.ylabel('Shift in Y (pixels)')
    plt.title(f'Measurements vs Predictions - Y (Frames {start_frame} to {end_frame})')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_axis_slice, mse_list_slice, 'b', label='Mean Squared Error')
    plt.xlabel('Frame')
    plt.ylabel('MSE')
    plt.title(f'Mean Squared Error over Time (Frames {start_frame} to {end_frame})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot Prediction Times
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis_slice, prediction_times_slice * 1000, 'm', label='Prediction Time')
    plt.xlabel('Frame')
    plt.ylabel('Time (milliseconds)')
    plt.title(f'Prediction Time per Frame (Frames {start_frame} to {end_frame})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Replace 'video.mp4' with the path to your video file
    video_file_path = '/Users/niloughazavi/Desktop/Desktop-Nilou/PhD/PhD_Courses/BIOENGR223A/Project/CA1/6.avi'

    # Specify the frame range you want to plot
    start_frame = 0
    end_frame = 1000

    # Run the processing and get the results
    measurements, predictions, mse_list, prediction_times = real_time_processing(video_file_path, start_frame, end_frame)
