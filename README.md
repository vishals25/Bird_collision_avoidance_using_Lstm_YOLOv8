# DATA DRIVEN BIRD MOVEMENT PREDICTION USING MACHINE LEARNING TECHNIQUES

### A MINI PROJECT REPORT

Submitted by

| NAME     | REGISTER NO. |
| -------- | ------------ |
| ANEESH V | 71762205006  |
| ANISH M  | 71762205007  |
| DINES S  | 71762205016  |
| VISHAL S | 71762205062  |

In partial fulfillment for the award of the degree of

**BACHELOR OF ENGINEERING**
in
**COMPUTER SCIENCE AND ENGINEERING**

**COIMBATORE INSTITUTE OF TECHNOLOGY**
*(Government Aided Autonomous Institution Affiliated to Anna University)*
COIMBATORE-641014

**ANNA UNIVERSITY: CHENNAI 600025**
**SEPTEMBER 2025**

---

### ABSTRACT

Predicting the movement patterns of birds is a critical task for ecological monitoring, aviation safety, and conservation efforts. This project presents a data-driven approach using deep learning to detect birds in video footage and predict their future flight paths. The system employs a two-stage process: first, a YOLOv8 (You Only Look Once) model is trained for robust bird detection in videos, extracting bounding box coordinates frame-by-frame. The coordinates of detected birds are then compiled into time-series data. Second, a Long Short-Term Memory (LSTM) neural network is trained on these sequential data points to learn and predict the future trajectory of the birds. By analyzing sequences of past positions, the LSTM model forecasts the subsequent positions in the flight path. The results demonstrate the system's effectiveness in both detecting birds with high accuracy and predicting their movement with quantifiable precision, offering a valuable tool for real-world applications in wildlife monitoring and collision avoidance.

---

### TABLE OF CONTENTS

1. **INTRODUCTION**
   * 1.1 Importance of Bird Movement Prediction
   * 1.2 Challenges in Trajectory Forecasting
   * 1.3 Scope of the Study
2. **SYSTEM ARCHITECTURE**
   * 2.1 System Architecture Overview
   * 2.2 Stage 1: YOLOv8 for Bird Detection
   * 2.3 Stage 2: LSTM for Trajectory Prediction
3. **SYSTEM DESIGN AND IMPLEMENTATION**
   * 3.1 Module 1: Bird Detection and Data Extraction
   * 3.2 Module 2: Data Preparation for LSTM
   * 3.3 Module 3: LSTM Model Training and Prediction
   * 3.4 Module 4: Prediction Visualization
4. **RESULTS AND DISCUSSIONS**
   * 4.1 YOLOv8 Detection and Training Performance
   * 4.2 LSTM Prediction Accuracy
   * 4.3 Visualizing Predicted Flight Paths
5. **CONCLUSION AND FUTURE WORK**
   * 5.1 Conclusion
   * 5.2 Future Work

---

### CHAPTER 1

### INTRODUCTION

#### 1.1 Importance of Bird Movement Prediction

The study of avian movement is crucial for several domains. In ecology, it provides insights into migration patterns, foraging behavior, and habitat utilization. For aviation, accurate prediction of bird flight paths is essential for mitigating the risk of bird-aircraft collisions, which pose a significant threat to flight safety. This project focuses on developing an automated system that can analyze video feeds to predict bird movement in near real-time.

#### 1.2 Challenges in Trajectory Forecasting

Predicting the trajectory of birds is a complex task due to the erratic and dynamic nature of their flight. Factors such as wind, weather conditions, flocking behavior, and reactions to the environment introduce a high degree of non-linearity. Traditional models often fail to capture these complex temporal dependencies. Deep learning models, particularly Recurrent Neural Networks (RNNs) like LSTMs, are well-suited for this task as they can learn long-term dependencies from sequential data.

#### 1.3 Scope of the Study

This study outlines the development of a complete pipeline for bird trajectory prediction. The scope includes:

1. Training a state-of-the-art object detection model (YOLOv8) to identify and track birds in video frames.
2. Extracting and processing the spatial coordinates of detected birds to form time-series datasets.
3. Designing, training, and evaluating an LSTM-based regression model to predict the future coordinates of a bird based on its past movements.
4. Visualizing the predicted flight paths and evaluating the model's performance using standard regression metrics.

---

### CHAPTER 2

### SYSTEM ARCHITECTURE

#### 2.1 System Architecture Overview

The system is designed as a two-stage pipeline. The first stage processes raw video input to generate structured data, and the second stage uses this data for prediction. The output is a video with visualized predictions.

* **Stage 1: Detection & Data Extraction:** A pre-trained YOLOv8 model analyzes each video frame to detect birds. The bounding box coordinates (x1, y1, x2, y2) for each detected bird are extracted and stored in a CSV file along with the frame number.
* **Stage 2: Prediction & Visualization:** The center coordinates from the CSV are used to train an LSTM model. This trained model then processes detection sequences from a test video in real-time to predict future flight paths, which are visualized as arrows on the final output video.

#### 2.2 Stage 1: YOLOv8 for Bird Detection

You Only Look Once (YOLO) is a highly efficient, state-of-the-art object detection algorithm. We use YOLOv8, which is known for its excellent balance of speed and accuracy. The model is trained on a custom dataset of bird images to specialize it for this task.

* **Input:** Video file (e.g., `test2.mp4`).
* **Process:** The video is processed frame by frame. For each frame, the YOLO model identifies all birds and outputs their bounding boxes.
* **Output:** A CSV file (`bird_detections.csv`) containing the frame number, class ID, confidence score, and bounding box coordinates for every detection.

#### 2.3 Stage 2: LSTM for Trajectory Prediction

Long Short-Term Memory (LSTM) networks are a type of Recurrent Neural Network (RNN) capable of learning order dependence in sequence prediction problems. They are ideal for time-series forecasting due to their ability to remember patterns over long sequences.

* **Input:** The `bird_detections.csv` file generated by the YOLO stage.
* **Process:**
  1. The data is prepared by creating input sequences (15 past (x, y) coordinates) and corresponding output sequences (5 future (x, y) coordinates).
  2. An LSTM model is built with LSTM layers, Dropout for regularization, and Dense layers for output.
  3. The model is trained to minimize the Mean Squared Error (MSE) between its predicted coordinates and the actual future coordinates.
* **Output:** A trained model file (`bird_movement_predictor.h5`) capable of making future predictions.

---

### CHAPTER 3

### SYSTEM DESIGN AND IMPLEMENTATION

#### 3.1 Module 1: Bird Detection and Data Extraction

This module uses the `ultralytics` library to perform inference with a trained YOLOv8 model (`best.pt`). The script processes a video file, performs detection on each frame, and saves the bounding box coordinates, frame number, and other metadata into a structured CSV file for later use. This serves as the primary data source for the prediction module.

#### 3.2 Module 2: Data Preparation for LSTM

The script reads the generated `bird_detections.csv` and transforms the raw detection data into a format suitable for training a sequence model. It calculates the center point of each bounding box and then creates overlapping sequences. For this project, a sequence length of **15** past coordinate pairs is used to predict the next **5** future coordinate pairs. The data is then split into training (80%) and testing (20%) sets.

#### 3.3 Module 3: LSTM Model Training and Prediction

An LSTM model is defined using the TensorFlow and Keras libraries. The architecture is designed to handle sequential data and output a multi-step forecast.

* **Input Layer:** Expects a shape of (15, 2), corresponding to 15 time steps with 2 features (x, y).
* **LSTM Layers:** Two LSTM layers (with 64 and 32 units respectively) are used to capture temporal patterns. Dropout layers (30%) are included after each LSTM layer to prevent overfitting.
* **Dense Layers:** A Dense layer with 32 units and a ReLU activation function processes the features from the LSTM layers.
* **Output Layer:** A final Dense layer outputs 10 values, which are then reshaped into the desired output shape of (5, 2), representing the 5 predicted future coordinate pairs.

The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss function, and trained for 100 epochs.

#### 3.4 Module 4: Prediction Visualization

The final part of the implementation involves using the trained LSTM model (`bird_movement_predictor.h5`) to generate and visualize predictions on a video. The script reads a video and its corresponding detection CSV. For each frame, it maintains a history of the bird's recent positions. Once the history reaches the required sequence length (15), it feeds this data into the LSTM model to get a prediction for the next 5 steps. The first predicted point is then used to draw a green arrow on the video frame, indicating the bird's likely direction of movement.

---

### CHAPTER 4

### RESULTS AND DISCUSSIONS

#### 4.1 YOLOv8 Detection and Training Performance

The YOLOv8 model was successfully trained for 90 epochs on the custom bird dataset. The training history shows a consistent decrease in box, class, and dfl loss for both the training and validation sets, indicating that the model was learning effectively without significant overfitting. The key performance metrics, such as precision, recall, and mean Average Precision (mAP), all showed strong improvement and stabilization over the course of the training.

**Table 4.1: YOLOv8 Training History Summary**

| epoch | metrics/precision(B) | metrics/recall(B) | metrics/mAP50(B) | metrics/mAP50-95(B) | val/box_loss | val/cls_loss |
| ----- | -------------------- | ----------------- | ---------------- | ------------------- | ------------ | ------------ |
| 1     | 0.63319              | 0.64403           | 0.62835          | 0.23615             | 2.0206       | 1.24399      |
| 10    | 0.92896              | 0.88788           | 0.93985          | 0.50169             | 1.45021      | 0.65169      |
| 20    | 0.96797              | 0.87064           | 0.95148          | 0.55081             | 1.32539      | 0.56496      |
| 30    | 0.96511              | 0.893             | 0.9576           | 0.58008             | 1.23587      | 0.50982      |
| 40    | 0.96511              | 0.91066           | 0.96419          | 0.61432             | 1.1634       | 0.4765       |
| 50    | 0.96377              | 0.92593           | 0.97035          | 0.62122             | 1.16457      | 0.46103      |
| 60    | 0.96359              | 0.92567           | 0.9673           | 0.62866             | 1.13543      | 0.44626      |
| 70    | 0.96462              | 0.92798           | 0.97069          | 0.6486              | 1.08648      | 0.431        |
| 80    | 0.96393              | 0.92387           | 0.97045          | 0.65396             | 1.06169      | 0.41875      |
| 90    | 0.96159              | 0.92593           | 0.96965          | 0.65735             | 1.05315      | 0.41679      |

The model achieved a final **mAP50-95 of 0.657**, indicating strong performance for the bird detection task.

#### 4.2 LSTM Prediction Accuracy

The LSTM model was evaluated on a held-out test set. The performance was measured by calculating the MSE, MAE, and the average Euclidean distance error (in pixels) for each of the 5 predicted future steps.

**Table 4.2: Prediction Accuracy on Test Data**

| prediction_step | mse      | mae   | avg_distance_error | std_distance_error |
| --------------- | -------- | ----- | ------------------ | ------------------ |
| 1               | 7990.60  | 51.37 | 83.00              | 95.35              |
| 2               | 8854.18  | 54.00 | 87.71              | 100.08             |
| 3               | 8965.77  | 55.97 | 90.33              | 98.85              |
| 4               | 9684.39  | 57.89 | 93.99              | 102.64             |
| 5               | 10305.01 | 60.66 | 98.32              | 104.61             |

As expected, the prediction error increases for steps further into the future, as uncertainty accumulates. However, the initial steps show a reasonably low pixel error, indicating that the model has successfully learned the short-term movement dynamics of the birds.

#### 4.3 Visualizing Predicted Flight Paths

A key result is the ability to visualize the model's predictions. The following plot compares the actual flight path with the one predicted by the model for a sample sequence from the test set.

**Figure 4.1: Actual vs. Predicted Path (Example)**

The visualization shows that the predicted path (in red) closely follows the general direction and curvature of the actual path (in green), confirming the model's predictive capability. This is further applied in the final script, which overlays these predictions as green arrows on the output video, demonstrating the practical application of the system.

---

### CHAPTER 5

### CONCLUSION AND FUTURE WORK

#### 5.1 Conclusion

This project successfully demonstrates a complete deep learning pipeline for bird detection and trajectory prediction. The YOLOv8 model proved effective for object detection, providing the necessary data for the subsequent time-series analysis. The LSTM network was able to learn complex temporal patterns from the coordinate data and make reasonable short-term predictions of flight paths. The results, validated both quantitatively with metrics and qualitatively through visualization, confirm the viability of this approach for automated wildlife monitoring and collision avoidance systems.

#### 5.2 Future Work

There are several avenues for extending this work:

1. **Multi-Object Tracking:** The current model simplifies the problem by tracking the most prominent bird in each frame. A more advanced system could incorporate multi-object tracking algorithms (e.g., DeepSORT) to handle scenarios with multiple birds simultaneously and maintain individual identities.
2. **Inclusion of More Features:** The model currently uses only (x, y) coordinates. Incorporating additional features derived from the sequence, such as velocity, acceleration, and changes in bounding box size, could provide richer context and improve prediction accuracy.
3. **Real-Time Implementation:** The pipeline could be optimized for real-time processing. This might involve using a more lightweight detection model (like YOLOv8-Nano), optimizing the code for faster inference, or deploying it on edge devices with GPU capabilities for on-site analysis.
4. **Longer-Term Prediction:** Exploring more advanced sequence models, such as Gated Recurrent Units (GRUs) or Transformers, could potentially improve the model's ability to make accurate predictions further into the future.
