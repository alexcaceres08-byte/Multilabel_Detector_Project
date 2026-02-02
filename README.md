#  Multi-Label AI Detector with Active Learning & MLOps

An intelligent web application capable of detecting multiple objects in the same image (e.g., Plane, Car, Boat).

This project goes beyond simple detection: it implements an **Active Learning Pipeline**. Users can correct the AI predictions directly from the web, which triggers automatic retraining. The entire lifecycle and improvement of the model is monitored in real-time using **MLflow**.

##  Key Features

-  **Multi-Label Detection:** Identifies multiple objects in the same scene with confidence bars.
-  **Active Learning Cycle:** Did the AI make a mistake? Correct the label on the web and the system retrains instantly with the new data.
-  **MLOps and Monitoring:** Complete integration with **MLflow** to log metrics (Precision, Loss) and visualize the model's evolution graphically.
-  **Modern Technology:** Built with **FastAPI** (Asynchronous Backend), **FastAI** (Deep Learning), and **Bootstrap 5** (Responsive Frontend).

##  Project Structure

- `app/`: Source code of the API, training logic, and HTML templates.
- `app/mlruns/`: Local database where MLflow stores experiments and charts.
- `data/`: Contains the image dataset and `labels.csv` file.
- `notebooks/`: Initial experiments in Jupyter.

##  Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd fastapi-multilabel

### 2. Install Dependencies
Make sure you have Python installed.
pip install -r requirements.txt

### 3. Run the Application
You need to open two terminal windows to run the full system.

**Terminal 1: The Web App (FastAPI)**
Runs the AI interface on port 8000.

```bash
cd app
uvicorn main:app --reload
```

Access: [http://127.0.0.1:8000](http://127.0.0.1:8000)

**Terminal 2: The MLOps Dashboard (MLflow)**
Runs the monitoring dashboard on port 5050.

```bash
cd app
mlflow ui --port 5050 --backend-store-uri ./mlruns
```

Access: [http://127.0.0.1:5050](http://127.0.0.1:5050)

### How to Use

1. **Upload:** Go to the web app and upload an image.
2. **Analyze:** Click "Analyze Image" to see the AI's predictions.
3. **Teach (Active Learning):**
   - If the prediction is wrong or missing objects, check the correct boxes (Plane, Car, Ship).
   - Click "Re-train Model".

The system will save the new data, fine-tune the neural network, and log the results.

4. **Monitor:** Open the MLflow dashboard to see the accuracy graph rising as you teach the model!