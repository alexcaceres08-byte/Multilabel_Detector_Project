# app/main.py
import os
import pathlib
import sys
import shutil

# --- 1. WINDOWS PATCH ---
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# --- 2. IMPORTS ---
from fastai.vision.all import *
import mlflow 

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# --- 3. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
DATA_PATH = BASE_DIR.parent / 'data'
IMAGES_PATH = DATA_PATH / 'images'
CSV_PATH = DATA_PATH / 'labels.csv'

# --- MLFLOW CONFIGURATION ---
mlruns_folder = BASE_DIR / "mlruns"
mlflow.set_tracking_uri(mlruns_folder.as_uri()) 
mlflow.set_experiment("Multilabel_Detector_Project")

# --- 4. FUNCTIONS ---
def get_image(row):
    return IMAGES_PATH / row['fname']

def get_labels(row):
    return row['labels'].split(' ')
obtener_imagen = get_image
obtener_etiquetas = get_labels
# =======================================================

def retrain_model_logic():
    print(f"--- 🎓 Saving data to: {mlruns_folder} ---")
    try:
        # Load CSV
        df = pd.read_csv(CSV_PATH)
        # Filter only existing images
        df = df[df['fname'].apply(lambda x: (IMAGES_PATH/x).exists())]
        
        dblock = DataBlock(
            blocks=(ImageBlock, MultiCategoryBlock),
            get_x=get_image,
            get_y=get_labels,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            item_tfms=Resize(460),
            batch_tfms=aug_transforms(size=224, min_scale=0.75)
        )
        
        dls = dblock.dataloaders(df, bs=8, num_workers=0)
        
        with mlflow.start_run(run_name="User-Retraining"):
            learn.dls = dls
            learn.fine_tune(1) 
            
            # --- DATA TYPE CORRECTION ---
            # Extract values from the last epoch
            values = learn.recorder.values[-1]
            train_loss = values[0]
            valid_loss = values[1]
            accuracy = values[2]
            
            # Convert to pure Python float so MLflow doesn't crash
            mlflow.log_metric("train_loss", float(train_loss))
            mlflow.log_metric("valid_loss", float(valid_loss))
            mlflow.log_metric("accuracy", float(accuracy))
            
            print(f"✅ Logged to MLflow -> Acc: {accuracy:.2%} (Saved as float)")

        learn.export(BASE_DIR / 'model.pkl')
        return "Model updated and logged to MLflow!"
    except Exception as e:
        print(f"Critical retraining error: {e}")
        return f"Error: {e}"

# --- 5. APP ---
app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

model_path = BASE_DIR / 'model.pkl'
try:
    learn = load_learner(model_path)
    print("✅ AI Brain loaded successfully")
except Exception as e:
    print(f"❌ Error loading the model: {e}")

# --- 6. ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = PILImage.create(io.BytesIO(img_bytes))
        
        # Get prediction
        pred_class, pred_idx, probabilities = learn.predict(img)
        
        results = {}
        for classname, prob in zip(learn.dls.vocab, probabilities):
            results[classname] = float(prob)
            
        # Returning 'confidence' instead of 'confianza'
        return {"confidence": results}
    except Exception as e:
        return {"error": str(e)}

@app.post("/teach")
async def teach(file: UploadFile = File(...), labels: str = Form(...)):
    try:
        # Create directory if it doesn't exist
        save_folder = IMAGES_PATH / "user_added"
        save_folder.mkdir(exist_ok=True)
        
        filename = f"new_{file.filename}"
        file_location = save_folder / filename
        
        # Save image file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Append to CSV
        rel_path = f"user_added/{filename}"
        line = f"\n{rel_path},{labels}"
        with open(CSV_PATH, "a") as f:
            f.write(line)
            
        # Trigger retraining
        message = retrain_model_logic()
        return {"status": "ok", "message": message}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
