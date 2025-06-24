from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, List, Any
import time
from datetime import datetime
#  new comment
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant disease detection using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
MIN_CONFIDENCE_THRESHOLD = 0.3  # Flag predictions below 30%
IMAGE_SIZE = (128, 128)  # Model input size

# Class names - ensure this matches your training data exactly
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    # Check if filename exists
    if not file.filename:
        return False
    
    # Check file extension
    extension = file.filename.split('.')[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False
    
    # Check content type
    if not file.content_type or not file.content_type.startswith('image/'):
        return False
    
    return True

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model prediction
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed image array ready for model prediction
        
    Raises:
        ValueError: If image preprocessing fails
    """
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (handles RGBA, grayscale, etc.)
        img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Image preprocessed successfully. Shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}")

def get_plant_info(prediction: str) -> Dict[str, Any]:
    """Get additional information about the predicted plant/disease"""
    plant_info = {
        "plant_type": prediction.split("___")[0].replace("_", " "),
        "condition": prediction.split("___")[1].replace("_", " "),
        "is_healthy": "healthy" in prediction.lower()
    }
    
    # Add severity level based on condition
    if plant_info["is_healthy"]:
        plant_info["severity"] = "None"
        plant_info["recommendation"] = "Plant appears healthy. Continue regular care."
    else:
        plant_info["severity"] = "Moderate"  # You can enhance this with more specific logic
        plant_info["recommendation"] = "Disease detected. Consider consulting agricultural expert."
    
    return plant_info

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Plant Disease Detection API...")
    if not load_model():
        logger.critical("Failed to load model. API will not function properly.")
    else:
        logger.info("API startup completed successfully")

# Health check endpoints
@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "message": "Plant Disease Detection API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "supported_classes": len(class_names),
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "model_input_size": IMAGE_SIZE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/classes")
async def get_supported_classes():
    """Get list of supported plant classes"""
    plants = {}
    for class_name in class_names:
        plant, condition = class_name.split("___")
        plant = plant.replace("_", " ")
        condition = condition.replace("_", " ")
        
        if plant not in plants:
            plants[plant] = []
        plants[plant].append(condition)
    
    return {
        "total_classes": len(class_names),
        "plants": plants
    }

# Main prediction endpoint
@app.post("/predict/")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction results
    """
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please try again later."
            )
        
        # Validate file
        if not validate_image_file(file):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid file format",
                    "supported_formats": list(ALLOWED_EXTENSIONS),
                    "received_filename": file.filename,
                    "received_content_type": file.content_type
                }
            )
        
        # Read and validate file size
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "File too large",
                    "max_size_mb": MAX_FILE_SIZE // (1024 * 1024),
                    "received_size_mb": round(file_size / (1024 * 1024), 2)
                }
            )
        
        # Check if file is empty
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Preprocess image
        try:
            processed_image = preprocess_image(contents)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Image processing failed: {str(e)}"
            )
        
        # Make prediction
        try:
            predictions = model.predict(processed_image, verbose=0)
            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            predicted_class = class_names[predicted_index]
            
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = []
        for idx in top_3_indices:
            top_predictions.append({
                "class": class_names[idx],
                "confidence": round(float(predictions[0][idx]), 4),
                "percentage": round(float(predictions[0][idx]) * 100, 2)
            })
        
        # Get plant information
        plant_info = get_plant_info(predicted_class)
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 3)
        
        # Prepare response
        response_data = {
            "success": True,
            "prediction": {
                "class": predicted_class,
                "confidence": round(confidence, 4),
                "percentage": round(confidence * 100, 2)
            },
            "plant_info": plant_info,
            "top_predictions": top_predictions,
            "file_info": {
                "filename": file.filename,
                "size_kb": round(file_size / 1024, 2),
                "content_type": file.content_type
            },
            "processing_time_seconds": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add warnings for low confidence
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            response_data["warning"] = {
                "message": f"Low confidence prediction ({confidence*100:.1f}%)",
                "threshold": f"{MIN_CONFIDENCE_THRESHOLD*100}%",
                "suggestions": [
                    "Ensure the image shows clear plant leaves",
                    "Check that lighting is adequate",
                    "Verify the plant type is supported",
                    "Try uploading a higher quality image",
                    "Make sure disease symptoms are visible"
                ]
            }
        
        # Log successful prediction
        logger.info(f"Prediction successful: {predicted_class} ({confidence:.3f}) in {processing_time}s")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred during prediction",
                "timestamp": datetime.now().isoformat()
            }
        )

# Batch prediction endpoint (bonus feature)
@app.post("/predict/batch/")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict diseases for multiple images
    
    Args:
        files: List of uploaded image files (max 5)
        
    Returns:
        JSON response with batch prediction results
    """
    if len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 files allowed per batch request"
        )
    
    results = []
    for i, file in enumerate(files):
        try:
            # Use the single prediction endpoint logic
            result = await predict_disease(file)
            results.append({
                "file_index": i,
                "filename": file.filename,
                "result": result
            })
        except HTTPException as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "error": e.detail
            })
    
    return {
        "batch_results": results,
        "total_files": len(files),
        "successful_predictions": sum(1 for r in results if "result" in r),
        "failed_predictions": sum(1 for r in results if "error" in r)
    }

# Error handlers
@app.exception_handler(413)
async def file_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "error": "File too large",
            "max_size_mb": MAX_FILE_SIZE // (1024 * 1024),
            "message": "Please upload a smaller image file"
        }
    )

@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "message": "Please check your request format and try again",
            "details": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
