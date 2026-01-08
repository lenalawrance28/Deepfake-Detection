import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set Streamlit page config
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODEL LOADING ---
# Use st.cache_resource to load the model only once.
@st.cache_resource
def load_detection_model():
    """
    Loads the pre-trained deepfake detection model.
    Make sure the 'cnn_model.h5' file is in the same directory.
    """
    try:
        model = tf.keras.models.load_model('cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure 'cnn_model.h5' is in the correct folder.")
        return None

model = load_detection_model()

# --- IMAGE PREPROCESSING ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    This model (MaanVad3r/DeepFake-Detector) expects 128x128 images.
    """
    # Resize to the model's expected input size
    size = (128, 128)
    img = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.asarray(img)
    
    # Normalize the image (if the model was trained on normalized images)
    # The search result 4.2 implies it's a standard CNN, so 0-1 scaling is safe.
    img_array = img_array / 255.0
    
    # Expand dimensions to create a "batch" of 1
    # Shape becomes (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- MAIN APP UI ---
st.title("🤖 Adversarial & Responsible AI Deepfake Detector")
st.markdown("""
Welcome to your Final Year Project application! 
This tool uses a pre-trained **Convolutional Neural Network (CNN)** to detect whether an image is real or a deepfake.

**Next steps for your project:**
1.  **Baseline Test:** Test this with real and fake images to see its baseline accuracy.
2.  **Adversarial Test:** Try to *fool* the model (e.g., add noise to an image).
3.  **Bias Test:** Test images across different demographics (see your "Responsible AI" brief).
""")

st.divider()

if model is not None:
    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Your Image")
        st.markdown("Upload an image file (jpg, png, jpeg) to classify.")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Open the uploaded image
            image = Image.open(uploaded_file)
            
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image
            processed_image = preprocess_image(image.convert('RGB')) # Ensure 3 channels

            # --- PREDICTION ---
            with st.spinner('Analyzing the image...'):
                try:
                    prediction = model.predict(processed_image)
                    score = prediction[0][0] # Get the single prediction value
                    
                    with col2:
                        st.header("Analysis Result")
                        
                        # The Hugging Face model (Result 4.2) is a binary classifier.
                        # We assume < 0.5 is 'REAL' and > 0.5 is 'FAKE'.
                        # You may need to *reverse* this (0 = FAKE, 1 = REAL) based on testing.
                        
                        if score > 0.5:
                            st.error(f"**Result: FAKE**", icon="❌")
                            st.progress(score)
                            st.markdown(f"Confidence Score: **{score*100:.2f}% FAKE**")
                        else:
                            st.success(f"**Result: REAL**", icon="✅")
                            st.progress(1.0 - score) # Show confidence for "REAL"
                            st.markdown(f"Confidence Score: **{(1.0 - score)*100:.2f}% REAL**")
                        
                        st.subheader("What does this mean?")
                        st.markdown(f"""
The model processed the image and returned a score of **{score:.4f}**. 
- Scores closer to **1.0** indicate the model believes it's **FAKE**.
- Scores closer to **0.0** indicate the model believes it's **REAL**.

For your report, this score is your key result. Your goal in an *adversarial attack* would be to take a FAKE image (e.g., score `0.9`) and make the model see it as REAL (e.g., score `0.1`).
""")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

    with col2:
        st.info("Upload an image on the left to see the analysis here.")

else:
    st.error("Model could not be loaded. Please check the console for errors and ensure 'cnn_model.h5' is in the same folder as 'app.py'.")