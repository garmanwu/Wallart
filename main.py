import streamlit as st
import torch
from PIL import Image
import os
import json
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from openai import OpenAI
import base64
from io import BytesIO

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model
model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K").to(device)
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image_gpt4o(image):
    base64_image = encode_image(image)
    
    client = OpenAI(
        base_url="https://api.gptsapi.net/v1",
        api_key="api_key"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Assume you are an interior designer responsible for indoor soft decoration matching. 1. Based on the interior design style of the uploaded photo, provide matching wall painting style in English. 2. Wall painting style includes but are not limited to style, color, theme elements, etc. 3. Output only the answer, no transition language before or after."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    
    return response.choices[0].message.content

def extract_features(text):
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features.cpu().numpy()

def load_json_features(directory):
    features = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                features.append(np.array(data['features']))
                filenames.append(data['file_name'])
    return np.array(features), filenames

def compute_similarity(query_features, database_features):
    return np.dot(database_features, query_features.T).squeeze()

st.title("Indoor Painting Assistant")

uploaded_file = st.file_uploader("Upload a living room photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Processing image..."):
        # GPT-4O analysis
        gpt4o_response = analyze_image_gpt4o(image)
        st.subheader("Painting Characteristics")
        st.write(gpt4o_response)
        
        # Feature extraction
        query_features = extract_features(gpt4o_response)
        st.write("Query features extracted.")
        
        # Load database features
        database_features, filenames = load_json_features("clip")
        st.write(f"Loaded {len(filenames)} database image features.")
        
        # Compute similarity
        similarities = compute_similarity(query_features, database_features)
        st.write("Similarity computation completed.")
        
        # Get TOP 10 results
        top_10_indices = np.argsort(similarities)[-10:][::-1]
        
        st.subheader("TOP 10 Similar Paintings")
        for i, idx in enumerate(top_10_indices):
            score = similarities[idx]
            filename = filenames[idx]
            image_path = os.path.join("wallpaint", filename)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_path, caption=f"Rank {i+1}", use_column_width=True)
            with col2:
                st.write(f"Filename: {filename}")
                st.write(f"Similarity Score: {score:.4f}")

st.write("Processing completed.")