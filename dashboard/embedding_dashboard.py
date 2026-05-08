import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image


# 1. Setup Model (Caching prevents reloading on every click)
@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")


model = load_model()


# 2. Mock Reference Data
# In a real app, you'd pre-calculate these and load a CSV/Parquet file.
@st.cache_data
def get_reference_data():
    np.random.seed(42)
    num_samples = 100
    # Simulating 2048-d ResNet embeddings
    ref_embeddings = np.random.randn(num_samples, 2048)
    filenames = [f"ref_image_{i}.jpg" for i in range(num_samples)]
    return ref_embeddings, filenames


ref_embeddings, ref_filenames = get_reference_data()

# --- Dashboard UI ---
st.title("🖼️ Image Embedding Cluster Map")
st.sidebar.header("Settings")
num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

uploaded_file = st.file_uploader(
    "Upload an image to see where it fits...", type=["jpg", "png", "jpeg"]
)

# 3. Processing
# Fit PCA on reference data
pca = PCA(n_components=2)
ref_2d = pca.fit_transform(ref_embeddings)

# Cluster reference data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(ref_2d)

# Create DataFrame for plotting
df = pd.DataFrame(
    {
        "x": ref_2d[:, 0],
        "y": ref_2d[:, 1],
        "label": [f"Cluster {c}" for c in clusters],
        "source": "Reference Library",
        "name": ref_filenames,
    }
)

# 4. Handle Uploaded Image
if uploaded_file:
    # Display the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Your Uploaded Image", width=200)

    # Preprocess and Extract Embeddings
    img_resized = img.resize((224, 224))
    x = keras_image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    new_embedding = model.predict(x)
    new_2d = pca.transform(new_embedding)  # Project into same PCA space

    # Add to DataFrame
    new_data = pd.DataFrame(
        {
            "x": [new_2d[0, 0]],
            "y": [new_2d[0, 1]],
            "label": ["Uploaded Image"],
            "source": "User Upload",
            "name": ["Your Image"],
        }
    )
    df = pd.concat([df, new_data], ignore_index=True)

# 5. Visualizing
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="label",
    symbol="source",
    hover_name="name",
    title="Latent Space Visualization (PCA)",
    color_discrete_map={"Uploaded Image": "#FFFFFF"},  # Make it stand out
)

# Style the "Uploaded Image" point specifically
fig.update_traces(
    marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
    selector=dict(name="Uploaded Image"),
)

st.plotly_chart(fig, use_container_width=True)
