# import streamlit as st
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
# from keras.preprocessing.image import img_to_array

# # Load the trained model
# model = load_model('model.h5')

# # Define class labels
# class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# # Set page config
# st.set_page_config(page_title="MRI Tumor Detection", layout="centered")

# # Title and description
# st.title("🧠 MRI Tumor Detection System")
# st.markdown("Upload an MRI image to detect if a brain tumor is present and its type.")

# # File uploader
# uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

# # Image prediction
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded MRI Image", use_column_width=True)

#     # Preprocess image
#     image = image.resize((128, 128))  # Resize to match model input
#     image_array = img_to_array(image) / 255.0
#     image_array = np.expand_dims(image_array, axis=0)

#     # Predict
#     prediction = model.predict(image_array)
#     predicted_class = np.argmax(prediction)
#     confidence = np.max(prediction) * 100
#     result_label = class_labels[predicted_class]

#     # Display result
#     st.success(f"🧬 Result: {'No Tumor' if result_label == 'notumor' else 'Tumor: ' + result_label.capitalize()}")
#     st.info(f"🧪 Confidence: {confidence:.2f}%")

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
# Optional: For Grad-CAM visualization
# from gradcam import get_gradcam_heatmap  # You'd need to implement or import this

# Load the trained model (add caching for faster reloads)
@st.cache_resource
def load_trained_model():
    return load_model('model.h5')

model = load_trained_model()
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# --- UI/UX Enhancements ---

# Language selection
lang = st.sidebar.selectbox(
    "Language / भाषा चुनें", 
    ("English", "हिन्दी")
)

# Page config
st.set_page_config(page_title="MRI Tumor Detection Pro", layout="centered", page_icon="🧠")

# Theming
st.markdown(
    """
    <style>
    .main {background-color: #f7f9fa;}
    .stButton>button {background-color: #4F8BF9; color: white;}
    </style>
    """, unsafe_allow_html=True
)

# Title and description
if lang == "English":
    st.title("🧠 MRI Tumor Detection Pro")
    st.write("Upload a brain MRI image to detect tumor presence and type. Enhanced with confidence scoring and optional visual explanations.")
else:
    st.title("🧠 एमआरआई ट्यूमर डिटेक्शन प्रो")
    st.write("ब्रेन एमआरआई इमेज अपलोड करें। ट्यूमर की उपस्थिति और प्रकार जानें। अब विश्वसनीयता स्कोर और विज़ुअल एक्सप्लेनेशन के साथ।")

# File uploader
file_label = "Choose an MRI image" if lang == "English" else "एमआरआई इमेज चुनें"
uploaded_file = st.file_uploader(file_label, type=["jpg", "jpeg", "png"])

# Sidebar: About and instructions
with st.sidebar.expander("ℹ️ About / जानकारी"):
    if lang == "English":
        st.markdown("""
        - **Multi-class tumor detection**: pituitary, glioma, meningioma, or no tumor.
        - **Confidence score** for every prediction.
        - **Optional Grad-CAM visualization** for interpretability.
        - **Bilingual interface**: English & Hindi.
        """)
    else:
        st.markdown("""
        - **मल्टी-क्लास ट्यूमर डिटेक्शन**: पिट्यूटरी, ग्लियोमा, मेनिंजियोमा या नो ट्यूमर।
        - **हर प्रेडिक्शन के लिए विश्वसनीयता स्कोर**।
        - **वैकल्पिक ग्रैड-कैम विज़ुअलाइजेशन**।
        - **द्विभाषी इंटरफेस**: इंग्लिश और हिंदी।
        """)

# # Image prediction
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded MRI" if lang == "English" else "अपलोडेड एमआरआई", use_column_width=True)

#     # Preprocess image
#     image_resized = image.resize((128, 128))
#     image_array = img_to_array(image_resized) / 255.0
#     image_array = np.expand_dims(image_array, axis=0)

#     # Predict
#     prediction = model.predict(image_array)
#     predicted_class = np.argmax(prediction)
#     confidence = np.max(prediction) * 100
#     result_label = class_labels[predicted_class]

#     # Show results
#     if result_label == "notumor":
#         result_text = "🧬 Result: No Tumor Detected" if lang == "English" else "🧬 परिणाम: कोई ट्यूमर नहीं मिला"
#         st.success(result_text)
#     else:
#         result_text = f"🧬 Result: Tumor Detected - {result_label.capitalize()}" if lang == "English" else f"🧬 परिणाम: ट्यूमर मिला - {result_label.capitalize()}"
#         st.error(result_text)
#     st.info(f"🧪 Confidence: {confidence:.2f}%" if lang == "English" else f"🧪 विश्वसनीयता: {confidence:.2f}%")

#     # Optional: Grad-CAM visualization for interpretability
    # if st.checkbox("Show Visual Explanation (Grad-CAM)" if lang == "English" else "विज़ुअल एक्सप्लेनेशन देखें (Grad-CAM)"):
    #     heatmap = get_gradcam_heatmap(model, image_array, predicted_class)
    #     st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)

    # Downloadable report
#     if st.button("Download Report" if lang == "English" else "रिपोर्ट डाउनलोड करें"):
#         report = f"""
#         MRI Tumor Detection Report
#         Result: {result_label}
#         Confidence: {confidence:.2f}%
#         """
#         st.download_button(
#             label="Download",
#             data=report,
#             file_name="tumor_detection_report.txt"
#         )

# else:
#     if lang == "English":
#         st.info("Please upload an MRI image to begin.")
#     else:
#         st.info("कृपया एमआरआई इमेज अपलोड करें।")

import matplotlib.pyplot as plt

# ... [Your previous code]

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded MRI" if lang == "English" else "अपलोडेड एमआरआई", use_column_width=True)

#     # Preprocess image
#     image_resized = image.resize((128, 128))
#     image_array = img_to_array(image_resized) / 255.0
#     image_array = np.expand_dims(image_array, axis=0)

#     # Predict
#     prediction = model.predict(image_array)[0]
#     predicted_class = np.argmax(prediction)
#     confidence = np.max(prediction) * 100
#     result_label = class_labels[predicted_class]

#     # Show results
#     if result_label == "notumor":
#         result_text = "🧬 Result: No Tumor Detected" if lang == "English" else "🧬 परिणाम: कोई ट्यूमर नहीं मिला"
#         st.success(result_text)
#     else:
#         result_text = f"🧬 Result: Tumor Detected - {result_label.capitalize()}" if lang == "English" else f"🧬 परिणाम: ट्यूमर मिला - {result_label.capitalize()}"
#         st.error(result_text)
#     st.info(f"🧪 Confidence: {confidence:.2f}%" if lang == "English" else f"🧪 विश्वसनीयता: {confidence:.2f}%")

#     # --- Graph/Ratio Section ---
#     st.subheader("Prediction Probabilities" if lang == "English" else "भविष्यवाणी संभावनाएँ")
#     fig, ax = plt.subplots()
#     bars = ax.bar(class_labels, prediction * 100, color=['#4F8BF9', '#F9A34F', '#4FF9A3', '#F94F8B'])
#     ax.set_ylabel('Probability (%)' if lang == "English" else 'संभावना (%)')
#     ax.set_ylim(0, 100)
#     ax.set_title('Model Confidence by Class' if lang == "English" else 'प्रत्येक वर्ग के लिए मॉडल की विश्वसनीयता')
#     for bar, prob in zip(bars, prediction * 100):
#         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{prob:.1f}%", ha='center', va='bottom', fontsize=10)
#     st.pyplot(fig)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI" if lang == "English" else "अपलोडेड एमआरआई", use_column_width=True)

    # Preprocess image
    image_resized = image.resize((128, 128))
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0]
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    result_label = class_labels[predicted_class]

    # Show results
    if result_label == "notumor":
        result_text = "🧬 Result: No Tumor Detected" if lang == "English" else "🧬 परिणाम: कोई ट्यूमर नहीं मिला"
        st.success(result_text)
    else:
        result_text = f"🧬 Result: Tumor Detected - {result_label.capitalize()}" if lang == "English" else f"🧬 परिणाम: ट्यूमर मिला - {result_label.capitalize()}"
        st.error(result_text)
    st.info(f"🧪 Confidence: {confidence:.2f}%" if lang == "English" else f"🧪 विश्वसनीयता: {confidence:.2f}%")
   
    if st.button("Download Report" if lang == "English" else "रिपोर्ट डाउनलोड करें"):
        report = f"""
        MRI Tumor Detection Report
        Result: {result_label}
        Confidence: {confidence:.2f}%
        """
        st.download_button(
            label="Download",
            data=report,
            file_name="tumor_detection_report.txt"
        )

    # --- ADD CUSTOM HTML SECTION HERE ---
    st.markdown("""
        <div style='background-color:#e6f2ff;padding:15px 10px 10px 10px;border-radius:10px;margin-bottom:15px;'>
            <h3 style='color:#1a237e;text-align:center;'>Prediction Probabilities</h3>
            <p style='text-align:center;color:#333;'>See the model's confidence for each tumor class below.</p>
        </div>
    """, unsafe_allow_html=True)
    st.subheader("Prediction Probabilities" if lang == "English" else "भविष्यवाणी संभावनाएँ")
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(class_labels, prediction * 100, color=['#4F8BF9', '#F9A34F', '#4FF9A3', '#F94F8B'], width=0.6, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Probability (%)' if lang == "English" else 'संभावना (%)', fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_title('Model Confidence by Class' if lang == "English" else 'प्रत्येक वर्ग के लिए मॉडल की विश्वसनीयता', fontsize=16, pad=15)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    for bar, prob in zip(bars, prediction * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{prob:.1f}%", ha='center', va='bottom', fontsize=13, color='#333')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    # Define tumor details for report

# --- Problem Detail Section (Graph ke baad) ---
    if result_label == "notumor":
        info_text = (
        "<div style='background-color:#e0f7fa;padding:10px;border-radius:8px;margin-top:20px;'>"
        "<h4 style='color:#00798b;'>No Tumor Detected</h4>"
        "<p>The MRI scan does not show any signs of a brain tumor.</p>"
        "</div>"
        if lang == "English" else
        "<div style='background-color:#e0f7fa;padding:10px;border-radius:8px;margin-top:20px;'>"
        "<h4 style='color:#00793b;'>कोई ट्यूमर नहीं मिला</h4>"
        "<p>एमआरआई स्कैन में किसी भी ब्रेन ट्यूमर का संकेत नहीं है।</p>"
        "</div>"
        )
    else:
    # Example details for detected tumor type
        tumor_details = {
        "pituitary": {
            "en": "<b>Pituitary Tumor:</b> This type affects the pituitary gland and may cause hormonal imbalance.",
            "hi": "<b>पिट्यूटरी ट्यूमर:</b> यह पिट्यूटरी ग्रंथि को प्रभावित करता है और हार्मोनल असंतुलन का कारण बन सकता है।"
        },
        "glioma": {
            "en": "<b>Glioma:</b> Glioma is a type of tumor that starts in the glial "
            "cells of the brain or spinal cord. Glial cells support and protect neurons (nerve cells),"
            " and gliomas are the most common type of primary brain tumor."
            "🧾 Symptoms (depend on tumor location & size) 1.Headaches ,2.Seizures,3.Nausea or vomiting,4Vision or speech problems, 5.Weakness or numbness in limbs , 6.Personality or cognitive changes"
            "💊 Treatment Options 1.Surgery (to remove as much tumor as possible), 2.Radiation therapy, 2.1 Chemotherapy (e.g., Temozolomide) , 3.Targeted therapy (based on tumor genetics) ,4.Supportive care (to manage symptoms)",
            "hi": "<b>ग्लियोमा:</b> 🧾 Symptoms (depend on tumor location & size) 1.Headaches ,2.Seizures,3.Nausea or vomiting,4Vision or speech problems, 5.Weakness or numbness in limbs , 6.Personality or cognitive changes",
            "hi": "<b>ग्लियोमा:</b> 💊 Treatment Options 1.Surgery (to remove as much tumor as possible), 2.Radiation therapy, 2.1 Chemotherapy (e.g., Temozolomide) , 3.Targeted therapy (based on tumor genetics) ,4.Supportive care (to manage symptoms)"
      
        },
        "meningioma": {
            "en": "<b>Meningioma:</b> Usually a slow-growing tumor from the meninges; often benign but can cause pressure symptoms.",
            "hi": "<b>मेनिंजियोमा:</b> आमतौर पर यह मेनिंजेस से उत्पन्न धीमी गति से बढ़ने वाला ट्यूमर है; अक्सर सौम्य होता है लेकिन दबाव के लक्षण दे सकता है।"
            }
        
        }
        # if st.button("Download Report" if lang == "English" else "रिपोर्ट डाउनलोड करें"):
        #     description = tumor_details[result_label]["en"] if lang == "English" else tumor_details[result_label]["hi"]
        # report = f"""MRI Tumor Detection Report

        # Result: {result_label.capitalize()}
        # Confidence: {confidence:.2f}%

        # Problem Details:
        # {description}
        # """
        # st.download_button(
        #     label="Download",
        #     data=report,
        #     file_name="tumor_detection_report.txt"
        # )
        tumor_info = tumor_details.get(result_label, {"en": "Tumor detected.", "hi": "ट्यूमर मिला।"})
        info_text = (
            # f"<div style='background-color:#4682b4;padding:10px;border-radius:8px;margin-top:20px;'>"
            f"<div style='background-color: #4a90e2;padding: 20px;border-radius: 12px;margin-top: 20px;color:white;font-family: Arial, sans-serif;font-size: 16px;line-height: 1.6;height: 300px;box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);overflow-y: auto;'>"
            f"<h4 style='color:#4682b5;'>Detected Tumor: {result_label.capitalize()}</h4>"
            f"<p>{tumor_info['en'] if lang == 'English' else tumor_info['hi']}</p>"
            "</div>"
        )

    st.markdown(info_text, unsafe_allow_html=True)




# Footer
st.markdown("---")
st.caption("For research and educational use only. Powered by Streamlit.")

