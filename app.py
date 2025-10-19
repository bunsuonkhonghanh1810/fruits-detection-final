import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

st.set_page_config(
    page_title="Fruit Detection System",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background: #f8f9fa;
    }
    .title-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .camera-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        padding: 0.5rem;
        font-weight: 500;
    }
    .color-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 2px;
        font-weight: 500;
    }
    .detection-item {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #3498db;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-container">
        <h1 style="color: #2c3e50; margin: 0; font-size: 2rem;">Fruit Detection System</h1>
        <p style="color: #7f8c8d; margin-top: 0.5rem;">Real-time fruit detection with color recognition</p>
    </div>
""", unsafe_allow_html=True)

def detect_color(frame, box):
    """Nhận diện màu chủ đạo của vùng được detect"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return "Red", "#e74c3c"
    
    h, w = roi.shape[:2]
    margin_h = int(h * 0.15)
    margin_w = int(w * 0.15)
    
    if margin_h >= h//2 or margin_w >= w//2:
        center_roi = roi
    else:
        center_roi = roi[margin_h:h-margin_h, margin_w:w-margin_w]
    
    hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
    
    mask = (hsv[:,:,2] > 30) & (hsv[:,:,2] < 240) & (hsv[:,:,1] > 20)
    
    if np.sum(mask) > 0:
        filtered_hsv = hsv[mask]
        avg_color = np.median(filtered_hsv, axis=0)
    else:
        avg_color = np.median(hsv.reshape(-1, 3), axis=0)
    
    h_value, s_value, v_value = avg_color
    
    if (0 <= h_value <= 10) or (170 <= h_value <= 180):
        return "Red", "#e74c3c"
    elif 11 <= h_value <= 22:
        return "Orange", "#e67e22"
    elif 23 <= h_value <= 38:
        return "Yellow", "#f39c12"
    else:
        return "Green", "#27ae60"

def get_fruit_color_mapping(fruit_name, detected_color):
    """Kết hợp tên quả và màu"""
    return f'{detected_color} {fruit_name}'

with st.sidebar:
    st.markdown("### Settings")
    
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05
    )
    
    camera_index = st.selectbox(
        "Camera",
        options=[0, 1, 2],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("### Display Options")
    show_labels = st.checkbox("Show Labels", value=True)
    show_conf = st.checkbox("Show Confidence", value=True)
    show_color = st.checkbox("Show Color Detection", value=True)
    
    st.markdown("---")
    
    st.markdown("### System Info")
    st.info("Model: YOLOv8\nMode: Live Detection")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    stframe = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### Statistics")
    
    fps_placeholder = st.empty()
    objects_placeholder = st.empty()
    status_placeholder = st.empty()
    
    st.markdown("---")
    st.markdown("### Detected Items")
    detection_list = st.empty()

col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    start_btn = st.button("Start", type="primary")

with col_btn2:
    stop_btn = st.button("Stop")

with col_btn3:
    reset_btn = st.button("Reset")

if 'detecting' not in st.session_state:
    st.session_state.detecting = False

if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

if 'detected_objects' not in st.session_state:
    st.session_state.detected_objects = 0

if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

if start_btn:
    st.session_state.detecting = True

if stop_btn:
    st.session_state.detecting = False

if reset_btn:
    st.session_state.frame_count = 0
    st.session_state.detected_objects = 0
    st.session_state.detection_history = []

if st.session_state.detecting:
    try:
        model = YOLO("runs/detect/fruit-detection/weights/best.pt")
        
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        status_placeholder.success("Camera Active")
        
        while st.session_state.detecting:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Camera Error")
                break
            
            results = model(frame, conf=confidence)
            
            current_detections = []
            
            annotated_frame = frame.copy()
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                color_name, color_hex = detect_color(frame, box)
                fruit_description = get_fruit_color_mapping(class_name, color_name)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if show_labels and show_conf and show_color:
                    label = f'{fruit_description} {conf:.2f}'
                elif show_labels and show_color:
                    label = fruit_description
                elif show_labels and show_conf:
                    label = f'{class_name} {conf:.2f}'
                elif show_labels:
                    label = class_name
                else:
                    label = ''
                
                if label:
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                current_detections.append({
                    'fruit': class_name,
                    'color': color_name,
                    'description': fruit_description,
                    'confidence': conf,
                    'color_hex': color_hex
                })
            
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            stframe.image(annotated_frame, use_column_width=True)
            
            st.session_state.frame_count += 1
            st.session_state.detected_objects = len(results[0].boxes)
            st.session_state.detection_history = current_detections
            
            fps_placeholder.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.frame_count}</div>
                    <div class="metric-label">Frames</div>
                </div>
            """, unsafe_allow_html=True)
            
            objects_placeholder.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.detected_objects}</div>
                    <div class="metric-label">Objects</div>
                </div>
            """, unsafe_allow_html=True)
            
            if current_detections:
                detection_html = ""
                for det in current_detections:
                    detection_html += f"""
                    <div class="detection-item">
                        <strong>{det['description']}</strong>
                        <br>
                        <span class="color-badge" style="background-color: {det['color_hex']}; color: white;">
                            {det['color']}
                        </span>
                        <span style="color: #7f8c8d; font-size: 0.85rem;">
                            {det['confidence']:.2%}
                        </span>
                    </div>
                    """
                detection_list.markdown(detection_html, unsafe_allow_html=True)
            else:
                detection_list.markdown("No objects detected")
        
        cap.release()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        status_placeholder.error("Detection Stopped")
else:
    status_placeholder.warning("Detection Paused")
    
    fps_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.frame_count}</div>
            <div class="metric-label">Frames</div>
        </div>
    """, unsafe_allow_html=True)
    
    objects_placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.detected_objects}</div>
            <div class="metric-label">Objects</div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.detection_history:
        detection_html = ""
        for det in st.session_state.detection_history:
            detection_html += f"""
            <div class="detection-item">
                <strong>{det['description']}</strong>
                <br>
                <span class="color-badge" style="background-color: {det['color_hex']}; color: white;">
                    {det['color']}
                </span>
                <span style="color: #7f8c8d; font-size: 0.85rem;">
                    {det['confidence']:.2%}
                </span>
            </div>
            """
        detection_list.markdown(detection_html, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p>Fruit Detection System with Color Recognition</p>
    </div>
""", unsafe_allow_html=True)