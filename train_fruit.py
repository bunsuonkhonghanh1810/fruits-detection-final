from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt") 

    results = model.train(
        data="dataset/dataset.yaml",  
        epochs=50,
        imgsz=640,
        batch=16,
        name="fruit-detection",
        workers=0,  
        device=0, 
        verbose=True
    )

if __name__ == "__main__":
    main()
