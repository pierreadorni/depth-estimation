import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

print("loading model")
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

print("setting device")
device = torch.device("mps")
midas.to(device)
midas.eval()

print("loading transforms")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

print("starting webcam")
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    input_batch = transform(frame).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    cv2.imshow("Webcam", (output/output.max()*255).astype(np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
