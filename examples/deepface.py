#!/usr/bin/env python3
from deepface import DeepFace

#DeepFace.stream(db_path="/dev/video0")
DeepFace.stream(db_path="./sample_images/", time_threshold=1, frame_threshold=1)
