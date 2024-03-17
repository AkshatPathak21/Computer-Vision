import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)
        feature = model.predict(np.expand_dims(frame, axis=0))
        features.append(feature)
    cap.release()
    return features

def calculate_cosine_similarity(feature_vector1, feature_vector2):
    feature_vector1 = feature_vector1.reshape(1, -1)
    feature_vector2 = feature_vector2.reshape(1, -1)
    
    similarity = cosine_similarity(feature_vector1, feature_vector2)
    return similarity[0][0]


def content_based_video_retrieval(query_video_path, video_database):
    video_scores = {}
    for video_path, database_features in video_database.items():
        similarity_score = calculate_cosine_similarity(query_video_features, database_features)
        video_scores[video_path] = similarity_score
    sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_videos

def calculate_similarity(query_features, database_features):
    feature_vector1 = feature_vector1.reshape(1, -1)
    feature_vector2 = feature_vector2.reshape(1, -1)

    similarity = cosine_similarity(feature_vector1, feature_vector2)
    return similarity[0][0]

query_video_path = "query_video.mp4"
query_video_features = extract_features_from_video(query_video_path)
video_database = {
    "video1.mp4": extract_features_from_video("video1.mp4"),
    "video2.mp4": extract_features_from_video("video2.mp4"),
}

similar_videos = content_based_video_retrieval(query_video_path, video_database)
print(similar_videos)
