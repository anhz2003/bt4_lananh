import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.metrics import accuracy_score

# Đường dẫn tới thư mục chứa ảnh
data_path = r"E:\xulyanh\kho100anh"

# Khởi tạo các biến dữ liệu và nhãn
images = []
labels = []

# Đọc ảnh và gán nhãn
for label, class_name in enumerate(os.listdir(data_path)):
    class_folder = os.path.join(data_path, class_name)
    if os.path.isdir(class_folder):
        for image_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, image_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh grayscale
            img_resized = cv2.resize(img, (64, 64))  # Resize ảnh về 64x64
            images.append(img_resized.flatten())  # Chuyển ảnh thành vector 1 chiều
            labels.append(label)

# Chuyển đổi dữ liệu thành mảng numpy
images = np.array(images)
labels = np.array(labels)

# Các kịch bản chia train-test
split_ratios = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.4, 0.6)]

for train_ratio, test_ratio in split_ratios:
    print(f"\nKịch bản chia dữ liệu: {int(train_ratio * 100)}-{int(test_ratio * 100)}")

    # Chia tập train và test
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_ratio, random_state=42)

    # SVM Classifier
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print(f"Độ chính xác SVM: {svm_accuracy * 100:.2f}%")

    # KNN Classifier
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    print(f"Độ chính xác KNN: {knn_accuracy * 100:.2f}%")
