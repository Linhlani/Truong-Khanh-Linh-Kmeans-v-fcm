# Import các thư viện cần thiết
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz

# Đường dẫn các ảnh vệ tinh
image_files = ['dau_vao/anh1.jpg', 'dau_vao/anh2.jpg']

# Thiết lập số cụm
n_clusters = 2

# Khởi tạo figure với bố cục đẹp mắt hơn
fig, axes = plt.subplots(len(image_files), 3, figsize=(15, 6 * len(image_files)))
fig.suptitle("So sánh Phân cụm K-means và Fuzzy C-means", fontsize=18, fontweight='bold')
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# Lặp qua từng ảnh để phân cụm và hiển thị
for idx, file in enumerate(image_files):
    # Đọc ảnh và chuyển đổi
    image = cv2.imread(file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hiển thị ảnh gốc
    axes[idx, 0].imshow(image_rgb)
    axes[idx, 0].set_title("Ảnh Gốc", fontsize=14)
    axes[idx, 0].axis('off')

    # Chuyển đổi ảnh thành mảng 2D cho K-means
    pixels = image_rgb.reshape((-1, 3))

    # --- Phân cụm bằng K-means ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels)
    segmented_image_kmeans = kmeans.labels_.reshape(image_rgb.shape[:2])

    # Hiển thị kết quả phân cụm K-means
    axes[idx, 1].imshow(segmented_image_kmeans, cmap='viridis')
    axes[idx, 1].set_title("K-means", fontsize=14)
    axes[idx, 1].axis('off')

    # --- Phân cụm bằng Fuzzy C-means (FCM) ---
    pixels_fcm = pixels.T
    _, u, _, _, _, _, _ = fuzz.cluster.cmeans(pixels_fcm, c=n_clusters, m=2, error=0.005, maxiter=1000)
    labels_fcm = np.argmax(u, axis=0).reshape(image_rgb.shape[:2])

    # Hiển thị kết quả phân cụm FCM
    axes[idx, 2].imshow(labels_fcm, cmap='viridis')
    axes[idx, 2].set_title("Fuzzy C-means", fontsize=14)
    axes[idx, 2].axis('off')

# Hiển thị giao diện đã nâng cấp
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
