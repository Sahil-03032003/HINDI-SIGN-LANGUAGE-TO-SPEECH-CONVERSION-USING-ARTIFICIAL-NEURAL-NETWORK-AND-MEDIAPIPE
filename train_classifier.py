import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Use 'Nirmala UI' (standard on Windows) or 'Arial Unicode MS' for Hindi support
plt.rcParams['font.family'] = 'Nirmala UI' 
plt.rcParams['axes.unicode_minus'] = False

# 1. Load Data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# 2. Encode Labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)
np.save('classes.npy', le.classes_)

# 3. Split Data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded
)

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# 4. Define Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train and Save History
print("🚀 Starting training...")
history = model.fit(
    x_train, y_train_cat, 
    epochs=30, 
    batch_size=32, 
    validation_data=(x_test, y_test_cat),
    verbose=1
)

model.save('model_mediapipe.h5')
print("✅ Model saved.")

# ==========================================
# 6. PLOT ACCURACY & LOSS GRAPHS
# ==========================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange', linestyle='--')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='red')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='darkred', linestyle='--')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_performance.png')
plt.show()

# ==========================================
# 7. CONFUSION MATRIX & EVALUATION
# ==========================================
print("\n📊 Generating Confusion Matrix...")

# 1. Get predictions
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# 2. Generate Matrix
cm = confusion_matrix(y_test, y_pred)
classes = le.classes_

# 3. Create a larger figure for 33+ characters
plt.figure(figsize=(20, 15)) 

# 4. Use a smaller font for annotations so they fit in the boxes
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 8}) # Smaller numbers inside boxes

plt.title('Confusion Matrix - Hindi Sign Language (Per Alphabet)', fontsize=16)
plt.ylabel('Actual Alphabet', fontsize=12)
plt.xlabel('Predicted Alphabet', fontsize=12)

# 5. Rotate labels for better visibility
plt.xticks(rotation=90) 
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('confusion_matrix_detailed.png', dpi=300) # High resolution for paper
plt.show() 

# Print text report
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=classes))