import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.callbacks import EarlyStopping
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier
from sklearn.preprocessing import StandardScaler

# Load Model
#xmodel = "concat_40"
#xmodel="maximum_40"
xmodel = "wgtav_40"
model_path = f"d:\\miniconda\\UNSW-NB15\\testing\\JP1_HB_V3\\hy_models\\shadeep_hyb_{xmodel}_model.keras"
print(model_path)
model = tf.keras.models.load_model(model_path)

# Load Data
df_train = pd.read_csv("d:\\miniconda\\UNSW-NB15\\testing\\data\\unsw-nb15_training_90_5_5b_.csv")
df_test = pd.read_csv("d:\\miniconda\\UNSW-NB15\\testing\\data\\unsw-nb15_testing_90_5_5b_.csv")

# Define Features
features = ['id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
            'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
            'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 
            'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

# Prepare Data
X_train = df_train[features].values
y_train = df_train['label'].values
X_test = df_test[features].values
y_test = df_test['label'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.astype(np.float32).reshape(-1, 1)
y_test = y_test.astype(np.float32).reshape(-1, 1)

# ART Classifier
classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=2,
    input_shape=(X_train.shape[1],),
    clip_values=(X_train.min().astype(np.float32), X_train.max().astype(np.float32)),
    loss_object=tf.keras.losses.BinaryCrossentropy(),
)

# PGD Attack
pgd_attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.05,
    eps_step=0.01,
    max_iter=40,
    targeted=False,
    num_random_init=1,
    batch_size=64,
    verbose=True
)

X_train_adv = pgd_attack.generate(X_train)
X_test_adv = pgd_attack.generate(X_test)

# Combine Clean and Adversarial Data
X_train_combined = np.vstack([X_train, X_train_adv])
y_train_combined = np.vstack([y_train, y_train])

# Compile Model
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train Model
history = model.fit(X_train_combined, y_train_combined, 
                    epochs=1,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

best_epoch = len(history.history["val_loss"]) - 5
print(f"\nBest Epoch to Stop Training: {best_epoch}")

# Save Model
model.save(f"{xmodel}_pgd_adversarial_trained.keras")

# Evaluate
clean_acc = np.mean((model.predict(X_test) > 0.5).astype(int) == y_test) * 100
adv_acc = np.mean((model.predict(X_test_adv) > 0.5).astype(int) == y_test) * 100

print(xmodel + " (PGD Attack)")
print(f"Accuracy on Clean Test Data: {clean_acc:.2f}%")
print(f"Accuracy on Adversarial Test Data (PGD): {adv_acc:.2f}%")

# Plot Training and Validation
plt.figure(figsize=(12, 5), dpi=300)

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.axvline(best_epoch, color='r', linestyle='dashed', label="Best Epoch")
plt.xlabel("Epochs", fontsize=14, fontfamily="serif")
plt.ylabel("Loss", fontsize=14, fontfamily="serif")
plt.title("Training vs Validation Loss", fontsize=14, fontfamily="serif")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.axvline(best_epoch, color='r', linestyle='dashed', label="Best Epoch")
plt.xlabel("Epochs", fontsize=14, fontfamily="serif")
plt.ylabel("Accuracy", fontsize=14, fontfamily="serif")
plt.title("Training vs Validation Accuracy", fontsize=14, fontfamily="serif")
plt.legend()

plt.tight_layout()
plt.savefig(f"{xmodel}_PGD_Training_Validation.png", dpi=300, bbox_inches="tight")
plt.show()
'''
xmodel = "concat_40"
.] - ETA: 0s - loss: 0.0834 - accuracy: 5204/5232 [============================>.] - ETA: 0s - loss: 0.0834 - accuracy: 5214/5232 [============================>.] - ETA: 0s - loss: 0.0834 - accuracy: 5224/5232 [============================>.] - ETA: 0s - loss: 0.0834 - accuracy: 5232/5232 [==============================] - 24s 4ms/step - loss: 0.0833 - accuracy: 0.9658 - val_loss: 0.0657 - val_accuracy: 0.9752

Best Epoch to Stop Training: -4
291/291 [==============================] - 1s 3ms/step
291/291 [==============================] - 1s 2ms/step
concat_40 (PGD Attack)
Accuracy on Clean Test Data: 97.52%
Accuracy on Adversarial Test Data (PGD): 96.33%
----------------------------------------------------

5224/5232 [============================>.] - ETA: 0s - loss: 0.0812 - accuracy: 5232/5232 [==============================] - 24s 4ms/step - loss: 0.0812 - accuracy: 0.9678 - val_loss: 0.0624 - val_accuracy: 0.9773

Best Epoch to Stop Training: -4
291/291 [==============================] - 1s 2ms/step
291/291 [==============================] - 1s 3ms/step
maximum_40 (PGD Attack)
Accuracy on Clean Test Data: 97.73%
Accuracy on Adversarial Test Data (PGD): 96.34%

-----------------------------------------------------

5215/5232 [============================>.] - ETA: 0s - loss: 0.0819 - accuracy: 5232/5232 [==============================] - 22s 4ms/step - loss: 0.0819 - accuracy: 0.9676 - val_loss: 0.0693 - val_accuracy: 0.9741

Best Epoch to Stop Training: -4
291/291 [==============================] - 1s 2ms/step
291/291 [==============================] - 1s 2ms/step
wgtav_40 (PGD Attack)
Accuracy on Clean Test Data: 97.41%
Accuracy on Adversarial Test Data (PGD): 96.03%

'''