import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time
import matplotlib.pyplot as plt

# --- 0. Cargar Datos Preprocesados ---

print("--- Cargando Datos Preprocesados ('malimg_processed_224_split.npz') ---")
try:
    data = np.load('malimg_processed_224_split.npz')
    X_train = data['X_train']
    y_train_onehot = data['y_train'] # Etiquetas One-Hot para Keras
    X_test = data['X_test']
    y_test_onehot = data['y_test']   # Etiquetas One-Hot para Keras
    class_names = data['class_names']
except FileNotFoundError:
    print("\n¡ERROR! No se encontró el archivo 'malimg_processed_224_split.npz'.")
    print("Asegúrate de que el archivo esté en el mismo directorio que este script.")
    exit()

# Para la Etapa 3 (SVM), necesitamos etiquetas 1D (ej. 0, 1, 2...)
y_train_labels = np.argmax(y_train_onehot, axis=1)
y_test_labels = np.argmax(y_test_onehot, axis=1)

print(f"Datos cargados: {X_train.shape[0]} train, {X_test.shape[0]} test.")

# --- Constantes del Modelo (Basadas en el Paper 1) ---
IMG_SIZE = 224
num_classes = len(class_names)

# Parámetros del MLP-Mixer (Paper 1, sección 3.4)
PATCH_SIZE = 8
PROJECTION_DIM = 128 # 'C' en el paper
DS = 64              # Dimensión oculta Token-mixing 'Ds'
DC = 512             # Dimensión oculta Channel-mixing 'Dc'
NUM_MIXER_BLOCKS = 8 # Valor estándar para un modelo ligero
NUM_EPOCHS_MIXER =10 # El paper menciona 50 épocas () pondre 10 para evaluar
BATCH_SIZE = 32

# Parámetros del Autoencoder (Paper 1, sección 3.4)
C = PROJECTION_DIM   # 128
LATENT_DIM = C // 2  # 64
NUM_EPOCHS_AE = 10   # El paper menciona 50 épocas () pondre 10 para evaluar

# --- ETAPA 1: MLP-MIXER (Extractor de Características) ---

print("\n--- INICIANDO ETAPA 1: MLP-MIXER ---")

# --- 1.1. Bloques de Construcción del MLP-Mixer ---
def MlpBlock(hidden_dim, mlp_dim):
    """Bloque MLP con activación GELU (como pide el paper)"""
    return tf.keras.Sequential([
        layers.Dense(units=mlp_dim),
        layers.Activation("gelu"),
        layers.Dense(units=hidden_dim),
    ])

class MixerBlock(layers.Layer):
    """Implementa un bloque Mixer (Token-Mixing y Channel-Mixing)."""
    # <-- CORRECCIÓN 1: Añadir 'projection_dim' al constructor
    def __init__(self, projection_dim, token_mlp_dim, channel_mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.token_mlp_dim = token_mlp_dim
        self.channel_mlp_dim = channel_mlp_dim

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Token-mixing MLP: Salida = token_mlp_dim (num_patches)
        self.token_mixing_mlp = MlpBlock(token_mlp_dim, self.token_mlp_dim)
        
        # <-- CORRECCIÓN 2: La salida del Channel-mixing debe ser 'projection_dim'
        #     No 'token_mlp_dim'
        self.channel_mixing_mlp = MlpBlock(self.projection_dim, self.channel_mlp_dim)
    
    def call(self, inputs):
        # Token Mixing
        norm_output1 = self.norm1(inputs)
        transposed_output = tf.transpose(norm_output1, perm=[0, 2, 1])
        token_mlp_output = self.token_mixing_mlp(transposed_output)
        transposed_back = tf.transpose(token_mlp_output, perm=[0, 2, 1])
        x = layers.Add()([inputs, transposed_back]) # Skip-connection
        
        # Channel Mixing
        norm_output2 = self.norm2(x)
        channel_mlp_output = self.channel_mixing_mlp(norm_output2)
        y = layers.Add()([x, channel_mlp_output]) # Skip-connection
        return y

# --- 1.2. Construir y Entrenar el Modelo Clasificador MLP-Mixer ---
print("Construyendo modelo MLP-Mixer...")

NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2

input_layer = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
patch_embed = layers.Conv2D(
    filters=PROJECTION_DIM, kernel_size=PATCH_SIZE, strides=PATCH_SIZE, name="patch_embed"
)(input_layer)
patch_flat = layers.Reshape(
    target_shape=(NUM_PATCHES, PROJECTION_DIM), name="flatten_patches"
)(patch_embed)

x = patch_flat
for _ in range(NUM_MIXER_BLOCKS):
    # <-- CORRECCIÓN 3: Pasar 'PROJECTION_DIM' al constructor del bloque
    x = MixerBlock(
        projection_dim=PROJECTION_DIM, 
        token_mlp_dim=NUM_PATCHES, 
        channel_mlp_dim=DC
    )(x)

# Esta es la capa que usaremos como salida de características
gap_layer = layers.GlobalAveragePooling1D(name="gap_layer")(x) 

# Cabeza de clasificación (Temporal, solo para entrenar)
output_layer = layers.Dense(num_classes, activation="softmax", name="classification_head")(gap_layer)

# Modelo clasificador completo
model_mixer_classifier = Model(inputs=input_layer, outputs=output_layer)

model_mixer_classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"Entrenando MLP-Mixer por {NUM_EPOCHS_MIXER} épocas...")
start_mixer_train = time.time()
history_mixer = model_mixer_classifier.fit(
    X_train, y_train_onehot,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS_MIXER,
    validation_data=(X_test, y_test_onehot),
    verbose=1
)
print(f"Entrenamiento MLP-Mixer finalizado en {time.time() - start_mixer_train:.2f}s")

# --- 1.3. Crear el Modelo Extractor y Extraer Características ---
print("\nExtrayendo características de Etapa 1 (salida de GAP)...")

model_mixer_extractor = Model(
    inputs=model_mixer_classifier.input, 
    outputs=model_mixer_classifier.get_layer("gap_layer").output
)

features_stage1_train = model_mixer_extractor.predict(X_train)
features_stage1_test = model_mixer_extractor.predict(X_test)

print(f"Shape de características (Etapa 1, train): {features_stage1_train.shape}")

# --- ETAPA 2: AUTOENCODER (Refinador de Características) ---

print("\n--- INICIANDO ETAPA 2: AUTOENCODER ---")

# --- 2.1. Construir y Entrenar el Autoencoder ---
print("Construyendo modelo Autoencoder...")

input_ae = layers.Input(shape=(C,))
e1 = layers.Dense(C * 2, activation="relu")(input_ae) # 256 nodos
e2 = layers.Dense(C, activation="relu")(e1)           # 128 nodos
encoded = layers.Dense(LATENT_DIM, activation="relu", name="latent_space")(e2) # 64 nodos

d1 = layers.Dense(C, activation="relu")(encoded)     # 128 nodos
d2 = layers.Dense(C * 2, activation="relu")(d1)      # 256 nodos
decoded = layers.Dense(C, activation=None)(d2)       # 128 nodos

autoencoder = Model(inputs=input_ae, outputs=decoded)
encoder_stage2 = Model(inputs=input_ae, outputs=encoded)

autoencoder.compile(optimizer="adam", loss="mse")

print(f"Entrenando Autoencoder por {NUM_EPOCHS_AE} épocas...")
start_ae_train = time.time()
history_ae = autoencoder.fit(
    features_stage1_train, 
    features_stage1_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS_AE,
    validation_data=(features_stage1_test, features_stage1_test),
    verbose=1
)
print(f"Entrenamiento AE finalizado en {time.time() - start_ae_train:.2f}s")

# --- 2.2. Extraer Características Refinadas (Etapa 2) ---
print("\nExtrayendo características refinadas de Etapa 2 (salida de Latent Space)...")

features_stage2_train = encoder_stage2.predict(features_stage1_train)
features_stage2_test = encoder_stage2.predict(features_stage1_test)

print(f"Shape de características (Etapa 2, train): {features_stage2_train.shape}")


# --- ETAPA 3: CLASIFICADOR FINAL (SVM) ---

print("\n--- INICIANDO ETAPA 3: CLASIFICADOR SVM ---")

svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

print("Entrenando SVM con características refinadas...")
start_svm_train = time.time()

svm_classifier.fit(features_stage2_train, y_train_labels)

print(f"Entrenamiento de SVM finalizado en {time.time() - start_svm_train:.2f}s")


# --- 4. EVALUACIÓN FINAL DEL PIPELINE ---

print("\n--- EVALUACIÓN FINAL (Pipeline Completo) ---")

y_pred_svm = svm_classifier.predict(features_stage2_test)

accuracy = accuracy_score(y_test_labels, y_pred_svm)
report = classification_report(
    y_test_labels, 
    y_pred_svm, 
    target_names=class_names, 
    zero_division=0
)

print(f"\n===========================================================")
print(f"Precisión (Accuracy) Final del Pipeline: {accuracy * 100:.2f}%")
print("===========================================================")
print("\nReporte de Clasificación (Etapa 3 - SVM):")
print(report)

# --- 5. Visualización Comparativa ---
print("\nGenerando gráficos de rendimiento...")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history_mixer.history['val_accuracy'], label='Precisión Test (Solo Mixer Etapa 1)')
plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Precisión Final (Mixer+AE+SVM): {accuracy*100:.2f}%')
plt.title('Comparación de Precisión del Pipeline')
plt.xlabel('Época (Solo Etapa 1)')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_ae.history['loss'], label='Pérdida de reconstrucción (Train AE)')
plt.plot(history_ae.history['val_loss'], label='Pérdida de reconstrucción (Test AE)')
plt.title('Pérdida del Autoencoder (Etapa 2)')
plt.xlabel('Época')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n--- Proceso de Replicación Finalizado ---")