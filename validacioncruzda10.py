import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K # Importante para limpiar la memoria
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold # Importamos KFold
import time

# --- 0. Cargar Datos Preprocesados ---

print("--- Cargando Datos Preprocesados ('malimg_processed_224_split.npz') ---")
try:
    data = np.load('malimg_processed_224_split.npz')
    X_train_orig = data['X_train']
    y_train_onehot_orig = data['y_train']
    X_test_orig = data['X_test']
    y_test_onehot_orig = data['y_test']
    class_names = data['class_names']
except FileNotFoundError:
    print("\n¡ERROR! No se encontró el archivo 'malimg_processed_224_split.npz'.")
    print("Asegúrate de que el archivo esté en el mismo directorio que este script.")
    exit()

# --- 0.1. Combinar Datos para Cross-Validation ---
X_all = np.concatenate((X_train_orig, X_test_orig), axis=0)
y_all_onehot = np.concatenate((y_train_onehot_orig, y_test_onehot_orig), axis=0)
y_all_labels = np.argmax(y_all_onehot, axis=1)

print(f"Datos totales combinados: {X_all.shape[0]} imágenes.")
del X_train_orig, X_test_orig, y_train_onehot_orig, y_test_onehot_orig

# --- Constantes del Modelo (Basadas en el Paper 1) ---
IMG_SIZE = 224
num_classes = len(class_names)
PATCH_SIZE = 8
PROJECTION_DIM = 128 # 'C' en el paper
DC = 512             # Dimensión oculta Channel-mixing 'Dc'
NUM_MIXER_BLOCKS = 8 
NUM_EPOCHS_MIXER = 10 # El paper menciona 50, usamos 10 para evaluar
BATCH_SIZE = 32
C = PROJECTION_DIM   # 128
LATENT_DIM = C // 2  # 64
NUM_EPOCHS_AE = 10   # El paper menciona 50, usamos 10 para evaluar
N_SPLITS = 10        # 10-fold Cross-Validation

# --- Bloques de Construcción (Definidos una vez) ---

def MlpBlock(hidden_dim, mlp_dim):
    """Bloque MLP con activación GELU (como pide el paper)"""
    return tf.keras.Sequential([
        layers.Dense(units=mlp_dim),
        layers.Activation("gelu"),
        layers.Dense(units=hidden_dim),
    ])

#
# --- AQUÍ ESTÁ LA CORRECCIÓN ---
#
class MixerBlock(layers.Layer):
    """Implementa un bloque Mixer (Token-Mixing y Channel-Mixing)."""
    def __init__(self, projection_dim, token_mlp_dim, channel_mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.token_mlp_dim = token_mlp_dim
        self.channel_mlp_dim = channel_mlp_dim
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.token_mixing_mlp = MlpBlock(token_mlp_dim, self.token_mlp_dim)
        self.channel_mixing_mlp = MlpBlock(self.projection_dim, self.channel_mlp_dim)
    
    def call(self, inputs):
        # 1. Token Mixing
        norm_output1 = self.norm1(inputs)
        # Transponer para mezclar tokens: (Batch, Patches, Proj) -> (Batch, Proj, Patches)
        transposed_output = tf.transpose(norm_output1, perm=[0, 2, 1])
        token_mlp_output = self.token_mixing_mlp(transposed_output)
        # Volver a transponer: (Batch, Proj, Patches) -> (Batch, Patches, Proj)
        transposed_back = tf.transpose(token_mlp_output, perm=[0, 2, 1])
        # Conexión residual
        x = layers.Add()([inputs, transposed_back]) 
        
        # 2. Channel Mixing
        norm_output2 = self.norm2(x)
        channel_mlp_output = self.channel_mixing_mlp(norm_output2)
        # Conexión residual
        y = layers.Add()([x, channel_mlp_output])
        return y
#
# --- FIN DE LA CORRECCIÓN ---
#

# --- Bucle de Validación Cruzada (Cross-Validation) ---

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_accuracies = []

print(f"\n--- INICIANDO VALIDACIÓN CRUZADA DE {N_SPLITS} PLIEGUES (FOLDS) ---")

for fold, (train_index, val_index) in enumerate(kf.split(X_all)):
    
    print(f"\n==================== FOLD {fold + 1} / {N_SPLITS} ====================")
    
    K.clear_session()
    
    print("Dividiendo datos para este fold...")
    X_train_fold, X_val_fold = X_all[train_index], X_all[val_index]
    y_train_onehot_fold, y_val_onehot_fold = y_all_onehot[train_index], y_all_onehot[val_index]
    y_train_labels_fold, y_val_labels_fold = y_all_labels[train_index], y_all_labels[val_index]

    # --- ETAPA 1 (Fold {fold + 1}): MLP-MIXER ---
    
    print(f"[Fold {fold + 1}] Iniciando Etapa 1: MLP-Mixer")
    
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
        x = MixerBlock(
            projection_dim=PROJECTION_DIM, 
            token_mlp_dim=NUM_PATCHES, 
            channel_mlp_dim=DC
        )(x)
    gap_layer = layers.GlobalAveragePooling1D(name="gap_layer")(x) 
    output_layer = layers.Dense(num_classes, activation="softmax", name="classification_head")(gap_layer)
    model_mixer_classifier = Model(inputs=input_layer, outputs=output_layer)
    
    model_mixer_classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"Entrenando MLP-Mixer por {NUM_EPOCHS_MIXER} épocas...")
    model_mixer_classifier.fit(
        X_train_fold, y_train_onehot_fold,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS_MIXER,
        verbose=1 
    )
    
    print(f"Extrayendo características de Etapa 1...")
    model_mixer_extractor = Model(
        inputs=model_mixer_classifier.input, 
        outputs=model_mixer_classifier.get_layer("gap_layer").output
    )
    features_stage1_train = model_mixer_extractor.predict(X_train_fold)
    features_stage1_val = model_mixer_extractor.predict(X_val_fold)

    # --- ETAPA 2 (Fold {fold + 1}): AUTOENCODER ---
    
    print(f"\n[Fold {fold + 1}] Iniciando Etapa 2: Autoencoder")
    
    input_ae = layers.Input(shape=(C,))
    e1 = layers.Dense(C * 2, activation="relu")(input_ae) # 256
    e2 = layers.Dense(C, activation="relu")(e1)           # 128
    encoded = layers.Dense(LATENT_DIM, activation="relu", name="latent_space")(e2) # 64
    d1 = layers.Dense(C, activation="relu")(encoded)     # 128
    d2 = layers.Dense(C * 2, activation="relu")(d1)      # 256
    decoded = layers.Dense(C, activation=None)(d2)       # 128
    
    autoencoder = Model(inputs=input_ae, outputs=decoded)
    encoder_stage2 = Model(inputs=input_ae, outputs=encoded)
    
    autoencoder.compile(optimizer="adam", loss="mse")
    print(f"Entrenando Autoencoder por {NUM_EPOCHS_AE} épocas...")
    autoencoder.fit(
        features_stage1_train, 
        features_stage1_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS_AE,
        verbose=1
    )
    
    print(f"Extrayendo características refinadas de Etapa 2...")
    features_stage2_train = encoder_stage2.predict(features_stage1_train)
    features_stage2_val = encoder_stage2.predict(features_stage1_val)

    # --- ETAPA 3 (Fold {fold + 1}): CLASIFICADOR SVM ---
    
    print(f"\n[Fold {fold + 1}] Iniciando Etapa 3: Clasificador SVM")
    
    svm_classifier = SVC(kernel='rbf') 
    print("Entrenando SVM con características refinadas...")
    svm_classifier.fit(features_stage2_train, y_train_labels_fold)
    
    print("Evaluando SVM en el set de validación del fold...")
    y_pred_svm = svm_classifier.predict(features_stage2_val)
    
    accuracy = accuracy_score(y_val_labels_fold, y_pred_svm)
    print(f"--- Precisión (Accuracy) del Fold {fold + 1}: {accuracy * 100:.2f}% ---")
    fold_accuracies.append(accuracy)
    
    del model_mixer_classifier, model_mixer_extractor, autoencoder, encoder_stage2, svm_classifier

# --- 4. EVALUACIÓN FINAL (Promedio de todos los Folds) ---

print("\n\n--- EVALUACIÓN FINAL DE CROSS-VALIDATION ---")

mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"Resultados de los {N_SPLITS} folds:")
for i, acc in enumerate(fold_accuracies):
    print(f"  Fold {i+1}: {acc * 100:.2f}%")

print(f"\n=====================================================================")
print(f"Precisión Promedio (Accuracy) Final: {mean_accuracy * 100:.2f}%")
print(f"Desviación Estándar:               +/- {std_accuracy * 100:.2f}%")
print("=====================================================================")
print("\n--- Proceso de Replicación (con K-Fold) Finalizado ---")
