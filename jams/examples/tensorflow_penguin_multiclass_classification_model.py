import tensorflow as tf
import pandas as pd
from keras.src.layers import Normalization, Dense
from sklearn.model_selection import train_test_split
import seaborn as sns
import json
import ssl

# bypass error which causes the script to crash when downloading dataset
ssl._create_default_https_context = ssl._create_unverified_context

# Load the dataset
penguins = sns.load_dataset('penguins')

# Drop rows with missing values
penguins.dropna(inplace=True)

# Convert categorical columns to numerical
penguins['species'] = penguins['species'].astype('category').cat.codes
penguins['island'] = penguins['island'].astype('category').cat.codes
penguins['sex'] = penguins['sex'].astype('category').cat.codes

# Separate features and labels
X = penguins.drop('species', axis=1)
y = penguins['species']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the normalization layer
normalizer = tf.keras.layers.Normalization(input_shape=[X_train.shape[1],])
normalizer.adapt(X_train)

# Define the model using the Functional API
# Define input layers with specific feature names
input_layers = []
for col in X.columns:
    input_layer = tf.keras.Input(shape=(1,), name=col)
    input_layers.append(input_layer)

# Concatenate all inputs into a single tensor
concatenated = tf.keras.layers.Concatenate()(input_layers)

normalization_layer = Normalization()
normalized_inputs = normalization_layer(concatenated)

# Dense layers for modeling
x = Dense(128, activation='relu')(normalized_inputs)
x = Dense(64, activation='relu')(x)
output = Dense(3, activation='softmax', name='species')(x)  # Assuming 3 species categories

model = tf.keras.Model(inputs=input_layers, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(dict(X_train), y_train, epochs=10, batch_size=32, validation_data=(dict(X_test), y_test))

# Save the sample input
sample_input = X_train.head(10).astype(float).to_dict(orient='list')
with open("tensorflow_input.json", "w") as outfile:
    json.dump(sample_input, outfile)

# Save the model
model.save('tensorflow_penguin_functional')
