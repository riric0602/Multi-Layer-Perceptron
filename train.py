from separate import preprocess_and_split_data

# Initialize weights matrices and biases vectors randomly
# calculate z the weighted sum of weights and biases
# activation function a(z) using sigmoid or relu
# calculate the CHAIN RULE which is the sensitivity of C (cost) to the weights
# Define cost function (f(w, b) = x)
# Gradient Descent to minimize cost function -> weights

if __name__ == "__main__":
    try:
        # Retrieve training and validation sets
        X_train, X_val, y_train, y_val = preprocess_and_split_data()



    except Exception as e:
        print(f"Error: {e}")