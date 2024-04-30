import numpy as np

class ART1:
    def __init__(self, input_size=64, output_size=10, rho=0.5):
        # Initialize the parameters of the ART1 model
        self.input_size = input_size
        self.output_size = output_size
        self.rho = rho
        self.n_cats = 0
        
        # Initialize bottom-up and top-down weights
        self.Bij = np.ones((input_size, output_size)) * 0.2
        self.Tji = np.ones((output_size, input_size))

    def train(self, data):
        # Train the model with input data
        for pattern in data:
            self.present_input_pattern(pattern)

    def present_input_pattern(self, idata):
        # Present an input pattern to the model and update weights
        Y = np.dot(self.Bij.T, idata)
        
        # Iterate until a suitable category is found
        while True:
            # Find the winning category (J)
            J = np.argmax(Y)
            
            # Calculate resonance match
            match = np.dot(self.Tji[J], idata) / np.sum(idata)
            
            # If match exceeds vigilance threshold (rho), update weights
            if match >= self.rho:
                # Update bottom-up weights
                self.Bij[:, J] += idata - self.Bij[:, J]
                
                # Update top-down weights
                self.Tji[J, :] = idata
                
                break
            else:
                # Disable the winning category and search for a new one
                Y[J] = -1

    def predict(self, idata):
        # Predict the category of an input pattern
        Y = np.dot(self.Bij.T, idata)
        J = np.argmax(Y)
        return J

def load_data(file_path):
    # Load traffic signal data from a text file
    data = []
    signal = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Remove trailing whitespace
            line = line.strip()
            
            # If the line is empty, it is a separation between signals
            if line == "":
                if signal:
                    # Convert the signal to a binary vector
                    vector = np.array([1 if char == 'X' else 0 for row in signal for char in row])
                    data.append(vector)
                    signal = []
            else:
                # Add the line to the current signal
                signal.append(line)
        
        # Process the last signal if it exists
        if signal:
            vector = np.array([1 if char == 'X' else 0 for row in signal for char in row])
            data.append(vector)
    
    return data

def test_model(model, test_data, expected_labels=None):
    # Test the model with test data
    correct_predictions = 0
    total_tests = len(test_data)
    
    for idx, pattern in enumerate(test_data):
        # Predict the category of the pattern
        predicted_category = model.predict(pattern)
        
        # If ground truth labels are provided, compare the prediction with the expected label
        if expected_labels is not None:
            if predicted_category == expected_labels[idx]:
                correct_predictions += 1
        
        print(f"Pattern {idx + 1}: Predicted Category: {predicted_category}")
    
    # If ground truth labels are provided, calculate accuracy
    if expected_labels is not None:
        accuracy = correct_predictions / total_tests
        print(f"Accuracy: {accuracy * 100:.2f}%")

# Path to the text file with traffic signals (change this to your file path)
file_path = 'transit.txt'

# Load data from the file
data = load_data(file_path)

# Create an instance of the ART1 model
art1_model = ART1(input_size=64, output_size=10, rho=0.5)

# Train the model with the loaded data
art1_model.train(data)

# Assuming you have test data and expected labels (change these variables to your test data and labels)
test_data = data  # You may need to load or split different data for testing
expected_labels = None  # Replace with your expected labels if available

# Test the model with the test data and optional expected labels
test_model(art1_model, test_data, expected_labels)
