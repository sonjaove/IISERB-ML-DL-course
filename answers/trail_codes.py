# form assignment 4.


def train_and_evaluate(X_train, y_train, X_test, y_test, degree):
    scale_poly=PolynomialFeatures(degree)
    X_train_poly = scale_poly.fit_transform(X_train, degree)
    X_test_poly = scale_poly.fit_transform(X_test, degree)

    #intializing parameters 
    X = np.c_[np.ones((X_train_poly.shape[0], 1)), X_train_poly]
    lambda_ = 1e-5
    I = np.eye(X.shape[1])
    I[0, 0] = 0 

    # Solve for theta using the regularized normal equation
    theta = np.linalg.inv(X.T.dot(X) + lambda_ * I).dot(X.T).dot(y)  # Here small constant has been added (Not to intercept term)
    #theta = np.linalg.inv(X_train_poly.T.dot(X_train_poly)).dot(X_train_poly.T).dot(y_train)
    # Make predictions
    y_train_pred = X_train_poly.dot(theta)
    y_test_pred = X_test_poly.dot(theta)

    # Evaluate the model on the training and test sets
    train_loss = (1/len(X_train_poly)) * np.sum((y_train_pred - y_train)**2)
    test_loss = (1/len(X_test_poly)) * np.sum((y_test_pred - y_test)**2)
    
    return y_train_pred, y_test_pred, loss_history_train, train_loss, test_loss, theta


for degree in degrees:
    _, y_test_pred, _, _, theta = train_and_evaluate(X_train, y_train, X_test, y_test, degree)
    scale_poly=PolynomialFeatures(degree)
    # Generate polynomial features for the plot
    X_plot = np.linspace(X_test.min(), X_test.max(), 500).reshape(-1, 1)
    X_plot_poly = scale_poly.fit_transform(X_plot, degree)
    y_plot = X_plot_poly.dot(theta)
    
    plt.subplot(2, 2, degrees.index(degree) + 1)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'Polynomial Degree {degree}')
    plt.xlabel('Day of Year (Standardized)')
    plt.ylabel('Maximum Temperature')
    plt.title(f'Polynomial Degree {degree}')
    plt.legend()

plt.tight_layout()
plt.show()


for degree in degrees:
    loss_history_train=[]
    y_train_pred, y_test_pred, train_loss, test_loss, theta = train_and_evaluate(X_train, y_train, X_test, y_test, degree)
    loss_history_train+=[train_loss]
    # Plot training loss
    plt.subplot(2, 2, degrees.index(degree) + 1)
    plt.plot(loss_history_train, label=f'Training Loss (Degree {degree})')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (Degree {degree})')
    plt.legend()

    # Print train and test losses
    print(f"Degree {degree} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

plt.tight_layout()
plt.show()



# from assignmnent 9

#sample cnnclass for the model
class ConvToDense(torch.nn.Module):
    def __init__(self):
        super(ConvToDense, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc = torch.nn.Linear(64 * 32 * 32, 10)  # 10 neuron if we want to do 10 class classification
    
    def forward(self, x): # x: (1, 128, 32, 32)
        x = self.conv(x) # First convolutional layer, x: (1, 64, 32, 32)
        x = torch.relu(x)  # Apply ReLU activation
        x = x.view(x.size(0), -1)  # Flatten the tensor, x: (1, 64 * 32 * 32)
        x = self.fc(x)
        return x



#sample cnn model
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.F.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.F.relu(self.fc1(x))
        x = torch.nn.functional.F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = ConvNet()



# Let us try with more complex model and more training epochs
class ConvNet1(torch.nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.F.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.F.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.nn.functional.F.relu(self.fc1(x))
        x = torch.nn.functional.F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = ConvNet1()



#trying to use coustom filter for assignment 9 

#using the custom filter for the CNN model.
#data_tensor is the tensors of the input data of the class.
class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1)
        self.conv1.weight=torch.nn.Parameter(filter_tensor.unsqueeze(0))
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1)
        self.conv2.weight=torch.nn.Parameter(filter_tensor.unsqueeze(0))
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(1* 3 * 4, 64)
        self.fc2 = torch.nn.Linear(64, 1)
    def forward(self, x): # x=data_tensor
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 1* 3 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x