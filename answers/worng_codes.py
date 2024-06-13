





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