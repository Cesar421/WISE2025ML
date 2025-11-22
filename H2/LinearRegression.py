import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ClassLMS():
    def __init__(self):
        print('-' * 100)
        self.lr = 0.01
        self.epochs = 100
        self.f1 = 'num_words'
        self.f2 = 'num_characters'
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.w = None
        np.random.seed(0)
    
    def _apply_defaults(self, lr=None, epochs=None, X_train=None, X_test=None, y_train=None, f1=None, f2=None, W0=None, W1=None, W2=None):
        return {
            'lr': lr if lr is not None else self.lr,
            'epochs': epochs if epochs is not None else self.epochs,
            'X_train': X_train if X_train is not None else self.X_train,
            'X_test': X_test if X_test is not None else self.X_test,
            'y_train': y_train if y_train is not None else self.y_train,
            'f1': f1 if f1 is not None else self.f1,
            'f2': f2 if f2 is not None else self.f2,
            'W0': W0 if W0 is not None else getattr(self, 'W0', None),
            'W1': W1 if W1 is not None else getattr(self, 'W1', None),
            'W2': W2 if W2 is not None else getattr(self, 'W2', None)
        }

    def load_features(self):
        print('-' * 30)
        print('Starting to load dataset')
        X_train = pd.read_csv('dataset/features-train.tsv', sep='\t')
        X_test = pd.read_csv('dataset/features-test.tsv', sep='\t')
        y_train = pd.read_csv('dataset/labels-train.tsv', sep='\t')

        if 'is_human' in y_train.columns:
            y_train['target'] = (
                y_train['is_human']
                .replace({'true': True, 'false': False})
                .astype(int) if y_train['is_human'].dtype == object else y_train['is_human'].astype(int)
            )
        print('Dataset loaded successfully')
        
        # Store in instance
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        
        return X_train, X_test, y_train

    def train_model(self, lr=None, epochs=None, X_train=None, y_train=None, f1=None, f2=None):
        print('-' * 30)
        print('Starting to train the model')
        d = self._apply_defaults(lr=lr, epochs=epochs, X_train=X_train, y_train=y_train, f1=f1, f2=f2)
        lr, epochs, X_train, y_train, f1, f2 = d['lr'], d['epochs'], d['X_train'], d['y_train'], d['f1'], d['f2']

        # Prepare data
        X2 = X_train[[f1, f2]].astype(float).copy()
        y = y_train['target'].astype(float).values

        # Standardize features (better convergence)
        mu = X2.mean()
        sigma = X2.std().replace(0, 1.0)
        X_std = (X2 - mu) / sigma
        X_vals = X_std.values

        # LMS training
        np.random.seed(0)
        w = np.zeros(3)
        MSE_history = []
        RSS_history = []

        for j in range(epochs):
            idx = np.random.permutation(len(X_vals))
            
            # Update weights for all samples
            for i in idx:
                x1, x2 = X_vals[i]
                x_aug = np.array([1.0, x1, x2])
                y_hat = np.dot(w, x_aug)
                e = y[i] - y_hat
                w += lr * e * x_aug
            
            # Compute MSE/RSS after all updates (standardized space)
            epoch_error = 0.0
            for i in range(len(X_vals)):
                x1, x2 = X_vals[i]
                x_aug = np.array([1.0, x1, x2])
                y_hat = np.dot(w, x_aug)
                e = y[i] - y_hat
                epoch_error += e ** 2
            
            # Store the MSE error
            mse = epoch_error / len(X_vals)
            MSE_history.append(mse)
            RSS_history.append(epoch_error)


        # Store weights and error history
        self.w = w
        self.MSE_history = MSE_history
        self.RSS_history = RSS_history

        # Convert weights to original feature space
        W0 = w[0] - w[1]*mu[f1]/sigma[f1] - w[2]*mu[f2]/sigma[f2]
        W1 = w[1] / sigma[f1]
        W2 = w[2] / sigma[f2]

        print(f"LMS weights (standardized): {w}")
        print(f"LMS weights (original space): W0={W0}, W1={W1}, W2={W2}")
        print('Model trained successfully')
        print (f'Final MSE: {MSE_history[-1]}')

        # Store all parameters for later visualization
        self.f1 = f1
        self.f2 = f2
        self.W0 = W0
        self.W1 = W1
        self.W2 = W2

        return W0, W1, W2

    def visualization(self, X_train=None, y_train=None, f1=None, f2=None, W0=None, W1=None, W2=None):
        print('-' * 30)
        print('Starting visualization')
        # Use defaults if not provided
        d = self._apply_defaults(X_train=X_train, y_train=y_train, f1=f1, f2=f2, W0=W0, W1=W1, W2=W2)
        X_train, y_train, f1, f2, W0, W1, W2 = d['X_train'], d['y_train'], d['f1'], d['f2'], d['W0'], d['W1'], d['W2']
            
        # Scatter plot with decision line y_hat = 0.5  =>  W0 + W1*x + W2*y = 0.5
        colors = ['red' if v == 1 else 'blue' for v in y_train['target']]
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[f1], X_train[f2], c=colors, alpha=0.6, edgecolor='k', label='Train samples')

        # Plot decision line if W2 != 0
        if W0 is not None and W1 is not None and W2 is not None and abs(W2) > 1e-12:
            x_min, x_max = X_train[f1].min(), X_train[f1].max()
            x_grid = np.linspace(x_min, x_max, 200)
            y_line = (0.5 - W0 - W1 * x_grid) / W2
            plt.plot(x_grid, y_line, 'k--', linewidth=2, label='LMS boundary (ŷ=0.5)')

        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f'{f1} vs {f2} with LMS line for training data (Final MSE: {self.MSE_history[-1]:.3f})')
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0],[0], marker='o', color='w', label='is_human = True', markerfacecolor='red', markersize=8),
            Line2D([0],[0], marker='o', color='w', label='is_human = False', markerfacecolor='blue', markersize=8)
        ]
        if W0 is not None:
            legend_elements.append(Line2D([0],[0], color='k', lw=2, linestyle='--', label='LMS boundary'))
        plt.legend(handles=legend_elements, loc='best')
        plt.tight_layout()
        plt.savefig(f'{f1}_vs_{f2}_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
        print('Visualization complete')
    
    def plot_error_history(self):
        print('-' * 30)
        print('Plotting error history')
        
        # Plot MSE
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.MSE_history) + 1), self.MSE_history, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE (Mean Squared Error)')
        plt.title(f'MSE vs Epochs (Final MSE: {self.MSE_history[-1]:.6f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('MSE_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot RSS
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.RSS_history) + 1), self.RSS_history, 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('RSS (Residual Sum of Squares)')
        plt.title(f'RSS vs Epochs (Final RSS: {self.RSS_history[-1]:.6f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('RSS_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print('Error history plotted')
    
    def compute_rss(self, X_train=None, y_train=None, f1=None, f2=None, W0=None, W1=None, W2=None):

        print('-' * 30)
        print('Computing RSS (Residual Sum of Squares)')
        # Use defaults if not provided
        d = self._apply_defaults(X_train=X_train, y_train=y_train, f1=f1, f2=f2, W0=W0, W1=W1, W2=W2)
        X_train, y_train, f1, f2, W0, W1, W2 = d['X_train'], d['y_train'], d['f1'], d['f2'], d['W0'], d['W1'], d['W2']
        
        # Extract features (ORIGINAL space, not standardized)
        X_features = X_train[[f1, f2]].astype(float).values
        y_true = y_train['target'].astype(float).values
        
        # Compute predictions in original space: y_hat = W0 + W1*x1 + W2*x2
        y_pred = W0 + W1 * X_features[:, 0] + W2 * X_features[:, 1]
        
        # Compute RSS: sum of squared residuals
        residuals = y_true - y_pred
        rss = np.sum(residuals ** 2)
        mse = rss / len(y_true)
        rmse = np.sqrt(mse)
        
        print(f"RSS (Residual Sum of Squares): {rss:.6f}")
        print(f"MSE (Mean Squared Error): {mse:.6f}")
        print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
        print(f"Note: Training RSS (standardized features, last epoch): {self.RSS_history[-1]:.6f}")
        print('RSS computation complete')
        
        return rss, mse, rmse
    
    def predict_test(self, X_test=None, f1=None, f2=None, W0=None, W1=None, W2=None, output_file='predictions-test.tsv'):
        print('-' * 30)
        print('Starting predictions on test set')
        # Use defaults if not provided
        d = self._apply_defaults(X_test=X_test, f1=f1, f2=f2, W0=W0, W1=W1, W2=W2)
        X_test, f1, f2, W0, W1, W2 = d['X_test'], d['f1'], d['f2'], d['W0'], d['W1'], d['W2']   
        # Extract features
        X_test_features = X_test[[f1, f2]].astype(float).values
        # Make predictions: y_hat = W0 + W1*x1 + W2*x2
        y_pred = W0 + W1 * X_test_features[:, 0] + W2 * X_test_features[:, 1]
        # Classify: if y_hat >= 0.5 -> True (human), else False (AI)
        is_human_pred = y_pred >= 0.5
        # Create dataframe with id and is_human columns
        predictions_df = pd.DataFrame({
            'id': X_test['id'],
            'is_human': is_human_pred
        })
        # Save to TSV file
        predictions_df.to_csv(output_file, sep='\t', index=False)
        print(f"Predictions saved to {output_file}")
        print(f"Total predictions in the test dataset: {len(predictions_df)}")
        print(f"Predicted as human (True): {is_human_pred.sum()}")
        print(f"Predicted as AI (False): {(~is_human_pred).sum()}")
        print('Predictions complete')
        return predictions_df
    
def main():
    model = ClassLMS()
    model.load_features()
    model.train_model(lr=0.001, epochs=100)
    model.visualization()
    model.plot_error_history()
    model.compute_rss()  # Requirement (c): Compute RSS
    model.predict_test()
    print('-' * 100)

if __name__ == "__main__":
    main()













