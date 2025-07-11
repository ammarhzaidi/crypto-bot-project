from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelEvaluator:
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, predictions > 0.5),
            'precision': precision_score(y_test, predictions > 0.5),
            'recall': recall_score(y_test, predictions > 0.5)
        }
        return metrics
