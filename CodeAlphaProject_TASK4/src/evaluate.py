from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
