from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def train_and_evaluate(X, y, label_encoder):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        "SVM": SVC(kernel='rbf', C=10),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name} Accuracy: {score:.4f}")
        print(classification_report(y_test, model.predict(X_test), target_names=label_encoder.classes_))

