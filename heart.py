# heart_ml_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os

class HeartDisease:

    def readDataset(self):
        # 1. Loa*d Dataset
        self.df = pd.read_csv("heart.csv")
        print("--- Dataset loaded successfully ---")

    def datasetInfo(self):
        # 2. Dataset info
        print("---Dataset Information:")
        print('--- Shape of Dataset ---')
        print(self.df.shape)
        print("--- Columns ---")
        print(self.df.columns)
        print("--- Data Types ---")
        print(self.df.dtypes)
        print("--- Dataset Information ---")
        print(self.df.info())

    def showDataset(self):    
        # 3. Show data
        print("--- First 5 rows ---")
        print(self.df.head())
        print("--- Last 5 rows ---")
        print(self.df.tail())
        print("--- Missing values per column ---")
        print(self.df.isnull().sum())

    def preProcessing(self):
        # 4. Feature selection
        X = self.df.drop("target", axis=1)
        y = self.df["target"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42) # 1 and 42
        print("Data preprocessing completed.")

        self.scaler = StandardScaler()

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def trainModels(self):
        # 5. Train test split
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier()
        }
        self.results = {}
        print("--- Training models ---")
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            self.results[name] = {
                "model": model,
                "accuracy": acc,
                "y_pred": y_pred
            }
            print(f"{name}: Training done (Accuracy: {acc:.4f})")

    def comparativeAnalysis(self):
        # Create directory if it does not exist
        os.makedirs("static/images", exist_ok=True)

        # 1. Model Accuracy Comparison
        acc_scores = {
            name: res["accuracy"]
            for name, res in self.results.items()
        }

        plt.figure(figsize=(10, 5))
        sns.barplot(
            x=list(acc_scores.keys()),
            y=list(acc_scores.values())
        )
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.xlabel("Machine Learning Algorithms")
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(
            "static/images/model_accuracy_comparison.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        plt.close()

        print("Saved: static/images/model_accuracy_comparison.png")

        # 2. Confusion Matrix for each model
        for name, res in self.results.items():
            cm = confusion_matrix(self.y_test, res["y_pred"])

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()

            # Convert model name to a safe filename
            filename = name.lower().replace(" ", "_")
            filename = filename.replace("-", "_")

            image_path = f"static/images/confusion_matrix_{filename}.png"

            plt.savefig(
                image_path,
                dpi=300,
                bbox_inches="tight"
            )
            plt.show()
            plt.close()

            print(f"Saved: {image_path}")

            print(f"--- Classification Report for {name} ---")
            print(
                classification_report(
                    self.y_test,
                    res["y_pred"]
                )
            )

    
    def trainKNNAndSave(self):
        # Train KNN model and save as pickle file
        import pickle

        print("--- Training KNN Model ---")

        # Create KNN model
        knn = KNeighborsClassifier(n_neighbors=5)

        # Train the model
        knn.fit(self.X_train, self.y_train)

        # Save trained model
        with open("knn.pkl", "wb") as file:
            pickle.dump(knn, file)

        # Evaluate model
        y_pred = knn.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        print("KNN model trained successfully.")
        print(f"KNN Accuracy: {accuracy:.4f}")
        print("Model saved as knn.pkl")
        with open("scaler.pkl", "wb") as file:
            pickle.dump(self.scaler, file)
    def runPipeLine(self):
         self.readDataset()
         self.datasetInfo()
         self.preProcessing()
         self.trainModels()
         self.comparativeAnalysis()
         self.trainKNNAndSave()

# End of class

# Main Program
obj = HeartDisease()
obj.runPipeLine()
   
