import pandas as pd
import numpy as np
import uuid
import warnings
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


warnings.filterwarnings("ignore")

# ==========================================
# STEP 1: Define Custom Scheduler (like RandomActivation)
# ==========================================

class CustomScheduler:
    def __init__(self, model):
        self.model = model
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        for agent in self.agents:
            agent.step()

# ==========================================
# STEP 2: Define Agent Classes
# ==========================================

class BaseAgent:
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model

class PlannerAgent(BaseAgent):
    def __init__(self, unique_id, model, disease):
        super().__init__(unique_id, model)
        self.disease = disease
        self.role = f"planner_{disease}"

    def step(self):
        if self.model.case_queue:
            remaining = []
            for case in self.model.case_queue:
                pid, data, domain, y_true = case
                if domain == self.disease:
                    task_id = f"task_{self.disease}_{pid}"
                    task = {"id": pid, "description": f"Analyze {self.disease} data", "data": data, "y_true": int(y_true)}
                    self.model.context[task_id] = task
                    print(f"[{self.role}] Planned task {task_id}")
                else:
                    remaining.append(case)
            self.model.case_queue = remaining

class ExecutorAgent(BaseAgent):
    def __init__(self, unique_id, model, disease, clf):
        super().__init__(unique_id, model)
        self.disease = disease
        self.role = f"executor_{disease}"
        self.clf = clf

    def step(self):
        for key in list(self.model.context.keys()):
            if key.startswith(f"task_{self.disease}"):
                task = self.model.context.pop(key)
                pred = self.clf.predict([task["data"]])[0]
                result_id = f"result_{self.disease}_{task['id']}"
                self.model.context[result_id] = {
                    "task_id": task["id"],
                    "prediction": int(pred),
                    "agent": self.role
                }
                y_true = int(task.get("y_true", -1))
                self.model.results.append({
                    "PatientID": task["id"],
                    "Disease": self.disease,
                    "y_true": y_true,
                    "y_pred": pred
                })
                print(f"[{self.role}] Executed task {task['id']} → prediction: {pred}")

# ==========================================
# STEP 3: Define the Mesa Model
# ==========================================

class MCPMesaModel:
    def __init__(self, X_d, X_b, X_h, clf_d, clf_b, clf_h):
        self.schedule = CustomScheduler(self)
        self.context = {}
        self.case_queue = []
        self.results = []

        # Add cases to queue (patients)
        for _ in range(100):
            pid = str(uuid.uuid4().int)[:4]
            idx = X_d.sample(1).index[0]
            self.case_queue.append((pid, X_d.loc[idx].values, "diabetes", int(y_d.loc[idx])))
            # Breast cancer
            idx = X_b.sample(1).index[0]
            self.case_queue.append((pid, X_b.loc[idx].values, "breast_cancer", int(y_b[idx])))
            # Heart
            idx = X_h.sample(1).index[0]
            self.case_queue.append((pid, X_h.loc[idx].values, "heart", int(y_h.loc[idx])))

        # Add agents
        self.schedule.add(PlannerAgent("planner_diabetes", self, "diabetes"))
        self.schedule.add(PlannerAgent("planner_breast_cancer", self, "breast_cancer"))
        self.schedule.add(PlannerAgent("planner_heart", self, "heart"))

        self.schedule.add(ExecutorAgent("executor_diabetes", self, "diabetes", clf_d))
        self.schedule.add(ExecutorAgent("executor_breast_cancer", self, "breast_cancer", clf_b))
        self.schedule.add(ExecutorAgent("executor_heart", self, "heart", clf_h))

    def step(self):
        self.schedule.step()


# ==========================================
# STEP 4: Run Simulation
# ==========================================

# Diabetes
diabetes = pd.read_csv("data/pima-indians-diabetes.data.csv",
                       names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                              "DiabetesPedigreeFunction", "Age", "Outcome"])
X_d, y_d = diabetes.drop("Outcome", axis=1), diabetes["Outcome"]

# Breast Cancer
breast = load_breast_cancer()
X_b = pd.DataFrame(breast.data, columns=breast.feature_names)
y_b = breast.target  # 0 = malignant, 1 = benign

# Heart
heart = pd.read_csv("data/heart.csv")
X_h, y_h = heart.drop("target", axis=1), heart["target"]

# load models from files
clf_diabetes = joblib.load("model/diabetes.pkl")
clf_breast = joblib.load("model/breast.pkl")
clf_heart = joblib.load("model/heart.pkl")


model = MCPMesaModel(X_d, X_b, X_h, clf_diabetes, clf_breast, clf_heart)

for step in range(3):
    print(f"\n========== MESA STEP {step + 1} ==========")
    model.step()

# ==========================================
# STEP 5: Visualize Results with Seaborn
# ==========================================

results_df = pd.DataFrame(model.results)
print("Results Collected:", len(results_df))
print(results_df.head())

if not results_df.empty and "Disease" in results_df.columns and "y_pred" in results_df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=results_df, x="Disease", hue="y_pred")
    plt.title("Predictions by Disease Type")
    plt.ylabel("Number of Cases")
    plt.xlabel("Disease")
    plt.legend(title="Prediction", labels=["Negative", "Positive"])
    plt.tight_layout()
    plt.savefig("results/output.png")
else:
    print("No results to display.")


for disease in results_df["Disease"].unique():
    sub = results_df[results_df["Disease"] == disease].dropna(subset=["y_true", "y_pred"])
    if sub.empty:
        print(f"\n--- {disease}: no labeled results ---")
        continue

    y_true = sub["y_true"].astype(int).values
    y_pred = sub["y_pred"].astype(int).values

    print(f"\n=== {disease} ===")
    print(classification_report(y_true, y_pred, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion matrix: {disease}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_{disease}.png")
    plt.close()