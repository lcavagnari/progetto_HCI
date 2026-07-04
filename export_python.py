import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Source - https://stackoverflow.com/a/47285662
# Posted by Shovalt, modified by community. See post 'Timeline' for change history
# Retrieved 2026-07-03, License - CC BY-SA 4.0

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


target_dir = 'deap-dataset/data_preprocessed_python/'

global subject_data
subjects_data = []

global subject_labels
subjects_labels = []



Fs = 128 # Hz
label_column = 0
segment_length = 15 * Fs
b, a = butter(
    4, # 4th-order
    [0.5/(Fs/2), 45/(Fs/2)], # [0.5–45 Hz], Typical EEG range
    btype='bandpass'
)

global  results
results = []

global  segment_splits
segment_splits = []


global X
X = np.array([])

global Y
Y={
    "valence": [],
    "arousal": [],
    "dominance": [],
    "liking": []
}

global Xtrain
global Xtest
global Ytrain
global Ytest

classifier_labels = [
    "SVM",
    "KNN",
    "Logistic Regression",
    "Decision Tree",
    "LDA"
]

segment_labels = [
    "0-15 s",
    "15-30 s",
    "30-45 s",
    "45-60 s"
]



scorers = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}

scorers = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def plot_classifier_metrics(df, title="Classifier Comparison") -> None:
    # Aggregated to ensure exactly 5 bars per chart are plotted per the criteria
    df_agg = df.groupby("classifier", as_index=False).mean(numeric_only=True)
    
    names = df_agg["classifier"].tolist()

    metrics = {
        "Accuracy": df_agg["accuracy"].tolist(),
        "Precision": df_agg["precision"].tolist(),
        "Recall": df_agg["recall"].tolist(),
        "F1": df_agg["f1"].tolist(),
    }

    x = np.arange(len(df_agg))
    width = 0.2

    plt.figure(figsize=(10, 5))

    for i, (metric, values) in enumerate(metrics.items()):
        plt.bar(x + i * width, values, width=width, label=metric)

    plt.xticks(x + width * 1.5, names)
    plt.title(title)
    plt.xlabel("Classifier")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()
    

def plot_data_distribution(labels, label_names) -> None:
    fig, axes = plt.subplots(4, figsize=(6, 8))
    labels_data = np.vstack(labels)
    
    for i, label in enumerate(label_names):
        axes[i].hist(
            x=labels_data[:, i], 
            bins=50,
            density=True,
            edgecolor="grey",
            facecolor="orange"
        )
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Count")
        axes[i].set_title(f"{label} Distribution")

    plt.tight_layout()
    plt.show()


def pca(Xtrain, n_comp=20) -> tuple:
    pca_model = PCA(n_components=n_comp)
    Xtrain = pca_model.fit_transform(Xtrain)
    
    return Xtrain


def split_traintest(X_red, Y) -> tuple:
    """Splits train and test data from normalised dataset"""

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X_red, Y, test_size=0.30, random_state=42, stratify=Y
    )
    
    return Xtrain, Xtest, Ytrain, Ytest


def runTrial(trial, data, segment, valence_labels, arousal_labels, dominance_labels, liking_labels) -> tuple:
    eeg = data[trial, 0:32, :]

    # Bandpass filter: 0.5-45 Hz, 4th-order Butterworth
    eeg_filt = np.zeros_like(eeg)
    for ch in range(32):
        eeg_filt[ch, :] = filtfilt(b, a, eeg[ch, :])


    # Slice current segment
    start_idx = (segment * segment_length)+384
    end_idx = ((segment + 1) * segment_length)+384
    eeg_seg = eeg_filt[:, start_idx:end_idx]


    # Extract 6 statistical features per channel
    feature_vector = []
    for ch in range(32):
        x = eeg_seg[ch, :]
        
        feature_vector.extend([
            np.mean(x),
            np.std(x),
            np.var(x),
            np.sqrt(np.mean(x**2)),
            skew(x),
            kurtosis(x)
        ])
    
    
    return feature_vector, valence_labels[trial], arousal_labels[trial], dominance_labels[trial], liking_labels[trial]


def _process_subject(data, labels, segment: int) -> tuple:
    v_labels = np.where(labels[:,0] > 5, 1, 0)
    a_labels = np.where(labels[:,1] > 5, 1, 0)
    d_labels = np.where(labels[:,2] > 5, 1, 0)
    l_labels = np.where(labels[:,3] > 5, 1, 0)
    
    
    x,v,a,d,l = [],[],[],[],[]
    for trial in range(40):

        features, valence, arousal, dominance, liking = runTrial(
            trial,
            data,
            segment,
            v_labels,
            a_labels,
            d_labels,
            l_labels,
        )

        x.append(features)

        v.append(valence)
        a.append(arousal)
        d.append(dominance)
        l.append(liking)
        
    return x, v, a, d, l


def pre_process() -> None:
    # PREPROCESS ALL SEGMENTS — Feature Extraction + PCA + Split
    for segment in range(4):

        print("\n")

        X = []

        # Dictionary to hold all label arrays
        Y = {
            "valence": [],
            "arousal": [],
            "dominance": [],
            "liking": []
        }

        with ProcessPoolExecutor(max_workers=round(os.cpu_count() * 0.80)) as executor:
            futures = {
                executor.submit(_process_subject, subjects_data[subj_idx], subjects_labels[subj_idx], segment): subj_idx
                    for subj_idx in range(len(subjects_data))
            }
            
            for i, future in enumerate(as_completed(futures), start=1):
                X_sub, v_sub, a_sub, d_sub, l_sub = future.result()
                
                X.extend(X_sub)
                Y["valence"].extend(v_sub)
                Y["arousal"].extend(a_sub)
                Y["dominance"].extend(d_sub)
                Y["liking"].extend(l_sub)

                print_progress_bar(i, len(subjects_data), after=f"SEGMENT {segment + 1}")

        # NORMALISATION
        X = np.array(X)
        for key in Y:
            Y[key] = np.array(Y[key])

        # PCA
        X = pca(X, 20)

        # Save the segment
        segment_splits.append((X, Y))

    print(f"\nPreprocessing complete.")


def evaluate_metrics(clf, Features, Labels) -> tuple:
    scores = {
        name: np.median(cross_val_score(clf, Features, Labels, cv=10, scoring=scorer))
        for name, scorer in scorers.items()
    }
    return scores["accuracy"], scores["precision"], scores["recall"], scores["f1"]



def run_model(model,segments_split,c_name="Classifier") -> []:
    pred = []
    
    for segment in range(4):
        X, Y = segments_split[segment]
        
        for label_name, y_values in Y.items():
            acc, prec, rec, f1 = evaluate_metrics(model, X, y_values)
            
            pred.append({
                "segment": segment+1, 
                "label": label_name, 
                "classifier": c_name, 
                "accuracy": acc, 
                "precision": prec, 
                "recall": rec, 
                "f1": f1
            })
        
    return pred    


def print_progress_bar(current, total, length=30, after:str = "", new_line=False) -> None:
    filled = int(length * current / total)
    bar = "*" * filled + "-" * (length - filled)
    
    print(f"\r[{bar}] {current}/{total}\t {after}", end="\n" if new_line else "" , flush=True)
    
    if current == total:
        print()


def load_data() -> None:
    if os.path.exists(target_dir):
        #os.chdir(target_dir)
        print(f"Changed directory to: {os.getcwd()}")
    else:
        print(f"Error: Directory not found: {os.path.join(os.getcwd(), target_dir)}. Please ensure 'deap-dataset.zip' was extracted correctly and contains 'data_preprocessed_python/'.")
    
    warnings = []
    for subject in range(1, 33):
        
        filename = f"{target_dir}/s{subject:02d}.dat"
        try:
            # Open binary file
            with open(filename, "rb") as f:
                dataset = pickle.load(f, encoding="latin1")
            
            # load features
            subjects_data.append(dataset["data"])
            
            # load labels
            subjects_labels.append(dataset["labels"])
            
            print_progress_bar(subject, 32)
            
        except (EOFError, pickle.UnpicklingError, OSError) as e:
            warnings.append(
                f"WARNING: skipping s{subject:02d}.dat ({e})" if len(warnings) < 1 else f"s{subject:02d}.dat ({e})"
            )
            
            print_progress_bar(subject, 32, after=', '.join(warnings), new_line=True)
            continue

    print(f"\nCached {len(subjects_data)} subjects successfully.")


if __name__ == "__main__":
    
    print("\nLoading subjects data\n")
    
    # Load data
    load_data()
    
    plot_data_distribution(subjects_labels,
        label_names=["Valence", "Arousal", "Dominance", "Liking"]
    )

    print("\n\nStarting label and feature pre-processing",end="")
    
    # Preprocess
    pre_process()
    
    # Classifiers
    
    models = {
        "SVM": SVC(kernel='rbf'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(),
        "LDA": LinearDiscriminantAnalysis()
    }
    
    print(f"\n\nRunning {len(models)} classifiers\n")
    
    with ProcessPoolExecutor(max_workers=round(os.cpu_count() * 0.80)) as executor:
        futures = {
            executor.submit(run_model, models[m_label], segment_splits, m_label): m_label
                for m_label in models
        }
        
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.extend(result)
            
            print_progress_bar(i, len(models), after=result[0]["classifier"], new_line=True)

    
    print("\n\nPrinting results")
    
    # Build DataFrame from results
    result_table = pd.DataFrame(results)
    display_table = result_table.copy()

    # Print and plot per label
    for label in ["valence", "arousal", "dominance", "liking"]:
        subset = display_table[display_table["label"] == label]
        print(f"\n--- {label.capitalize()} ---")
        print(subset.to_string(index=False))
        plot_classifier_metrics(subset, title=f"Classifier comparison — {label}")

    # Best model
    best_index = result_table["f1"].idxmax()

    print("\nBEST MODEL\n")
    print(display_table.loc[best_index])
