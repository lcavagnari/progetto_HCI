# FIXES

## 1 — PCA function

Return model alongside transformed data:

```python
def pca(X, n_comp=20):
    model = PCA(n_components=n_comp)
    X_pca = model.fit_transform(X)
    return X_pca, model
```

## 2 — New helper (add after PCA function)

```python
def split_testtrain(X, Y, test_size=0.30, n_components=20):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        X, Y, test_size=test_size, random_state=42, stratify=Y
    )
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    Xtrain, pca_model = pca(Xtrain, n_components)
    Xtest = pca_model.transform(Xtest)
    return Xtrain, Xtest, Ytrain, Ytest
```

## 3 — Move caching cell

Place this BEFORE preprocessing loop:

```python
# CACHE ALL SUBJECT DATA (load .dat files ONCE)
all_subject_data = []
all_subject_labels = []
for subject in range(1, 33):
    filename = f"s{subject:02d}.dat"
    try:
        with open(filename, "rb") as f:
            dataset = pickle.load(f, encoding="latin1")
        all_subject_data.append(dataset["data"])
        all_subject_labels.append(dataset["labels"])
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f"WARNING: skipping s{subject:02d}.dat (corrupted: {e})")
        continue
print(f"Cached {len(all_subject_data)} subjects successfully.")
```

## 4 — Preprocessing loop

Replace with:

```python
# PREPROCESS ALL SEGMENTS — Feature Extraction + Split + Scale + PCA
for segment in range(4):

    print("\n")
    print("="*40)
    print(f"SEGMENT {segment+1}")
    print("="*40)

    X = []
    Y = []

    for subj_idx in range(len(all_subject_data)):

        data = all_subject_data[subj_idx]
        labels = all_subject_labels[subj_idx]

        current_labels = labels[:, label_column]
        current_labels = np.where(current_labels > 5, 1, 0)

        for trial in range(40):

            eeg = data[trial, 0:32, :]

            eeg_filt = np.zeros_like(eeg)
            for ch in range(32):
                eeg_filt[ch, :] = filtfilt(b, a, eeg[ch, :])

            start_idx = segment * segment_length
            end_idx = (segment + 1) * segment_length
            eeg_seg = eeg_filt[:, start_idx:end_idx]

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

            X.append(feature_vector)
            Y.append(current_labels[trial])

    X = np.array(X)
    Y = np.array(Y)

    Xtrain, Xtest, Ytrain, Ytest = split_testtrain(X, Y)
    segment_splits.append((Xtrain, Xtest, Ytrain, Ytest))

print("Preprocessing complete. segment_splits has 4 entries.")
```
