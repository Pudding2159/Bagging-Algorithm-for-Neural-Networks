import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

def add_noise_to_features(X, noise_mean, noise_std_dev):
    noise = np.random.normal(noise_mean, noise_std_dev, X.shape)
    return X + noise

def create_bagging_samples(strategy, X, y, num_estimators, sample_size):
    bagging_samples = []

    if strategy == "small_bags":
        for i in range(num_estimators):
            indices = np.random.choice(X.shape[0], size=sample_size, replace=True)
            bagging_samples.append((X[indices], y[indices]))

    elif strategy == "small_bags_without_repeats":
        for _ in range(num_estimators):
            indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
            bagging_samples.append((X[indices], y[indices]))

    elif strategy == "disjunctive_distributions":
        step = X.shape[0] // num_estimators
        for i in range(num_estimators):
            start_idx = i * step
            end_idx = (i + 1) * step if i < num_estimators - 1 else X.shape[0]
            bagging_samples.append((X[start_idx:end_idx], y[start_idx:end_idx]))

    elif strategy == "disjunctive_bags":
        step = X.shape[0] // num_estimators
        num_replications = sample_size - step

        for i in range(num_estimators):
            start_idx = i * step
            end_idx = (i + 1) * step if i < num_estimators - 1 else X.shape[0]
            base_subset = (X[start_idx:end_idx], y[start_idx:end_idx])

            replicated_indices = np.random.choice(base_subset[0].shape[0], size=num_replications, replace=True)
            X_replicated = np.vstack([base_subset[0], base_subset[0][replicated_indices]])
            y_replicated = np.concatenate([base_subset[1], base_subset[1][replicated_indices]])

            bagging_samples.append((X_replicated, y_replicated))

    else:
        raise ValueError("Invalid bagging strategy.")

    return bagging_samples

# Načítanie súboru údajov Iris
iris = load_iris()
X = iris.data
y = iris.target


noise_mean = 0.5
noise_std_dev = 0.5
X_noisy = add_noise_to_features(X, noise_mean, noise_std_dev)

# Rozdeliť súbor údajov na tréningovú a testovaciu množinu
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)

# Bagging parameters
num_estimators = 20
max_samples = 0.6
sample_size = int(max_samples * X_train.shape[0])

print("Choose the desired strategy: 'small_sacks', 'small_sacks_without_repetition', 'disjunctive_partitions', or 'disjunctive_sacks'")
bagging_strategy = input("Enter the strategy: ")
bagging_samples = create_bagging_samples(bagging_strategy, X_train, y_train, num_estimators, sample_size)

# Trénujte súbor neurónových sietí
estimators = []
loss_curves = []

for i in range(num_estimators):
    X_sample, y_sample = bagging_samples[i]
    estimator = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=6000, random_state=42 + i)
    estimator.fit(X_sample, y_sample)
    estimators.append(estimator)
    loss_curves.append(estimator.loss_curve_)

# Vykreslite krivky tréningových strát
plt.figure(figsize=(15, 7))
for i, loss_curve in enumerate(loss_curves):
    plt.plot(loss_curve, label=f"Model {i + 1}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curves for Individual Models")
plt.legend()
plt.show()

# Predpovedajte pomocou hlasovania väčšinou hlasov
y_pred_ensemble = np.zeros(y_test.shape)

for estimator in estimators:
    y_pred_individual = estimator.predict(X_test)
    y_pred_ensemble += y_pred_individual

y_pred_ensemble = np.round(y_pred_ensemble / num_estimators).astype(int)

# Hodnotenie modelu súboru
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

# Presnosti jednotlivých modelov
individual_accuracies = []

for estimator in estimators:
    y_pred_individual = estimator.predict(X_test)
    individual_accuracy = accuracy_score(y_test, y_pred_individual)
    individual_accuracies.append(individual_accuracy)

def add_value_labels(ax, spacing=5):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        space = spacing
        va = 'bottom'

        label = "{:.2f}".format(y_value)
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va)

ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

# Vypočítať presnosť, odvolanie a skóre F1 pre model súboru
ensemble_precision = precision_score(y_test, y_pred_ensemble, average='macro')
ensemble_recall = recall_score(y_test, y_pred_ensemble, average='macro')
ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='macro')

# Individual model accuracies, precisions, recalls, and F1 scores
individual_accuracies = []
individual_recalls = []
individual_f1s = []
individual_log_losses = []
y_pred_ensemble = np.zeros(y_test.shape)
for estimator in estimators:
    y_pred_individual = estimator.predict(X_test)
    y_pred_ensemble += y_pred_individual

y_pred_ensemble = np.round(y_pred_ensemble / num_estimators).astype(int)
ensemble_proba = np.zeros((y_test.shape[0], len(np.unique(iris.target))))

for estimator in estimators:
    y_pred_individual = estimator.predict(X_test)
    y_proba_individual = estimator.predict_proba(X_test)

    individual_accuracy = accuracy_score(y_test, y_pred_individual)
    individual_recall = recall_score(y_test, y_pred_individual, average='macro')
    individual_f1 = f1_score(y_test, y_pred_individual, average='macro')

    individual_accuracies.append(individual_accuracy)
    individual_recalls.append(individual_recall)
    individual_f1s.append(individual_f1)

    y_proba_individual_extended = np.zeros((y_proba_individual.shape[0], len(np.unique(iris.target))))
    for i, label in enumerate(estimator.classes_):
        y_proba_individual_extended[:, label] = y_proba_individual[:, i]

    ensemble_proba += y_proba_individual_extended

    # Use y_proba_individual_extended instead of y_proba_individual
    individual_log_loss = log_loss(y_test, y_proba_individual_extended, labels=np.unique(iris.target))
    individual_log_losses.append(individual_log_loss)


ensemble_proba /= num_estimators
ensemble_log_loss = log_loss(y_test, ensemble_proba, labels=np.unique(iris.target))



# Pridať výstup metrických hodnôt
metrics = {
    "Accuracy": individual_accuracies + [ensemble_accuracy],
    "Recall": individual_recalls + [ensemble_recall],
    "F1 Score": individual_f1s + [ensemble_f1],
    "Loss": individual_log_losses + [ensemble_log_loss]
}

num_models = len(estimators) + 1  # Počet jednotlivých modelov + model súboru

for metric_name, metric_values in metrics.items():
    print(f"{metric_name}:")
    for i, value in enumerate(metric_values):
        model_label = f"Model {i}" if i < num_models - 1 else "Ensemble Model"
        print(f"{model_label}: {value:.4f}")
    print()

# vykreslite presnosť, presnosť, vyvolanie a skóre F1
fig, ax = plt.subplots(2, 2, figsize=(15, 15))

ax[0, 0].bar(range(num_estimators), individual_accuracies)
ax[0, 0].axhline(ensemble_accuracy, color='red', linestyle='--')
ax[0, 0].set_xlabel("Model")
ax[0, 0].set_ylabel("Accuracy")
ax[0, 0].set_title("Accuracy of Individual Models and the Ensemble Model")

ax[0, 1].bar(range(num_estimators), individual_log_losses)
ax[0, 1].axhline(ensemble_log_loss, color='red', linestyle='--')
ax[0, 1].set_xlabel("Model")
ax[0, 1].set_ylabel("Log Loss")
ax[0, 1].set_title("Log Loss of Individual Models and the Ensemble Model")

ax[1, 0].bar(range(num_estimators), individual_recalls)
ax[1, 0].axhline(ensemble_recall, color='red', linestyle='--')
ax[1, 0].set_xlabel("Model")
ax[1, 0].set_ylabel("Recall")
ax[1, 0].set_title("Recall of Individual Models and the Ensemble Model")

ax[1, 1].bar(range(num_estimators), individual_f1s)
ax[1, 1].axhline(ensemble_f1, color='red', linestyle='--')
ax[1, 1].set_xlabel("Model")
ax[1, 1].set_ylabel("F1 Score")
ax[1, 1].set_title("F1 Score of Individual Models and the Ensemble Model")

for i in range(2):
    for j in range(2):
        add_value_labels(ax[i, j])

plt.show()
