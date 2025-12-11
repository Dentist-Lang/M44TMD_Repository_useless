import ssl

# ssl._create_default_https_context = ssl._unverified_context
ssl._create_default_https_context = ssl._create_unverified_context

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # Adjust based on your environment

import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import numpy as np
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt

from mri_dataloader import CustomDataset
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import roc_curve, auc

torch.set_num_threads(8)
# The log file name is consistent with train.py, used for inference results
inference_log_file = "inference_log_M44TMD_ResNet50_fea-fus.txt"
writer = open(inference_log_file, "w+")

# --- Helper Functions: Consistent with train.py ---

def setup_seed():
    import numpy as np
    import random
    from torch.backends import cudnn

    # We typically don't need to set random seeds for inference, but the structure is preserved
    torch.manual_seed(np.random.randint(0, 2**32))
    torch.cuda.manual_seed_all(np.random.randint(0, 2**32))
    torch.cuda.manual_seed(np.random.randint(0, 2**32))
    np.random.seed(np.random.randint(0, 2**32))
    random.seed(np.random.randint(0, 2**32))
    cudnn.deterministic = False
    cudnn.benchmark = True

setup_seed()


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor()
])


# --- Model Definition: Consistent with train.py ---

def freeze_layers(model, freeze_layers_num):
    # Layers do not need to be frozen during inference, but the model structure must be consistent
    if freeze_layers_num == 0:
        return
    layers = list(model.children())
    total_layers = len(layers)
    layers_to_freeze = min(freeze_layers_num, total_layers)
    for i in range(layers_to_freeze):
        for param in layers[i].parameters():
            param.requires_grad = False


class MultiTaskModel(nn.Module):
    def __init__(self, freeze_layers_num=0):
        super(MultiTaskModel, self).__init__()
        resnet50 = models.resnet50(pretrained=True)

        self.closed_pd_features = nn.Sequential(*list(resnet50.children())[:-1])
        self.closed_t2w_features = nn.Sequential(*list(resnet50.children())[:-1])
        self.open_pd_features = nn.Sequential(*list(resnet50.children())[:-1])

        freeze_layers_num_per_branch = freeze_layers_num
        freeze_layers(self.closed_pd_features, freeze_layers_num_per_branch)
        freeze_layers(self.closed_t2w_features, freeze_layers_num_per_branch)
        freeze_layers(self.open_pd_features, freeze_layers_num_per_branch)

        self.clinical_features = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.task1_classifier = nn.Linear(2048 * 3 + 256, 2)
        self.task2_classifier = nn.Linear(2048 * 3 + 256, 3)
        self.task3_classifier = nn.Linear(2048 * 3 + 256, 2)

    def forward(self, closed_pd_images, closed_t2w_images, open_pd_images, clinical_data):

        # Ensure image input lists are not empty
        closed_pd_features = [self.closed_pd_features(img.to(device)) for img in closed_pd_images]
        closed_t2w_features = [self.closed_t2w_features(img.to(device)) for img in closed_t2w_images]
        open_pd_features = [self.open_pd_features(img.to(device)) for img in open_pd_images]

        closed_pd_features = torch.mean(torch.stack(closed_pd_features), dim=0)
        closed_t2w_features = torch.mean(torch.stack(closed_t2w_features), dim=0)
        open_pd_features = torch.mean(torch.stack(open_pd_features), dim=0)

        combined_feature = torch.cat([closed_pd_features, closed_t2w_features, open_pd_features], dim=1)
        combined_feature = combined_feature.view(combined_feature.size(0), -1)

        clinical_feature = self.clinical_features(clinical_data.to(device))

        final_feature = torch.cat([combined_feature, clinical_feature], dim=1)

        task1_output = self.task1_classifier(final_feature)
        task2_output = self.task2_classifier(final_feature)
        task3_output = self.task3_classifier(final_feature)

        return task1_output, task2_output, task3_output


def wilson_confidence_interval(successes, nobs, alpha=0.05):
    """Calculate the Wilson confidence interval"""
    if nobs == 0:
        return (0, 0)
    # Returns the lower and upper bounds
    return proportion_confint(successes, nobs, alpha=alpha, method='wilson')


def bootstrap_auc(y_true, y_score, n_bootstrap=1000, alpha=0.05, multi_class=False):
    """Calculate the mean and 95% confidence interval for AUC using bootstrapping"""
    try:
        rng = np.random.RandomState(42)
        auc_values = []
        n_size = len(y_true)

        for _ in range(n_bootstrap):
            indices = rng.randint(0, n_size, n_size)
            y_true_bs = y_true[indices]
            y_score_bs = y_score[indices] if not multi_class else y_score[indices, :]

            if not multi_class:
                # Binary classification AUC
                try:
                    # Ensure there are both positive and negative classes in the sample
                    if len(np.unique(y_true_bs)) < 2:
                        continue
                    auc_val = roc_auc_score(y_true_bs, y_score_bs)
                    auc_values.append(auc_val)
                except ValueError:
                    continue
            else:
                # Multi-class AUC (macro-average OVR)
                classes = np.unique(y_true_bs)
                if len(classes) < 2:
                    continue
                try:
                    # Get all possible classes, even if some are missing in the bootstrap sample
                    all_classes = np.unique(y_true)
                    y_true_bs_bin = label_binarize(y_true_bs, classes=all_classes)

                    # Assuming classes are [0, 1, 2] and y_score_bs has 3 columns
                    if y_score_bs.shape[1] != len(all_classes):
                        pass

                    auc_val = roc_auc_score(
                        y_true_bs_bin,
                        y_score_bs,
                        average='macro', multi_class='ovr'
                    )
                    auc_values.append(auc_val)
                except ValueError as e:
                    print(f"Bootstrap AUC for multi-class failed: {e}")
                    continue

        if len(auc_values) == 0:
            # Return default values if no valid AUC is found
            print("Warning: No valid AUC values from bootstrap samples.")
            return None, None, None

        print(f"Bootstrap AUC values count: {len(auc_values)}")
        print(f"Bootstrap AUC values range: {min(auc_values):.4f} - {max(auc_values):.4f}")

        sorted_scores = np.sort(auc_values)
        mean_val = np.mean(sorted_scores)
        lower_idx = int((alpha / 2) * len(sorted_scores))
        upper_idx = int((1 - alpha / 2) * len(sorted_scores))
        ci_lower = sorted_scores[lower_idx]
        # Ensure upper_idx is a valid index
        ci_upper = sorted_scores[upper_idx - 1 if upper_idx > 0 else 0]

        return mean_val, ci_lower, ci_upper
    except Exception as e:
        print(f"Error in bootstrap_auc: {e}")
        return None, None, None


def compute_sensitivity_specificity(conf_matrix, num_classes):
    """Calculate sensitivity and specificity for each class and their Wilson CIs"""
    results = {}
    for cls in range(num_classes):
        # TP/FN/FP/TN in a multi-class scenario
        TP = conf_matrix[cls, cls]
        # FN is the sum of row 'cls' excluding TP
        FN = conf_matrix[cls, :].sum() - TP
        # FP is the sum of column 'cls' excluding TP
        FP = conf_matrix[:, cls].sum() - TP
        # TN is the total number of samples minus TP, FN, and FP
        TN = conf_matrix.sum() - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Calculate confidence intervals
        sens_lower, sens_upper = wilson_confidence_interval(TP, TP + FN) if (TP + FN) > 0 else (0, 0)
        spec_lower, spec_upper = wilson_confidence_interval(TN, TN + FP) if (TN + FP) > 0 else (0, 0)

        results[cls] = (sensitivity, sens_lower, sens_upper, specificity, spec_lower, spec_upper)

    return results

def plot_roc_curve(test_labels_binarized, preds_binarized, task_name, num_classes, figures_folder):
    """Plot and save the ROC curve"""
    fpr = {}
    tpr = {}
    roc_auc_dict = {}
    plt.figure()

    if num_classes == 2:
        # Binary classification, assuming binarized labels [0, 1] and preds_binarized is the probability of the positive class
        fpr[1], tpr[1], _ = roc_curve(test_labels_binarized, preds_binarized)
        roc_auc_dict[1] = auc(fpr[1], tpr[1])
        plt.plot(fpr[1], tpr[1], label=f"Class 1 (AUC = {roc_auc_dict[1]:.4f})")
    else:
        # Multi-class, OVR
        for i in range(num_classes):
            # i is the class index
            fpr[i], tpr[i], _ = roc_curve(test_labels_binarized[:, i], preds_binarized[:, i])
            roc_auc_dict[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc_dict[i]:.4f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {task_name} (Inference)")
    plt.legend(loc="lower right")
    # Save the filename with "inference" to distinguish it from training figures
    plt.savefig(os.path.join(figures_folder, f"M44TMD_ResNet50_fea-fus_{task_name}_inference_roc_curve.png"))
    plt.close()

# --- Main Inference Logic ---

def inference(checkpoint_path, best_freeze_layers_num):
    """
    Loads the model and performs inference and evaluation on the test set.
    checkpoint_path: Path to the model weights file.
    best_freeze_layers_num: The optimal number of frozen layers determined during training, used to initialize the model.
    """
    print("--- Starting Inference ---")
    writer.write("--- Starting Inference ---\n")

    # Dataset and DataLoader settings
    root_folder = "../data"
    data_folder = "../data/image"
    label_folder = "../data/annotation"
    test_file = os.path.join(label_folder, "test.txt")

    test_dataset = CustomDataset(label_folder=label_folder, data_folder=data_folder,
                                 file=test_file, transform=transform, mode="test")
    batch_size = 8
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model loading
    model = MultiTaskModel(freeze_layers_num=best_freeze_layers_num)

    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs for inference!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        writer.write(f"Error: Checkpoint file not found: {checkpoint_path}\n")
        return

    # Load model weights
    print(f"Loading model weights: {checkpoint_path}")
    writer.write(f"Loading model weights: {checkpoint_path}\n")
    if torch.cuda.device_count() > 1:
        # Special handling for DataParallel models
        state_dict = torch.load(checkpoint_path, map_location=device)
        # If the state_dict of a DataParallel model was saved, keys might have a 'module.' prefix
        if list(state_dict.keys())[0].startswith('module.'):
            model.load_state_dict(state_dict)
        else:
            # If a single-GPU state_dict was saved, but the current model is DataParallel, add 'module.'
            new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
    else:
        # Single-GPU loading
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()

    # Create directories for figures
    figures_folder = "./figures/M44TMD_ResNet50_fea-fus"
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    # Store prediction results
    test_task1_labels, test_task1_preds, test_task1_outputs = [], [], []
    test_task2_labels, test_task2_preds, test_task2_outputs = [], [], []
    test_task3_labels, test_task3_preds, test_task3_outputs = [], [], []

    with torch.no_grad():
        for closed_pd_images, closed_t2w_images, open_pd_images, label1, label2, label3, clinical_data in tqdm(test_loader, desc="Inferencing"):
            label1, label2, label3 = label1.to(device), label2.to(device), label3.to(device)
            clinical_data = clinical_data.to(device)

            task1_output, task2_output, task3_output = model(closed_pd_images, closed_t2w_images, open_pd_images, clinical_data)

            _, task1_preds_batch = torch.max(task1_output, 1)
            _, task2_preds_batch = torch.max(task2_output, 1)
            _, task3_preds_batch = torch.max(task3_output, 1)

            test_task1_labels.extend(label1.cpu().numpy())
            test_task1_preds.extend(task1_preds_batch.cpu().numpy())
            test_task1_outputs.append(task1_output.cpu().numpy())

            test_task2_labels.extend(label2.cpu().numpy())
            test_task2_preds.extend(task2_preds_batch.cpu().numpy())
            test_task2_outputs.append(task2_output.cpu().numpy())

            test_task3_labels.extend(label3.cpu().numpy())
            test_task3_preds.extend(task3_preds_batch.cpu().numpy())
            test_task3_outputs.append(task3_output.cpu().numpy())

    # Concatenate results
    test_task1_outputs = np.concatenate(test_task1_outputs, axis=0)
    test_task2_outputs = np.concatenate(test_task2_outputs, axis=0)
    test_task3_outputs = np.concatenate(test_task3_outputs, axis=0)

    # Calculate metrics
    # --- Task 1 (Binary Classification) ---
    task1_probs = torch.softmax(torch.tensor(test_task1_outputs), dim=1).cpu().numpy()
    test_task1_accuracy = accuracy_score(test_task1_labels, test_task1_preds)
    acc_count_task1 = sum(np.array(test_task1_labels) == np.array(test_task1_preds))
    acc_lower_t1, acc_upper_t1 = wilson_confidence_interval(acc_count_task1, len(test_task1_labels))

    precision_task1, recall_task1, f1_task1, _ = precision_recall_fscore_support(
        test_task1_labels, test_task1_preds, average='weighted', zero_division=1)
    auc_task1 = roc_auc_score(
        label_binarize(test_task1_labels, classes=[0, 1]),
        task1_probs[:, 1], average='macro'
    )
    conf_matrix_task1 = confusion_matrix(test_task1_labels, test_task1_preds)
    tn_t1, fp_t1, fn_t1, tp_t1 = conf_matrix_task1.ravel() if conf_matrix_task1.shape == (2, 2) else (0, 0, 0, 0)
    sens_t1 = tp_t1 / (tp_t1 + fn_t1) if (tp_t1 + fn_t1) else 0
    spec_t1 = tn_t1 / (tn_t1 + fp_t1) if (tn_t1 + fp_t1) else 0
    sens_lower_t1, sens_upper_t1 = wilson_confidence_interval(tp_t1, tp_t1 + fn_t1) if (tp_t1 + fn_t1) > 0 else (0, 0)
    spec_lower_t1, spec_upper_t1 = wilson_confidence_interval(tn_t1, tn_t1 + fp_t1) if (tn_t1 + fp_t1) > 0 else (0, 0)

    mean_auc_t1, ci_lower_t1, ci_upper_t1 = bootstrap_auc(
        np.array(test_task1_labels), task1_probs[:, 1], n_bootstrap=1000, alpha=0.05, multi_class=False
    )
    if mean_auc_t1 is None: mean_auc_t1, ci_lower_t1, ci_upper_t1 = auc_task1, 0.0, 0.0

    # --- Task 2 (Three Classes) ---
    task2_probs = torch.softmax(torch.tensor(test_task2_outputs), dim=1).cpu().numpy()
    test_task2_labels_binarized = label_binarize(test_task2_labels, classes=[0, 1, 2])
    test_task2_accuracy = accuracy_score(test_task2_labels, test_task2_preds)
    acc_count_task2 = sum(np.array(test_task2_labels) == np.array(test_task2_preds))
    acc_lower_t2, acc_upper_t2 = wilson_confidence_interval(acc_count_task2, len(test_task2_labels))

    precision_task2, recall_task2, f1_task2, _ = precision_recall_fscore_support(
        test_task2_labels, test_task2_preds, average='weighted', zero_division=1)
    auc_task2 = roc_auc_score(
        test_task2_labels_binarized,
        task2_probs,
        average='macro', multi_class='ovr'
    )
    conf_matrix_task2 = confusion_matrix(test_task2_labels, test_task2_preds)
    sensitivity_specificity_task2 = compute_sensitivity_specificity(conf_matrix_task2, num_classes=3)

    mean_auc_t2, ci_lower_t2, ci_upper_t2 = bootstrap_auc(
        np.array(test_task2_labels),
        task2_probs,
        n_bootstrap=1000, alpha=0.05, multi_class=True
    )
    if mean_auc_t2 is None: mean_auc_t2, ci_lower_t2, ci_upper_t2 = auc_task2, 0.0, 0.0


    # --- Task 3 (Binary Classification) ---
    task3_probs = torch.softmax(torch.tensor(test_task3_outputs), dim=1).cpu().numpy()
    test_task3_accuracy = accuracy_score(test_task3_labels, test_task3_preds)
    acc_count_task3 = sum(np.array(test_task3_labels) == np.array(test_task3_preds))
    acc_lower_t3, acc_upper_t3 = wilson_confidence_interval(acc_count_task3, len(test_task3_labels))

    precision_task3, recall_task3, f1_task3, _ = precision_recall_fscore_support(
        test_task3_labels, test_task3_preds, average='weighted', zero_division=1)
    auc_task3 = roc_auc_score(
        label_binarize(test_task3_labels, classes=[0, 1]),
        task3_probs[:, 1], average='macro'
    )
    conf_matrix_task3 = confusion_matrix(test_task3_labels, test_task3_preds)
    tn_t3, fp_t3, fn_t3, tp_t3 = conf_matrix_task3.ravel() if conf_matrix_task3.shape == (2, 2) else (0, 0, 0, 0)
    sens_t3 = tp_t3 / (tp_t3 + fn_t3) if (tp_t3 + fn_t3) else 0
    spec_t3 = tn_t3 / (tn_t3 + fp_t3) if (tn_t3 + fp_t3) else 0
    sens_lower_t3, sens_upper_t3 = wilson_confidence_interval(tp_t3, tp_t3 + fn_t3) if (tp_t3 + fn_t3) > 0 else (0, 0)
    spec_lower_t3, spec_upper_t3 = wilson_confidence_interval(tn_t3, tn_t3 + fp_t3) if (tn_t3 + fp_t3) > 0 else (0, 0)

    mean_auc_t3, ci_lower_t3, ci_upper_t3 = bootstrap_auc(
        np.array(test_task3_labels), task3_probs[:, 1], n_bootstrap=1000, alpha=0.05, multi_class=False
    )
    if mean_auc_t3 is None: mean_auc_t3, ci_lower_t3, ci_upper_t3 = auc_task3, 0.0, 0.0


    # --- Print and Write Results ---

    print("\n--- Inference Results (Epoch: Final) ---")
    writer.write("\n--- Inference Results (Epoch: Final) ---\n")

    # Task 1 Results
    print(
        f"Task 1 Accuracy: {test_task1_accuracy:.4f} (95%CI: {acc_lower_t1:.4f}-{acc_upper_t1:.4f}), "
        f"Precision: {precision_task1:.4f}, Recall: {recall_task1:.4f}, F1: {f1_task1:.4f}, "
        f"AUC: {auc_task1:.4f}"
    )
    print(
        f"  Sensitivity: {sens_t1:.4f} (95%CI: {sens_lower_t1:.4f}-{sens_upper_t1:.4f}), "
        f"Specificity: {spec_t1:.4f} (95%CI: {spec_lower_t1:.4f}-{spec_upper_t1:.4f})"
    )
    print(f"  AUC Bootstrap (95% CI): {mean_auc_t1:.4f} [{ci_lower_t1:.4f}, {ci_upper_t1:.4f}]")
    writer.write(
        f"Task 1 Accuracy: {test_task1_accuracy:.4f} (95%CI: {acc_lower_t1:.4f}-{acc_upper_t1:.4f}), "
        f"Precision: {precision_task1:.4f}, Recall: {recall_task1:.4f}, F1: {f1_task1:.4f}, "
        f"AUC: {auc_task1:.4f}\n"
    )
    writer.write(
        f"  Sensitivity: {sens_t1:.4f} (95%CI: {sens_lower_t1:.4f}-{sens_upper_t1:.4f}), "
        f"Specificity: {spec_t1:.4f} (95%CI: {spec_lower_t1:.4f}-{spec_upper_t1:.4f})\n"
    )
    writer.write(f"  AUC Bootstrap (95% CI): {mean_auc_t1:.4f} [{ci_lower_t1:.4f}, {ci_upper_t1:.4f}]\n")
    writer.write(f"Task 1 Confusion Matrix:\n{conf_matrix_task1}\n")

    # Task 2 Results
    print(
        f"\nTask 2 Accuracy: {test_task2_accuracy:.4f} (95%CI: {acc_lower_t2:.4f}-{acc_upper_t2:.4f}), "
        f"Precision: {precision_task2:.4f}, Recall: {recall_task2:.4f}, F1: {f1_task2:.4f}, "
        f"AUC: {auc_task2:.4f}"
    )
    writer.write(
        f"\nTask 2 Accuracy: {test_task2_accuracy:.4f} (95%CI: {acc_lower_t2:.4f}-{acc_upper_t2:.4f}), "
        f"Precision: {precision_task2:.4f}, Recall: {recall_task2:.4f}, F1: {f1_task2:.4f}, "
        f"AUC: {auc_task2:.4f}\n"
    )
    for cls in range(3):
        sens, sens_lower, sens_upper, spec, spec_lower, spec_upper = sensitivity_specificity_task2[cls]
        print(f"  Class {cls}: Sensitivity={sens:.4f} (95%CI: {sens_lower:.4f}-{sens_upper:.4f}), "
              f"Specificity={spec:.4f} (95%CI: {spec_lower:.4f}-{spec_upper:.4f})")
        writer.write(f"  Class {cls}: Sensitivity={sens:.4f} (95%CI: {sens_lower:.4f}-{sens_upper:.4f}), "
                     f"Specificity={spec:.4f} (95%CI: {spec_lower:.4f}-{spec_upper:.4f})\n")
    print(f"  AUC Bootstrap (95% CI): {mean_auc_t2:.4f} [{ci_lower_t2:.4f}, {ci_upper_t2:.4f}]")
    writer.write(f"  AUC Bootstrap (95% CI): {mean_auc_t2:.4f} [{ci_lower_t2:.4f}, {ci_upper_t2:.4f}]\n")
    writer.write(f"Task 2 Confusion Matrix:\n{conf_matrix_task2}\n")

    # Task 3 Results
    print(
        f"\nTask 3 Accuracy: {test_task3_accuracy:.4f} (95%CI: {acc_lower_t3:.4f}-{acc_upper_t3:.4f}), "
        f"Precision: {precision_task3:.4f}, Recall: {recall_task3:.4f}, F1: {f1_task3:.4f}, "
        f"AUC: {auc_task3:.4f}"
    )
    print(
        f"  Sensitivity: {sens_t3:.4f} (95%CI: {sens_lower_t3:.4f}-{sens_upper_t3:.4f}), "
        f"Specificity: {spec_t3:.4f} (95%CI: {spec_lower_t3:.4f}-{spec_upper_t3:.4f})"
    )
    print(f"  AUC Bootstrap (95% CI): {mean_auc_t3:.4f} [{ci_lower_t3:.4f}, {ci_upper_t3:.4f}]")
    writer.write(
        f"\nTask 3 Accuracy: {test_task3_accuracy:.4f} (95%CI: {acc_lower_t3:.4f}-{acc_upper_t3:.4f}), "
        f"Precision: {precision_task3:.4f}, Recall: {recall_task3:.4f}, F1: {f1_task3:.4f}, "
        f"AUC: {auc_task3:.4f}\n"
    )
    writer.write(
        f"  Sensitivity: {sens_t3:.4f} (95%CI: {sens_lower_t3:.4f}-{sens_upper_t3:.4f}), "
        f"Specificity: {spec_t3:.4f} (95%CI: {spec_lower_t3:.4f}-{spec_upper_t3:.4f})\n"
    )
    writer.write(f"  AUC Bootstrap (95% CI): {mean_auc_t3:.4f} [{ci_lower_t3:.4f}, {ci_upper_t3:.4f}]\n")
    writer.write(f"Task 3 Confusion Matrix:\n{conf_matrix_task3}\n")


    # Plot ROC Curves
    plot_roc_curve(label_binarize(test_task1_labels, classes=[0, 1]),
                   task1_probs[:, 1], "Task1", 2, figures_folder)
    plot_roc_curve(label_binarize(test_task2_labels, classes=[0, 1, 2]),
                   task2_probs, "Task2", 3, figures_folder)
    plot_roc_curve(label_binarize(test_task3_labels, classes=[0, 1]),
                   task3_probs[:, 1], "Task3", 2, figures_folder)


# --- Execute Script ---

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming the best model was saved. We use 0 as a placeholder for the best_freeze_layers_num
    # but in a real-world scenario, this value should be retrieved from the training log.
    best_freeze_layers_num = 0
    checkpoint_path = "../checkpoint/best_model.pth"

    inference(checkpoint_path, best_freeze_layers_num)
    writer.close()
    print("Inference completed. Results written to", inference_log_file)