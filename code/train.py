import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from mri_dataloader import CustomDataset
import numpy as np
import torchvision.models as models


from statsmodels.stats.proportion import proportion_confint
from sklearn.utils import resample

torch.set_num_threads(8)
writer = open("log_M44TMD_ResNet50_fea-fus.txt", "w+")



def setup_seed():
    import numpy as np
    import random
    from torch.backends import cudnn

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


root_folder = "../data"
data_folder = "../data/image"
label_folder = "../data/annotation"
train_file = os.path.join(label_folder, "train.txt")
test_file = os.path.join(label_folder, "test.txt")


train_dataset = CustomDataset(label_folder=label_folder, data_folder=data_folder,
                              file=train_file, transform=transform, mode="train")
test_dataset = CustomDataset(label_folder=label_folder, data_folder=data_folder,
                             file=test_file, transform=transform, mode="test")


batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def freeze_layers(model, freeze_layers_num):


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

checkpoint_folder = "../checkpoint"
if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)

figures_folder = "./figures/M44TMD_ResNet50_fea-fus"
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

num_epochs = 80


def wilson_confidence_interval(successes, nobs, alpha=0.05):

    if nobs == 0:
        return (0, 0)
    return proportion_confint(successes, nobs, alpha=alpha, method='wilson')



def bootstrap_auc(y_true, y_score, n_bootstrap=1000, alpha=0.05, multi_class=False):
    try:
        rng = np.random.RandomState(42)
        auc_values = []
        n_size = len(y_true)

        for _ in range(n_bootstrap):
            indices = rng.randint(0, n_size, n_size)
            y_true_bs = y_true[indices]
            y_score_bs = y_score[indices] if not multi_class else y_score[indices, :]

            if not multi_class:
                try:
                    auc_val = roc_auc_score(y_true_bs, y_score_bs)
                    auc_values.append(auc_val)
                except ValueError:
                    continue
            else:
                classes = np.unique(y_true_bs)
                if len(classes) < 2:
                    continue
                try:
                    auc_val = roc_auc_score(
                        label_binarize(y_true_bs, classes=classes),
                        y_score_bs,
                        average='macro', multi_class='ovr'
                    )
                    auc_values.append(auc_val)
                except ValueError:
                    continue

        if len(auc_values) == 0:
            raise ValueError("No valid AUC values from bootstrap samples.")

        print(f"Bootstrap AUC values count: {len(auc_values)}")
        print(f"Bootstrap AUC values range: {min(auc_values):.4f} - {max(auc_values):.4f}")

        sorted_scores = np.sort(auc_values)
        mean_val = np.mean(sorted_scores)
        lower_idx = int((alpha / 2) * len(sorted_scores))
        upper_idx = int((1 - alpha / 2) * len(sorted_scores))
        ci_lower = sorted_scores[lower_idx]
        ci_upper = sorted_scores[upper_idx - 1 if upper_idx > 0 else 0]

        return mean_val, ci_lower, ci_upper
    except Exception as e:
        print(f"Error in bootstrap_auc: {e}")
        return None, None, None


def compute_sensitivity_specificity(conf_matrix, num_classes):

    results = {}
    for cls in range(num_classes):
        TP = conf_matrix[cls, cls]
        FN = conf_matrix[cls, :].sum() - TP
        FP = conf_matrix[:, cls].sum() - TP
        TN = conf_matrix.sum() - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        sens_lower, sens_upper = wilson_confidence_interval(TP, TP + FN) if (TP + FN) > 0 else (0, 0)
        spec_lower, spec_upper = wilson_confidence_interval(TN, TN + FP) if (TN + FP) > 0 else (0, 0)

        results[cls] = (sensitivity, sens_lower, sens_upper, specificity, spec_lower, spec_upper)

    return results


def plot_roc_curve(test_labels_binarized, preds_binarized, task_name, num_classes, epoch):
    fpr = {}
    tpr = {}
    roc_auc_dict = {}
    plt.figure()

    if num_classes == 2:
        fpr[1], tpr[1], _ = roc_curve(test_labels_binarized, preds_binarized)
        roc_auc_dict[1] = auc(fpr[1], tpr[1])
        plt.plot(fpr[1], tpr[1], label=f"Class 1 (AUC = {roc_auc_dict[1]:.2f})")
    else:
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(test_labels_binarized[:, i], preds_binarized[:, i])
            roc_auc_dict[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc_dict[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {task_name} (Epoch {epoch})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(figures_folder, f"M44TMD_ResNet50_fea-fus_{task_name}_epoch_{epoch}_roc_curve.png"))
    plt.close()

def train_and_evaluate(freeze_layers_num):
    print(f"Training with {freeze_layers_num} frozen layers")

    model = MultiTaskModel(freeze_layers_num=freeze_layers_num)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=None, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.reduction = reduction
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        def forward(self, logits, targets):
            ce_loss = self.ce_loss(logits, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

            if self.alpha is not None:
                alpha_factor = self.alpha.clone().detach().to(logits.device)
                focal_loss = alpha_factor[targets] * focal_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

    task1_class_counts = [374, 685]
    task1_alpha = [sum(task1_class_counts) / (len(task1_class_counts) * count)
                   for count in task1_class_counts]
    task1_alpha_tensor = torch.tensor(task1_alpha).to(device)

    criterion_task1 = FocalLoss(alpha=task1_alpha_tensor, gamma=2)
    criterion_task2 = nn.CrossEntropyLoss()
    criterion_task3 = nn.CrossEntropyLoss()

    best_auc_task1 = 0
    best_auc_task2 = 0
    best_auc_task3 = 0
    best_overall_auc = 0
    best_accuracy_task1 = 0
    best_accuracy_task2 = 0
    best_accuracy_task3 = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        train_task1_loss = 0
        train_task2_loss = 0
        train_task3_loss = 0
        train_task1_labels, train_task1_preds = [], []
        train_task2_labels, train_task2_preds = [], []
        train_task3_labels, train_task3_preds = [], []
        train_task1_outputs, train_task2_outputs, train_task3_outputs = [], [], []
        total_batches = len(train_loader)

        for i, (closed_pd_images, closed_t2w_images, open_pd_images, label1, label2, label3, clinical_data) in enumerate(train_loader):
            label1, label2, label3 = label1.to(device), label2.to(device), label3.to(device)
            clinical_data = clinical_data.to(device)

            optimizer.zero_grad()
            task1_output, task2_output, task3_output = model(closed_pd_images, closed_t2w_images, open_pd_images, clinical_data)

            loss_task1 = criterion_task1(task1_output, label1)
            loss_task2 = criterion_task2(task2_output, label2)
            loss_task3 = criterion_task3(task3_output, label3.long())
            loss = loss_task1 + loss_task2 + loss_task3
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_task1_loss += loss_task1.item()
            train_task2_loss += loss_task2.item()
            train_task3_loss += loss_task3.item()

            _, task1_preds_batch = torch.max(task1_output, 1)
            _, task2_preds_batch = torch.max(task2_output, 1)
            _, task3_preds_batch = torch.max(task3_output, 1)

            train_task1_labels.extend(label1.cpu().numpy())
            train_task1_preds.extend(task1_preds_batch.cpu().numpy())
            train_task2_labels.extend(label2.cpu().numpy())
            train_task2_preds.extend(task2_preds_batch.cpu().numpy())
            train_task3_labels.extend(label3.cpu().numpy())
            train_task3_preds.extend(task3_preds_batch.cpu().numpy())

            train_task1_outputs.append(task1_output.detach().cpu().numpy())
            train_task2_outputs.append(task2_output.detach().cpu().numpy())
            train_task3_outputs.append(task3_output.detach().cpu().numpy())

        avg_train_loss = train_loss / total_batches
        avg_train_task1_loss = train_task1_loss / total_batches
        avg_train_task2_loss = train_task2_loss / total_batches
        avg_train_task3_loss = train_task3_loss / total_batches

        train_task1_accuracy = accuracy_score(train_task1_labels, train_task1_preds)
        train_task2_accuracy = accuracy_score(train_task2_labels, train_task2_preds)
        train_task3_accuracy = accuracy_score(train_task3_labels, train_task3_preds)

        precision_task1, recall_task1, f1_task1, _ = precision_recall_fscore_support(
            train_task1_labels, train_task1_preds, average='weighted', zero_division=1)
        precision_task2, recall_task2, f1_task2, _ = precision_recall_fscore_support(
            train_task2_labels, train_task2_preds, average='weighted', zero_division=1)
        precision_task3, recall_task3, f1_task3, _ = precision_recall_fscore_support(
            train_task3_labels, train_task3_preds, average='weighted', zero_division=1)

        train_task1_outputs = np.concatenate(train_task1_outputs, axis=0)
        train_task2_outputs = np.concatenate(train_task2_outputs, axis=0)
        train_task3_outputs = np.concatenate(train_task3_outputs, axis=0)

        train_task1_probs = torch.softmax(torch.from_numpy(train_task1_outputs), dim=1).cpu().numpy()
        train_task2_probs = torch.softmax(torch.from_numpy(train_task2_outputs), dim=1).cpu().numpy()
        train_task3_probs = torch.softmax(torch.from_numpy(train_task3_outputs), dim=1).cpu().numpy()


        auc_task1 = roc_auc_score(
            train_task1_labels,
            train_task1_probs[:, 1],
        )

        train_task2_labels_binarized = label_binarize(train_task2_labels, classes=[0, 1, 2])
        auc_task2 = roc_auc_score(
            train_task2_labels_binarized,
            train_task2_probs,
            average='macro',
            multi_class='ovr'
        )

        auc_task3 = roc_auc_score(
            train_task3_labels,
            train_task3_probs[:, 1],
        )

        print(f"Train epoch={epoch}, Avg Loss={avg_train_loss:.4f},"
              f" Task1 Loss={avg_train_task1_loss:.4f},"
              f" Task2 Loss={avg_train_task2_loss:.4f},"
              f" Task3 Loss={avg_train_task3_loss:.4f}")
        print(f"Train epoch={epoch}, Task1 accuracy={train_task1_accuracy},"
              f" Precision={precision_task1:.4f}, Recall={recall_task1:.4f},"
              f" F1={f1_task1:.4f}, AUC={auc_task1:.4f}",
              file=writer)
        print(f"Train epoch={epoch}, Task2 accuracy={train_task2_accuracy},"
              f" Precision={precision_task2:.4f}, Recall={recall_task2:.4f},"
              f" F1={f1_task2:.4f}, AUC={auc_task2:.4f}",
              file=writer)
        print(f"Train epoch={epoch}, Task3 accuracy={train_task3_accuracy},"
              f" Precision={precision_task3:.4f}, Recall={recall_task3:.4f},"
              f" F1={f1_task3:.4f}, AUC={auc_task3:.4f}",
              file=writer)

        model.eval()
        test_task1_labels, test_task1_preds = [], []
        test_task1_outputs = []
        test_task2_labels, test_task2_preds = [], []
        test_task2_outputs = []
        test_task3_labels, test_task3_preds = [], []
        test_task3_outputs = []

        with torch.no_grad():
            for closed_pd_images, closed_t2w_images, open_pd_images, label1, label2, label3, clinical_data in test_loader:
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

        test_task1_outputs = np.concatenate(test_task1_outputs, axis=0)
        test_task2_outputs = np.concatenate(test_task2_outputs, axis=0)
        test_task3_outputs = np.concatenate(test_task3_outputs, axis=0)

        assert len(test_task2_labels) == test_task2_outputs.shape[0], \
            f"Inconsistent lengths: labels={len(test_task2_labels)}, outputs={test_task2_outputs.shape[0]}"

        test_task1_accuracy = accuracy_score(test_task1_labels, test_task1_preds)
        test_task2_accuracy = accuracy_score(test_task2_labels, test_task2_preds)
        test_task3_accuracy = accuracy_score(test_task3_labels, test_task3_preds)

        acc_count_task1 = sum(np.array(test_task1_labels) == np.array(test_task1_preds))
        acc_lower_t1, acc_upper_t1 = wilson_confidence_interval(acc_count_task1, len(test_task1_labels))

        acc_count_task2 = sum(np.array(test_task2_labels) == np.array(test_task2_preds))
        acc_lower_t2, acc_upper_t2 = wilson_confidence_interval(acc_count_task2, len(test_task2_labels))

        acc_count_task3 = sum(np.array(test_task3_labels) == np.array(test_task3_preds))
        acc_lower_t3, acc_upper_t3 = wilson_confidence_interval(acc_count_task3, len(test_task3_labels))

        precision_task1, recall_task1, f1_task1, _ = precision_recall_fscore_support(
            test_task1_labels, test_task1_preds, average='weighted', zero_division=1)
        task1_probs = torch.softmax(torch.tensor(test_task1_outputs), dim=1).cpu().numpy()
        auc_task1 = roc_auc_score(
            label_binarize(test_task1_labels, classes=[0, 1]),
            task1_probs[:, 1], average='macro'
        )

        precision_task2, recall_task2, f1_task2, _ = precision_recall_fscore_support(
            test_task2_labels, test_task2_preds, average='weighted', zero_division=1)
        task2_probs = torch.softmax(torch.tensor(test_task2_outputs), dim=1).cpu().numpy()
        test_task2_labels_binarized = label_binarize(test_task2_labels, classes=[0, 1, 2])
        auc_task2 = roc_auc_score(
            test_task2_labels_binarized,
            task2_probs,
            average='macro', multi_class='ovr'
        )

        precision_task3, recall_task3, f1_task3, _ = precision_recall_fscore_support(
            test_task3_labels, test_task3_preds, average='weighted', zero_division=1)
        task3_probs = torch.softmax(torch.tensor(test_task3_outputs), dim=1).cpu().numpy()
        auc_task3 = roc_auc_score(
            label_binarize(test_task3_labels, classes=[0, 1]),
            task3_probs[:, 1], average='macro'
        )

        conf_matrix_task1 = confusion_matrix(test_task1_labels, test_task1_preds)
        conf_matrix_task2 = confusion_matrix(test_task2_labels, test_task2_preds)
        conf_matrix_task3 = confusion_matrix(test_task3_labels, test_task3_preds)

        if conf_matrix_task1.shape == (2, 2):
            tn_t1, fp_t1, fn_t1, tp_t1 = conf_matrix_task1.ravel()
            sens_t1 = tp_t1 / (tp_t1 + fn_t1) if (tp_t1 + fn_t1) else 0
            spec_t1 = tn_t1 / (tn_t1 + fp_t1) if (tn_t1 + fp_t1) else 0

            if (tp_t1 + fn_t1) > 0:
                sens_lower_t1, sens_upper_t1 = wilson_confidence_interval(tp_t1, tp_t1 + fn_t1)
            else:
                sens_lower_t1, sens_upper_t1 = (0, 0)
            if (tn_t1 + fp_t1) > 0:
                spec_lower_t1, spec_upper_t1 = wilson_confidence_interval(tn_t1, tn_t1 + fp_t1)
            else:
                spec_lower_t1, spec_upper_t1 = (0, 0)
        else:
            sens_t1, spec_t1 = 0, 0
            sens_lower_t1, sens_upper_t1 = (0, 0)
            spec_lower_t1, spec_upper_t1 = (0, 0)

        if conf_matrix_task3.shape == (2, 2):
            tn_t3, fp_t3, fn_t3, tp_t3 = conf_matrix_task3.ravel()
            sens_t3 = tp_t3 / (tp_t3 + fn_t3) if (tp_t3 + fn_t3) else 0
            spec_t3 = tn_t3 / (tn_t3 + fp_t3) if (tn_t3 + fp_t3) else 0

            if (tp_t3 + fn_t3) > 0:
                sens_lower_t3, sens_upper_t3 = wilson_confidence_interval(tp_t3, tp_t3 + fn_t3)
            else:
                sens_lower_t3, sens_upper_t3 = (0, 0)
            if (tn_t3 + fp_t3) > 0:
                spec_lower_t3, spec_upper_t3 = wilson_confidence_interval(tn_t3, tn_t3 + fp_t3)
            else:
                spec_lower_t3, spec_upper_t3 = (0, 0)
        else:
            sens_t3, spec_t3 = 0, 0
            sens_lower_t3, sens_upper_t3 = (0, 0)
            spec_lower_t3, spec_upper_t3 = (0, 0)

        sensitivity_specificity_task2 = compute_sensitivity_specificity(conf_matrix_task2, num_classes=3)

        print(
            f"Test epoch={epoch}, Task1 accuracy={test_task1_accuracy:.4f} (95%CI: {acc_lower_t1:.4f}-{acc_upper_t1:.4f}), "
            f"Precision={precision_task1:.4f}, Recall={recall_task1:.4f}, F1={f1_task1:.4f}, "
            f"AUC={auc_task1:.4f}, "
            f"AUC_bootstrap=({auc_task1:.4f}, 95%CI: {auc_task1:.4f}-{auc_task1:.4f})")
        print(f"    Sensitivity={sens_t1:.4f} (95%CI: {sens_lower_t1:.4f}-{sens_upper_t1:.4f}), "
              f"Specificity={spec_t1:.4f} (95%CI: {spec_lower_t1:.4f}-{spec_upper_t1:.4f})")

        print(
            f"Test epoch={epoch}, Task1 accuracy={test_task1_accuracy:.4f} (95%CI: {acc_lower_t1:.4f}-{acc_upper_t1:.4f}), "
            f"Precision={precision_task1:.4f}, Recall={recall_task1:.4f}, F1={f1_task1:.4f}, "
            f"AUC={auc_task1:.4f}, "
            f"AUC_bootstrap=({auc_task1:.4f}, 95%CI: {auc_task1:.4f}-{auc_task1:.4f})",
            file=writer)
        print(f"    Sensitivity={sens_t1:.4f} (95%CI: {sens_lower_t1:.4f}-{sens_upper_t1:.4f}), "
              f"Specificity={spec_t1:.4f} (95%CI: {spec_lower_t1:.4f}-{spec_upper_t1:.4f})",
              file=writer)

        for cls in range(3):
            sens, sens_lower, sens_upper, spec, spec_lower, spec_upper = sensitivity_specificity_task2[cls]
            print(f"Test epoch={epoch}, Task2 Class {cls}: "
                  f"Sensitivity={sens:.4f} (95%CI: {sens_lower:.4f}-{sens_upper:.4f}), "
                  f"Specificity={spec:.4f} (95%CI: {spec_lower:.4f}-{spec_upper:.4f})")
            print(f"    Task2 Class {cls}: Sensitivity={sens:.4f} (95%CI: {sens_lower:.4f}-{sens_upper:.4f}), "
                  f"Specificity={spec:.4f} (95%CI: {spec_lower:.4f}-{spec_upper:.4f})",
                  file=writer)

        print(
            f"Test epoch={epoch}, Task2 accuracy={test_task2_accuracy:.4f} (95%CI: {acc_lower_t2:.4f}-{acc_upper_t2:.4f}), "
            f"Precision={precision_task2:.4f}, Recall={recall_task2:.4f}, F1={f1_task2:.4f}, "
            f"AUC={auc_task2:.4f}, "
            f"AUC_bootstrap=({auc_task2:.4f}, 95%CI: {auc_task2:.4f}-{auc_task2:.4f})")
        print(
            f"Test epoch={epoch}, Task2 accuracy={test_task2_accuracy:.4f} (95%CI: {acc_lower_t2:.4f}-{acc_upper_t2:.4f}), "
            f"Precision={precision_task2:.4f}, Recall={recall_task2:.4f}, F1={f1_task2:.4f}, "
            f"AUC={auc_task2:.4f}, "
            f"AUC_bootstrap=({auc_task2:.4f}, 95%CI: {auc_task2:.4f}-{auc_task2:.4f})",
            file=writer)
        print(f"    Task2 Class Sensitivity and Specificity already printed above.")

        print(f"Test epoch={epoch}, Task1 Confusion Matrix:\n{conf_matrix_task1}",
              file=writer)
        print(f"Test epoch={epoch}, Task2 Confusion Matrix:\n{conf_matrix_task2}",
              file=writer)
        print(f"Test epoch={epoch}, Task3 Confusion Matrix:\n{conf_matrix_task3}",
              file=writer)

        mean_auc_t1, ci_lower_t1, ci_upper_t1 = bootstrap_auc(
            np.array(test_task1_labels), task1_probs[:, 1], n_bootstrap=1000, alpha=0.05, multi_class=False
        )

        mean_auc_t2, ci_lower_t2, ci_upper_t2 = bootstrap_auc(
            np.array(test_task2_labels),
            task2_probs,
            n_bootstrap=1000, alpha=0.05, multi_class=True
        )

        mean_auc_t3, ci_lower_t3, ci_upper_t3 = bootstrap_auc(
            np.array(test_task3_labels), task3_probs[:, 1], n_bootstrap=1000, alpha=0.05, multi_class=False
        )
        if ci_lower_t3 is None or ci_upper_t3 is None:
            print(f"Bootstrap AUC failed for Task3. Setting default values.")
            ci_lower_t3, ci_upper_t3 = 0.0, 0.0

        print(f"Test epoch={epoch}, Task1 AUC：{auc_task1:.4f}, 95% CI: [{ci_lower_t1:.4f}, {ci_upper_t1:.4f}]")
        print(f"    Task2 AUC：{auc_task2:.4f}, 95% CI: [{ci_lower_t2:.4f}, {ci_upper_t2:.4f}]")
        print(f"    Task3 AUC：{auc_task3:.4f}, 95% CI: [{ci_lower_t3:.4f}, {ci_upper_t3:.4f}]")

        print(f"Test epoch={epoch}, Task1 AUC：{auc_task1:.4f}, 95% CI: [{ci_lower_t1:.4f}, {ci_upper_t1:.4f}]",
              file=writer)
        print(f"    Task2 AUC：{auc_task2:.4f}, 95% CI: [{ci_lower_t2:.4f}, {ci_upper_t2:.4f}]",
              file=writer)
        print(f"    Task3 AUC：{auc_task3:.4f}, 95% CI: [{ci_lower_t3:.4f}, {ci_upper_t3:.4f}]",
              file=writer)

        print(
            f"Test epoch={epoch}, Task3 accuracy={test_task3_accuracy:.4f} (95%CI: {acc_lower_t3:.4f}-{acc_upper_t3:.4f}), "
            f"Precision={precision_task3:.4f}, Recall={recall_task3:.4f}, F1={f1_task3:.4f}, "
            f"AUC={auc_task3:.4f}, "
            f"AUC_bootstrap=({auc_task3:.4f}, 95%CI: {ci_lower_t3:.4f}-{ci_upper_t3:.4f})")
        print(f"    Sensitivity={sens_t3:.4f} (95%CI: {sens_lower_t3:.4f}-{sens_upper_t3:.4f}), "
              f"Specificity={spec_t3:.4f} (95%CI: {spec_lower_t3:.4f}-{spec_upper_t3:.4f})")

        print(
            f"Test epoch={epoch}, Task3 accuracy={test_task3_accuracy:.4f} (95%CI: {acc_lower_t3:.4f}-{acc_upper_t3:.4f}), "
            f"Precision={precision_task3:.4f}, Recall={recall_task3:.4f}, F1={f1_task3:.4f}, "
            f"AUC={auc_task3:.4f}, "
            f"AUC_bootstrap=({auc_task3:.4f}, 95%CI: {ci_lower_t3:.4f}-{ci_upper_t3:.4f})",
            file=writer)
        print(f"    Sensitivity={sens_t3:.4f} (95%CI: {sens_lower_t3:.4f}-{sens_upper_t3:.4f}), "
              f"Specificity={spec_t3:.4f} (95%CI: {spec_lower_t3:.4f}-{spec_upper_t3:.4f})",
              file=writer)

        plot_roc_curve(label_binarize(test_task1_labels, classes=[0, 1]),
                       task1_probs[:, 1], "Task1", 2, epoch)
        plot_roc_curve(label_binarize(test_task2_labels, classes=[0, 1, 2]),
                       task2_probs, "Task2", 3, epoch)
        plot_roc_curve(label_binarize(test_task3_labels, classes=[0, 1]),
                       task3_probs[:, 1], "Task3", 2, epoch)

        if auc_task1 > best_auc_task1:
            best_auc_task1 = auc_task1
        if auc_task2 > best_auc_task2:
            best_auc_task2 = auc_task2
        if auc_task3 > best_auc_task3:
            best_auc_task3 = auc_task3

        if test_task1_accuracy > best_accuracy_task1:
            best_accuracy_task1 = test_task1_accuracy
        if test_task2_accuracy > best_accuracy_task2:
            best_accuracy_task2 = test_task2_accuracy
        if test_task3_accuracy > best_accuracy_task3:
            best_accuracy_task3 = test_task3_accuracy

        overall_auc = (auc_task1 + auc_task2 + auc_task3) / 3
        if overall_auc > best_overall_auc:
            best_overall_auc = overall_auc

        print(
            f"Best AUCs so far - Task1: {best_auc_task1:.4f}, "
            f"Task2: {best_auc_task2:.4f}, Task3: {best_auc_task3:.4f}, "
            f"Overall: {best_overall_auc:.4f}")
        print(
            f"Best accuracy so far - Task1: {best_accuracy_task1:.4f}, "
            f"Task2: {best_accuracy_task2:.4f}, Task3: {best_accuracy_task3:.4f}",
            file=writer)

        print(
            f"Best AUCs so far - Task1: {best_auc_task1:.4f}, "
            f"Task2: {best_auc_task2:.4f}, Task3: {best_auc_task3:.4f}, "
            f"Overall: {best_overall_auc:.4f}",
            file=writer)
        print(
            f"Best accuracy so far - Task1: {best_accuracy_task1:.4f}, "
            f"Task2: {best_accuracy_task2:.4f}, Task3: {best_accuracy_task3:.4f}",
            file=writer)

        writer.flush()

    checkpoint_path = os.path.join(checkpoint_folder, "best_model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    print(f"Model saved to {checkpoint_path}", file=writer)

    return best_overall_auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

freeze_options = list(range(0, 14))
best_overall_auc = 0
best_x = 0

for x in freeze_options:
    current_auc = train_and_evaluate(freeze_layers_num=x)
    print(f"Finished training with {x} frozen layers. Overall AUC: {current_auc:.4f}")
    print(f"Finished training with {x} frozen layers. Overall AUC: {current_auc:.4f}", file=writer)

    if current_auc > best_overall_auc:
        best_overall_auc = current_auc
        best_x = x

print(f"Best number of frozen layers: {best_x} with Overall AUC: {best_overall_auc:.4f}")
print(f"Best number of frozen layers: {best_x} with Overall AUC: {best_overall_auc:.4f}", file=writer)

writer.close()