from abc import ABC, abstractmethod

class Metric(ABC):
    @abstractmethod
    def compute_average(self, predictions, targets):
        pass

class LossMetric(Metric):
    def compute_average(self, predictions, targets):
        loss = 0.
        for prediction, target in zip(predictions, targets):
            loss += self.compute(prediction, target)
        return loss / len(prediction)
    
    @abstractmethod
    def compute(self, prediction, target):
        pass

class ClassificationMetric(Metric):    
    def compute_average(self, predictions, targets):
        prediction_labels = []
        target_labels = []
        for prediction, target in zip(predictions, targets):
            if not isinstance(prediction, list): prediction = prediction.tolist()
            prediction_labels.append(prediction.index(max(prediction)))
            target_labels.append(target.index(max(target)))    

        num_classes = len(set(target_labels))
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
        for target_label, prediction_label in zip(target_labels, prediction_labels):
            confusion_matrix[target_label][prediction_label] += 1

        return self.compute_matrix(confusion_matrix)
    
    @abstractmethod
    def compute_matrix(self, confusion_matrix):
        pass
    
class Accuracy(ClassificationMetric):
    def compute_matrix(self, confusion_matrix):
        correct = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
        total = sum(sum(row) for row in confusion_matrix)
        return (correct / total) * 100 if total > 0 else 0
    
class Precision(ClassificationMetric):
    def compute_matrix(self, confusion_matrix):
        num_classes = len(confusion_matrix)
        precision_per_class = []
        for i in range(num_classes):
            tp = confusion_matrix[i][i]
            fp = sum(confusion_matrix[j][i] for j in range(num_classes)) - tp
            if tp + fp > 0:
                precision_per_class.append(tp / (tp + fp))
            else:
                precision_per_class.append(0)
        return (sum(precision_per_class) / num_classes) * 100
    
class Recall(ClassificationMetric):
    def compute_matrix(self, confusion_matrix):
        num_classes = len(confusion_matrix)
        recall_per_class = []
        for i in range(num_classes):
            tp = confusion_matrix[i][i]
            fn = sum(confusion_matrix[i]) - tp
            if tp + fn > 0:
                recall_per_class.append(tp / (tp + fn))
            else:
                recall_per_class.append(0)
        return (sum(recall_per_class) / num_classes) * 100

class F1Score(ClassificationMetric):
    def compute_matrix(self, confusion_matrix):
        num_classes = len(confusion_matrix)
        f1_per_class = []
        for i in range(num_classes):
            tp = confusion_matrix[i][i]
            fp = sum(confusion_matrix[j][i] for j in range(num_classes)) - tp
            fn = sum(confusion_matrix[i]) - tp
            
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                f1_per_class.append(f1)
            else:
                f1_per_class.append(0)
        return (sum(f1_per_class) / num_classes) * 100