import torch
import torch.nn.functional as F


def distillation_loss(student_logits, labels, teacher_logits, temp=5.0, alpha=0.7):
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    if temp <= 0:
        raise ValueError("temp must be > 0.")

    kl = F.kl_div(
        F.log_softmax(student_logits / temp, dim=1),
        F.softmax(teacher_logits / temp, dim=1),
        reduction="batchmean",
    ) * (temp * temp)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1.0 - alpha) * ce


@torch.no_grad()
def predict_teacher(teacher, inputs):
    teacher.eval()
    return teacher(inputs)
