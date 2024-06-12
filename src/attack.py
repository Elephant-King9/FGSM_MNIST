import torch


def fgsm_attack(image, epsilon, data_grad):
    """
    Perform FGSM with
    :param image: 输入图片
    :param epsilon: 𝜀超参数
    :param data_grad: 梯度
    :return:
    """
    # 获取梯度方向
    sign_data_grad = data_grad.sign()
    # 对原始图像添加扰动
    perturbed_image = image + epsilon * sign_data_grad
    # 将生成的对抗样本的扰动控制在0~1之间
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# restores the tensors to their original scale
def denorm(batch, device, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        device:
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean, requires_grad=True).to(device)
    if isinstance(std, list):
        std = torch.tensor(std, requires_grad=True).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
