import torch


def mahalanobis(querys, mean, cov_inv, norm=2):
    """
    args:
        querys: [n, dim]
        mean: [dim]
        cov_inv: [dim, dim]
    return：
        [n]
    """
    diff = querys - mean
    # [n, dim] = ([n, dim] @ [dim, dim]) * [n, dim] = [n, dim] * [n, dim]
    maha_dis = torch.matmul(diff, cov_inv) * diff

    if norm == 2:
        return maha_dis.sum(dim=1)
    if norm == 1:
        return maha_dis.abs().sqrt().sum(dim=1)
    if norm == 'inf':
        return maha_dis.max(dim=1)

