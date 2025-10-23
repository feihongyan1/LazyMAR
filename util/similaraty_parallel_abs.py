import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool, Manager, set_start_method


def compute_cosine_similarity(z1, z2):
    """计算两个 tensor 的余弦相似度"""
    z1_norm = z1 / z1.norm(dim=1, keepdim=True)
    z2_norm = z2 / z2.norm(dim=1, keepdim=True)
    return (z1_norm * z2_norm).sum(dim=1)  # 返回 (64*256,) 的余弦相似度向量


def compute_l2_distance(z1, z2):
    """计算两个 tensor 的 L2 距离"""
    z_mean = (z1 - z2).norm(dim=1)
    z_mean_f = z_mean.mean()
    return z_mean_f


def compute_abs_distance(z1, z2):
    """计算两个 tensor 的 L2 距离"""
    z_abs_mean = (z1 - z2).abs().mean()
    return z_abs_mean


def calculate_similarity(i, j, cache_dir, device, avg_similarity_matrix_predicted, avg_similarity_matrix_unpredicted,
                         avg_similarity_matrix_mixed):
    print(f"Calculating similarity between step {i} and step {j}")
    """计算单对 (i, j) 的相似性，并更新共享矩阵"""
    z1 = torch.load(os.path.join(cache_dir, f"z_step_{i}.pt")).to(device)
    mask1 = torch.load(os.path.join(cache_dir, f"mask_step_{i}.pt")).to(device)
    z2 = torch.load(os.path.join(cache_dir, f"z_step_{j}.pt")).to(device)
    mask2 = torch.load(os.path.join(cache_dir, f"mask_step_{j}.pt")).to(device)

    # 计算未预测部分的相似度
    mask_unpredicted_common = (mask1 & mask2).view(-1)
    z1_unpredicted = z1.view(-1, z1.size(-1))[mask_unpredicted_common]
    z2_unpredicted = z2.view(-1, z2.size(-1))[mask_unpredicted_common]

    if z1_unpredicted.size(0) > 0:
        with torch.no_grad():
            z1_abs_mean = z1_unpredicted.abs().mean(dim=1)
            z2_abs_mean = z2_unpredicted.abs().mean(dim=1)
            feature_abs_diff = (z1_unpredicted - z2_unpredicted).abs().mean(dim=1)
            normalized_measure = feature_abs_diff / ((z1_abs_mean + z2_abs_mean) / 2)
            sim_unpredicted = normalized_measure.mean().cpu().numpy()

        avg_similarity_matrix_unpredicted[i][j] = avg_similarity_matrix_unpredicted[j][i] = sim_unpredicted

    # 计算已预测部分的相似度
    mask_predicted_common = (~mask1 & ~mask2).view(-1)
    z1_predicted = z1.view(-1, z1.size(-1))[mask_predicted_common]
    z2_predicted = z2.view(-1, z2.size(-1))[mask_predicted_common]

    if z1_predicted.size(0) > 0:
        with torch.no_grad():
            z1_abs_mean = z1_predicted.abs().mean(dim=1)
            z2_abs_mean = z2_predicted.abs().mean(dim=1)
            feature_abs_diff = (z1_predicted - z2_predicted).abs().mean(dim=1)
            normalized_measure = feature_abs_diff / ((z1_abs_mean + z2_abs_mean) / 2)
            sim_predicted = normalized_measure.mean().cpu().numpy()

        avg_similarity_matrix_predicted[i][j] = avg_similarity_matrix_predicted[j][i] = sim_predicted

    # 计算一个step预测另一个step未预测的相似度
    mask_mixed = (mask1 & ~mask2).view(-1) | (~mask1 & mask2).view(-1)
    z1_mixed = z1.view(-1, z1.size(-1))[mask_mixed]
    z2_mixed = z2.view(-1, z2.size(-1))[mask_mixed]

    if z1_mixed.size(0) > 0:
        with torch.no_grad():
            z1_abs_mean = z1_mixed.abs().mean(dim=1)
            z2_abs_mean = z2_mixed.abs().mean(dim=1)
            feature_abs_diff = (z1_mixed - z2_mixed).abs().mean(dim=1)
            normalized_measure = feature_abs_diff / ((z1_abs_mean + z2_abs_mean) / 2)
            sim_mixed = normalized_measure.mean().cpu().numpy()

        avg_similarity_matrix_mixed[i][j] = avg_similarity_matrix_mixed[j][i] = sim_mixed

    # 清理显存
    del z1, mask1, z2, mask2, z1_unpredicted, z2_unpredicted, z1_predicted, z2_predicted, z1_mixed, z2_mixed, mask_unpredicted_common, mask_predicted_common, mask_mixed
    torch.cuda.empty_cache()


def calculate_and_plot_similarity(num_steps, cache_dir, output_dir, k=10, device="cuda:0", max_concurrent_tasks=2):
    """计算相似性并绘制相似性矩阵图"""
    manager = Manager()
    avg_similarity_matrix_predicted = manager.list([manager.list([0] * num_steps) for _ in range(num_steps)])
    avg_similarity_matrix_unpredicted = manager.list([manager.list([0] * num_steps) for _ in range(num_steps)])
    avg_similarity_matrix_mixed = manager.list([manager.list([0] * num_steps) for _ in range(num_steps)])  # 新增矩阵
    set_start_method("spawn", force=True)

    # 使用多进程池并行计算相似度，限制最大并发数量
    with Pool(processes=max_concurrent_tasks) as pool:
        tasks = [
            pool.apply_async(
                calculate_similarity,
                args=(i, j, cache_dir, device, avg_similarity_matrix_predicted, avg_similarity_matrix_unpredicted,
                      avg_similarity_matrix_mixed)
            )
            for i in range(num_steps) for j in range(i + 1, num_steps)
        ]
        for task in tasks:
            task.get()  # 等待所有任务完成

    # 将共享的 list 转为 numpy 数组
    avg_similarity_matrix_predicted = np.array(avg_similarity_matrix_predicted)
    avg_similarity_matrix_unpredicted = np.array(avg_similarity_matrix_unpredicted)
    avg_similarity_matrix_mixed = np.array(avg_similarity_matrix_mixed)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存数据到 Excel 文件
    with pd.ExcelWriter(os.path.join(output_dir, "similarity_matrices.xlsx")) as writer:
        pd.DataFrame(avg_similarity_matrix_unpredicted).to_excel(writer, sheet_name="Unpredicted Similarity")
        pd.DataFrame(avg_similarity_matrix_predicted).to_excel(writer, sheet_name="Predicted Similarity")
        pd.DataFrame(avg_similarity_matrix_mixed).to_excel(writer, sheet_name="Mixed Similarity")  # 新增

    # 绘制相似性矩阵图并保存
    fig, axs = plt.subplots(1, 3, figsize=(24, 8), dpi=1024)  # 增加图片分辨率并新增一个图

    # 动态调整 vmin 和 vmax 为相似度矩阵的最小值和最大值
    vmin_unpredicted = avg_similarity_matrix_unpredicted.min()
    vmax_unpredicted = avg_similarity_matrix_unpredicted.max()
    vmin_predicted = avg_similarity_matrix_predicted.min()
    vmax_predicted = avg_similarity_matrix_predicted.max()
    vmin_mixed = avg_similarity_matrix_mixed.min()
    vmax_mixed = avg_similarity_matrix_mixed.max()

    # 未预测部分相似度矩阵图
    im1 = axs[0].imshow(avg_similarity_matrix_unpredicted, cmap='viridis', vmin=vmin_unpredicted, vmax=vmax_unpredicted)
    axs[0].set_title('Unpredicted Average Similarity')
    fig.colorbar(im1, ax=axs[0])

    # 已预测部分相似度矩阵图
    im2 = axs[1].imshow(avg_similarity_matrix_predicted, cmap='viridis', vmin=vmin_predicted, vmax=vmax_predicted)
    axs[1].set_title('Predicted Average Similarity')
    fig.colorbar(im2, ax=axs[1])

    # 混合部分相似度矩阵图
    im3 = axs[2].imshow(avg_similarity_matrix_mixed, cmap='viridis', vmin=vmin_mixed, vmax=vmax_mixed)
    axs[2].set_title('Mixed Average Similarity')
    fig.colorbar(im3, ax=axs[2])

    plt.savefig(os.path.join(output_dir, 'similarity_matrices.png'), dpi=1024)
    plt.close(fig)  # 关闭图像，释放内存
