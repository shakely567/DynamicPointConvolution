import open3d as o3d
import numpy as np
import trimesh
import importlib
import torch
import json
import os
import sys
import argparse
from sklearn.decomposition import PCA
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from models import pointnet2_sem_seg
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from glob import glob


import matplotlib.pyplot as plt

# 先定义一个简单的颜色映射函数
def create_color_map(labels, num_classes):
    colors = plt.get_cmap("tab10")(labels / max(labels))
    return colors[:, :3]  # 忽略alpha通道并规范化到0-1范围

def update_point_cloud_colors(pcd, seg_pred_labels, colors):
    # 根据标签创建颜色映射
    colors[seg_pred_labels == 1] = [0.75, 0, 0]
    colors[seg_pred_labels == 3] = [0.25, 0, 0]
    colors[seg_pred_labels == 2] = [0.5, 0, 0]
    colors[seg_pred_labels == 4] = [0, 1, 0]
    colors[seg_pred_labels == 5] = [0, 0.75, 0]
    colors[seg_pred_labels == 6] = [0, 0.5, 0]
    colors[seg_pred_labels == 7] = [0, 0.25, 0]
    colors[seg_pred_labels == 8] = [0, 0, 1]
    colors[seg_pred_labels == 9] = [0, 0, 0.75]
    colors[seg_pred_labels == 10] = [0, 0, 0.5]
    colors[seg_pred_labels == 11] = [0, 0, 0.25]
    colors[seg_pred_labels == 12] = [1, 0, 1]
    colors[seg_pred_labels == 13] = [0, 1, 1]
    colors[seg_pred_labels == 14] = [1, 1, 0]
    colors[seg_pred_labels == 15] = [0.5, 1, 1]
    colors[seg_pred_labels == 16] = [0, 0.5, 1]

    # 更新点云的颜色
    pcd.vertex_colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_as_ply(filename, points, colors):
    # 确保点和颜色的数量相同
    assert points.shape[0] == colors.shape[0], "The number of points must be the same as the number of colors"

    # 打开文件以写入
    with open(filename, 'w') as ply_file:
        # 写入 PLY 文件头
        ply_file.write('ply\n')
        ply_file.write('format ascii 1.0\n')
        ply_file.write('element vertex {}\n'.format(len(points)))
        ply_file.write('property float x\n')
        ply_file.write('property float y\n')
        ply_file.write('property float z\n')
        ply_file.write('property uchar red\n')
        ply_file.write('property uchar green\n')
        ply_file.write('property uchar blue\n')
        ply_file.write('end_header\n')

        # 写入点云数据
        for i in range(len(points)):
            color = colors[i] * 255  # 假设颜色值在 0-1 范围
            ply_file.write('{} {} {} {} {} {}\n'.format(points[i][0], points[i][1], points[i][2],
                                                        int(color[0]), int(color[1]), int(color[2])))


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc
'''标准化后的点云在原点附近，且整体尺度被缩放到单位球范围内'''

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def random_subsamples(points, labels):
    num_points = 50000
    if len(points) >= num_points:
        sampled_indices = random.sample(range(len(points)), num_points)
        points = [points[i] for i in sampled_indices]
        labels = [labels[i] for i in sampled_indices]
    return points, labels

def random_subsample_ind(points):
    num_points = len(points)
    if len(points) >= num_points:
        sampled_indices = random.sample(range(len(points)), num_points)
        points = [points[i] for i in sampled_indices]
    return points, sampled_indices

def random_subsample(points):
    num_points = len(points)
    if len(points) >= num_points:
        sampled_indices = random.sample(range(len(points)), num_points)
        points = [points[i] for i in sampled_indices]
    return points

def visualize_pointcloud_with_labels(points, labels):
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 添加每个点的标签作为颜色
    max_label = np.max(labels)
    colors = np.zeros((len(labels), 3))
    colors[:, 0] = labels / max_label  # 将标签映射到 [0, 1] 之间
    print(colors.shape)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

def concatenate_arrays(ply):
    point = np.empty((0,))
    color = np.empty((0,))
    lable = []
    for i in ply:
        pcd = o3d.io.read_point_cloud(i)
        point_cloud = np.asarray(pcd.points)
        point_color = np.asarray(pcd.colors)

        if point.size == 0:
            point = point_cloud
            color = point_color
        else:
            point = np.vstack((point, point_cloud))
            color = np.vstack((color, point_color))

        file_name = os.path.basename(i)
        if 'teeth00.ply' in file_name:
            lab = [0] * point_cloud.shape[0]
        elif 'teeth01.ply' in file_name:
            lab = [1] * point_cloud.shape[0]
        elif 'teeth02.ply' in file_name:
            lab = [2] * point_cloud.shape[0]
        elif 'teeth03.ply' in file_name:
            lab = [3] * point_cloud.shape[0]
        elif 'teeth04.ply' in file_name:
            lab = [4] * point_cloud.shape[0]
        elif 'teeth05.ply' in file_name:
            lab = [5] * point_cloud.shape[0]
        elif 'teeth06.ply' in file_name:
            lab = [6] * point_cloud.shape[0]
        elif 'teeth07.ply' in file_name:
            lab = [7] * point_cloud.shape[0]
        elif 'teeth08.ply' in file_name:
            lab = [8] * point_cloud.shape[0]
        elif 'teeth09.ply' in file_name:
            lab = [9] * point_cloud.shape[0]
        elif 'teeth10.ply' in file_name:
            lab = [10] * point_cloud.shape[0]
        elif 'teeth11.ply' in file_name:
            lab = [11] * point_cloud.shape[0]
        elif 'teeth12.ply' in file_name:
            lab = [12] * point_cloud.shape[0]
        elif 'teeth13.ply' in file_name:
            lab = [13] * point_cloud.shape[0]
        elif 'teeth14.ply' in file_name:
            lab = [14] * point_cloud.shape[0]
        elif 'teeth15.ply' in file_name:
            lab = [15] * point_cloud.shape[0]
        elif 'teeth16.ply' in file_name:
            lab = [16] * point_cloud.shape[0]
        lable.extend(lab)
    lable = np.array(lable)
    point = pc_normalize(point)
    pca = PCA(n_components=3)  # PCA
    pca.fit(point)
    point = pca.transform(point)
    point[:, 2] = -point[:, 2]
    points = np.hstack((point, color))

    return points, lable
def demo(args):
    '''MODEL LOADING'''
    NUM_CLASSES = 17
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load('F:/point/pointnet_step_wcnl/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        '''读取obj文件'''
        # obj_file = 'D:/pointcloud/Teeth3DS/data_part_4/upper/01FUZ18R/01FUZ18R_upper.obj'
        # obj_file = 'D:/pointcloud/Teeth3DS/test/data_part_1/upper/A9TECAGP/A9TECAGP_upper.obj'
        # mesh = trimesh.load_mesh(obj_file)
        # point_cloud = mesh.vertices
        # point_cloud = random_subsample(point_cloud)
        # point_cloud_obj = pc_normalize(point_cloud)
        #
        # point_colors = mesh.visual.vertex_colors
        # point_colors = point_colors[:, 0:3] / 255
        # point = np.hstack((point_cloud_obj, point_colors))

        # json_file = 'D:/pointcloud/Teeth3DS/data_part_4/upper/01FUZ18R/01FUZ18R_upper.json'
        # json_file = 'D:/pointcloud/Teeth3DS/testing_lower/data_part_6/lower/0VYQUKGQ/0VYQUKGQ_lower.json'
        # with open(json_file, 'r') as f:
        #     data = json.load(f)
        # labels = data.get("labels", [])
        # label = []
        # point = []
        # for i, j in zip(labels, point_cloud):
        #     if i != 0:
        #         label.append(i)
        #         point.append(j)
        # point = np.array(point)
        # point = pc_normalize(point)
        # point = random_subsample(point)

        '''读取txt'''
        # file_path = 'D:/pointcloud/Area_1/hallway_4/hallway_4.txt'  # 替换为你的文件路径
        # with open(file_path, 'r') as file:
        #     lines = file.readlines()
        # # 将文本数据转换为数组
        # point_cloud = np.array([list(map(float, line.split())) for line in lines])

        '''读取ply文件'''
        ply_path = "F:/point/06 LowerJawScan.ply"
        pcd = o3d.io.read_triangle_mesh(ply_path)
        point_cloud = np.asarray(pcd.vertices)
        point_color = np.asarray(pcd.vertex_colors)
        # pcd = o3d.io.read_point_cloud(ply_path)
        # point_cloud = np.asarray(pcd.points)
        # point_color = np.asarray(pcd.colors)
        point_cloud = pc_normalize(point_cloud)
        pca = PCA(n_components=3)                                # PCA
        pca.fit(point_cloud)
        point_cloud = pca.transform(point_cloud)
        point_cloud[:, 2] = -point_cloud[:, 2]
        point = np.hstack((point_cloud, point_color))
        point, ind = random_subsample_ind(point)               # 改变了点序
        point = np.array(point)
        point_xyz = point[:, 0:3]
        colors = point[:, 3:6]
        # point_cloud_ply = pc_normalize(point)

        '''toothlabel合并'''
        # root_path = "C:/Users/Administrator/Desktop/toothlabel/test/1_tartar/wuhongcheng_"
        # ply = sorted(glob(os.path.join(root_path, "*.ply")))
        # point_cloud, lable = concatenate_arrays(ply)
        #
        # point_cloud, lable = random_subsamples(point_cloud, lable)
        # point = np.asarray(point_cloud)
        # lable = np.asarray(lable)
        # point_xyz = point[:, 0:3]
        # colors = point[:, 3:6]

        # point = np.hstack((point_cloud_ply, point_color))
        # pcd_vector = o3d.geometry.PointCloud()
        # pcd_vector.points = o3d.utility.Vector3dVector(points[:, 0:3])
        # pcd_vector.colors = o3d.utility.Vector3dVector(points[:, 3:6])
        # o3d.visualization.draw_geometries([pcd_vector])
        # # exit()
        '''分割点云'''
        point_dim = np.expand_dims(point, axis=0)
        points = torch.Tensor(point_dim)
        points = points.float().cuda()
        points = points.transpose(2, 1)
             # torch.Size([1, 3, 92781])
        seg_pred,_ = classifier(points)   # , trans_feat
        # torch.Size([1, 92781, 2])
        seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
           # torch.Size([92781, 2/num_class])
        seg_prob = F.softmax(seg_pred, dim=-1)
        seg_pred_labels = torch.argmax(seg_prob, dim=-1)
        seg_pred_labels = seg_pred_labels.cpu().numpy()

        full_point_labels = -np.ones(len(point), dtype=int)
        for idx, label in zip(ind, seg_pred_labels):
            full_point_labels[idx] = label

        '''添加每个点的标签作为颜色'''
        # # colors = np.ones((len(seg_pred_labels), 3)) * 0.5
        # # colors[seg_pred_labels == 0] = [1, 0, 0]
        # colors[seg_pred_labels == 1] = [0.75, 0, 0]
        # colors[seg_pred_labels == 3] = [0.25, 0, 0]
        # colors[seg_pred_labels == 2] = [0.5, 0, 0]
        # colors[seg_pred_labels == 4] = [0, 1, 0]
        # colors[seg_pred_labels == 5] = [0, 0.75, 0]
        # colors[seg_pred_labels == 6] = [0, 0.5, 0]
        # colors[seg_pred_labels == 7] = [0, 0.25, 0]
        # colors[seg_pred_labels == 8] = [0, 0, 1]
        # colors[seg_pred_labels == 9] = [0, 0, 0.75]
        # colors[seg_pred_labels == 10] = [0, 0, 0.5]
        # colors[seg_pred_labels == 11] = [0, 0, 0.25]
        # colors[seg_pred_labels == 12] = [1, 0, 1]
        # colors[seg_pred_labels == 13] = [0, 1, 1]
        # colors[seg_pred_labels == 14] = [1, 1, 0]
        # colors[seg_pred_labels == 15] = [0.5, 1, 1]
        # colors[seg_pred_labels == 16] = [0, 0.5, 1]
        # # print(colors[0:50])
        #
        # colors[lable == 1] = [0.75, 0, 0]
        # colors[lable == 3] = [0.25, 0, 0]
        # colors[lable == 2] = [0.5, 0, 0]
        # colors[lable == 4] = [0, 1, 0]
        # colors[lable == 5] = [0, 0.75, 0]
        # colors[lable == 6] = [0, 0.5, 0]
        # colors[lable == 7] = [0, 0.25, 0]
        # colors[lable == 8] = [0, 0, 1]
        # colors[lable == 9] = [0, 0, 0.75]
        # colors[lable == 10] = [0, 0, 0.5]
        # colors[lable == 11] = [0, 0, 0.25]
        # colors[lable == 12] = [1, 0, 1]
        # colors[lable == 13] = [0, 1, 1]
        # colors[lable == 14] = [1, 1, 0]
        # colors[lable == 15] = [0.5, 1, 1]
        # colors[lable == 16] = [0, 0.5, 1]

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_xyz)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # # 可视化点云
        # o3d.visualization.draw_geometries([pcd])

        seg_pcd = update_point_cloud_colors(pcd, full_point_labels, colors)
        # 最后，保存更新颜色后的点云为新的PLY文件
        o3d.io.write_triangle_mesh("segmented_mesh_path.ply", seg_pcd)

        # save_as_ply("output.ply", point_xyz, colors)      # 点云ply

        '''同时显示2个点云'''
        # cloud1 = o3d.geometry.PointCloud()
        # cloud1.points = o3d.utility.Vector3dVector(point_cloud)
        # cloud2 = o3d.geometry.PointCloud()
        # cloud2.points = o3d.utility.Vector3dVector(point_cloud2)
        # # 将两个点云加入到同一个窗口中显示
        # o3d.visualization.draw_geometries([cloud2])

        '''PCA'''
        # 使用 PCA 进行坐标中心化和平面对齐
        # pca = PCA(n_components=3)
        # # data_centered = point_cloud_ply - np.mean(point_cloud_ply, axis=0)  # 中心化
        # pca.fit(point_cloud_ply)
        # transformed_data = pca.transform(point_cloud_ply)
        # # 显示原始点云
        # cloud_original = o3d.geometry.PointCloud()
        # cloud_original.points = o3d.utility.Vector3dVector(point_cloud_ply)
        # # 显示经过 PCA 处理后的点云
        # cloud_transformed = o3d.geometry.PointCloud()
        # cloud_transformed.points = o3d.utility.Vector3dVector(transformed_data)
        # # 将两个点云加入到同一个窗口中显示
        # o3d.visualization.draw_geometries([cloud_original])


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg]')
    return parser.parse_args()

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    args = parse_args()

    demo(args)


                             # paconv_sem_seg2 , dgcnn_sem_seg , pointnet2_sem_seg_msg





