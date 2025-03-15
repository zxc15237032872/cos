

# 导入所需的模型类
from lib.model_test import D2Net
from models.matching import Matching

# 初始化 D2Net 模型
checkpoint = torch.load(os.path.join(d2net_dir, 'models/d2_tf.pth'))
d2net = D2Net(model_file=checkpoint, use_relu=True, use_cuda=torch.cuda.is_available())
d2net.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d2net = d2net.to(device)

# 初始化 SuperGlue 匹配器
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matching = Matching(config).eval()
matching = matching.to(device)


def register_images(img_path1, img_path2):
    # 读取图像
    img0_raw = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img1_raw = cv2.imread(img_path2, cv2.IMREAD_COLOR)

    # 调整图像大小
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    # 转换为 PyTorch 张量并归一化
    img0 = torch.from_numpy(img0_raw / 255.).float().permute(2, 0, 1).unsqueeze(0).to(device)
    img1 = torch.from_numpy(img1_raw / 255.).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # 使用 D2Net 提取特征
    with torch.no_grad():
        feats0 = d2net({'image': img0})
        feats1 = d2net({'image': img1})

    # 进行特征匹配
    data = {'image0': img0, 'image1': img1,
            'keypoints0': feats0['keypoints'], 'scores0': feats0['scores'], 'descriptors0': feats0['descriptors'],
            'keypoints1': feats1['keypoints'], 'scores1': feats1['scores'], 'descriptors1': feats1['descriptors']}
    with torch.no_grad():
        pred = matching(data)

    # 获取匹配点
    mkpts0 = pred['matches0'][0].cpu().numpy()
    mkpts1 = pred['matches1'][0].cpu().numpy()
    valid = mkpts0 > -1
    mkpts0 = data['keypoints0'][0][valid].cpu().numpy()
    mkpts1 = data['keypoints1'][0][mkpts0].cpu().numpy()

    # 计算单应性矩阵
    H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)

    # 应用单应性矩阵进行图像配准
    registered_img = cv2.warpPerspective(img1_raw, H, (img0_raw.shape[1], img0_raw.shape[0]))

    return img0_raw, img1_raw, registered_img


def main():
    # 输入图片路径
    img_path1 = 'che001.png'
    img_path2 = 'che002.png'

    # 进行图像配准
    img0_raw, img1_raw, registered_img = register_images(img_path1, img_path2)

    # 绘制三张图
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img0_raw, cv2.COLOR_BGR2RGB))
    plt.title('车 A 原始图')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img1_raw, cv2.COLOR_BGR2RGB))
    plt.title('车 B 原始图')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(registered_img, cv2.COLOR_BGR2RGB))
    plt.title('车 B 配准后图')
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()