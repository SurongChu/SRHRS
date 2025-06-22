import os
import random
import csv
import argparse
from PIL import Image
from tqdm import tqdm


def random_crop_and_save(image, filename, output_dir, crop_num):
    """
    从图像中随机裁剪指定数量的块并保存

    参数:
    image: PIL Image对象
    filename: 原始文件名(不带扩展名)
    output_dir: 输出目录路径
    crop_num: 要生成的裁剪块数量

    返回:
    包含裁剪块信息的列表
    """
    w, h = image.size
    min_dim = min(w, h)  # 获取最小边长
    crop_data = []

    for i in range(crop_num):
        # 随机选择裁剪比例 (0.2, 0.4, 0.6, 0.8)
        scale = random.choice([0.2, 0.4, 0.6, 0.8])
        crop_size = int(min_dim * scale)

        # 计算随机裁剪位置 (确保不超出边界)
        max_x = w - crop_size
        max_y = h - crop_size
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0

        # 执行裁剪
        crop = image.crop((x, y, x + crop_size, y + crop_size))

        # 将裁剪块缩放到512×512
        resized_crop = crop.resize((512, 512), Image.LANCZOS)

        # 生成输出文件名
        crop_filename = f"{filename}_scale{scale:.1f}_crop{i}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)

        # 保存缩放后的图像
        resized_crop.save(crop_path, "JPEG")

        # 计算相对位置 (0-1之间)
        rel_x = round(x / w, 1) if w > 0 else 0.0
        rel_y = round(y / h, 1) if h > 0 else 0.0

        # 记录元数据
        crop_data.append({
            'filename': crop_filename,
            'original_image': f"{filename}.jpg",
            'scale': scale,
            'crop_size': crop_size,  # 原始裁剪尺寸
            'original_width': w,  # 原始图像宽度
            'original_height': h,  # 原始图像高度
            'rel_x': rel_x,  # 相对X位置 (0-1)
            'rel_y': rel_y,  # 相对Y位置 (0-1)
        })

    return crop_data


def process_dataset(source_dir, output_dir, output_csv):
    """
    处理整个数据集

    参数:
    source_dir: 原始图像目录
    output_dir: 处理后的图像输出目录
    output_csv: CSV元数据文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有JPEG图像文件
    image_files = [f for f in os.listdir(source_dir)
                   if f.lower().endswith(('.jpg', '.jpeg'))]

    if not image_files:
        print(f"在 {source_dir} 中没有找到图像文件")
        return

    # 创建CSV文件并写入表头
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'filename', 'original_image', 'scale',
            'crop_size', 'original_width','original_height','rel_x', 'rel_y'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 处理每张图像
        for img_file in tqdm(image_files, desc="处理图像"):
            try:
                # 打开图像
                img_path = os.path.join(source_dir, img_file)
                image = Image.open(img_path)

                # 获取不带扩展名的文件名
                filename_base = os.path.splitext(img_file)[0]

                # 生成裁剪块
                crop_data = random_crop_and_save(
                    image, filename_base, output_dir, crop_num=5
                )

                # 写入CSV
                for data in crop_data:
                    writer.writerow(data)

            except Exception as e:
                print(f"处理 {img_file} 时出错: {str(e)}")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='图像数据集预处理')
    parser.add_argument('--source', type=str, default='sourcedata',
                        help='原始图像目录路径')
    parser.add_argument('--output', type=str, default='processeddata',
                        help='处理后的图像输出目录路径')
    parser.add_argument('--csv', type=str, default='crop_metadata.csv',
                        help='元数据CSV文件路径')

    args = parser.parse_args()

    print(f"开始处理数据集...")
    print(f"源目录: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"CSV文件: {args.csv}")

    # 执行处理
    process_dataset(args.source, args.output, args.csv)

    print("处理完成！")