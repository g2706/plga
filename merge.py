from PIL import Image

def combine_images_horizontally(image_paths, output_path):
    # 打开所有图片并获取它们的宽度和高度
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(img.size for img in images))

    # 创建一个新的空白图像，宽度是所有图像宽度之和，高度取最大值
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    # 将每个图像粘贴到新图像中
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # 保存新图像
    new_image.save(output_path)
    print(f"Combined image saved to {output_path}")

# 使用函数
# combine_images_horizontally(['compare/results_1_kp.png', 'compare/results_24_kp.png', 'compare/results_35_kp.png', 'compare/results_39_kp.png'], 'compare/combined_horizontal_kp.png')


def combine_images_vertically(image_paths, output_path):
    # 打开所有图片并获取它们的宽度和高度
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(img.size for img in images))

    # 创建一个新的空白图像，宽度取最大值，高度是所有图像高度之和
    max_width = max(widths)
    total_height = sum(heights)

    new_image = Image.new('RGB', (max_width, total_height))

    # 将每个图像粘贴到新图像中
    y_offset = 0
    for img in images:
        new_image.paste(img, (0, y_offset))
        y_offset += img.height

    # 保存新图像
    new_image.save(output_path)
    print(f"Combined image saved to {output_path}")

# 使用函数
combine_images_vertically(['compare/combined_horizontal_kp.png', 'compare/combined_horizontal_weibull.png', 'compare/combined_horizontal_ours.png'], 'compare/combined_vertical.png')