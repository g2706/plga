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
# combine_images_horizontally(['pic-our-5/results_1.png', 'pic-our-5/results_2.png', 'pic-our-5/results_24.png', 'pic-our-5/results_35.png', 'pic-our-5/results_39.png'], 'pic-our-5/combined.png')

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

# combine_images_vertically(['visual/combined_first.png','visual/combined_higuchi.png','visual/combined_hc.png','visual/combined_square.png','visual/combined_three.png','visual/combined_weibull.png','visual/combined_kp.png','visual/combined.png'],'visual/all.png')
if __name__ == '__main__':
    combine_images_horizontally(['pic-first-5/results_1_first.png',
                                 'pic-first-5/results_2_first.png',
                                 'pic-first-5/results_24_first.png',
                                 'pic-first-5/results_35_first.png',
                                 'pic-first-5/results_39_first.png',],
                                'combined/first.png')
    combine_images_horizontally(['pic-higuchi-5/results_1_higuchi.png',
                                 'pic-higuchi-5/results_2_higuchi.png',
                                 'pic-higuchi-5/results_24_higuchi.png',
                                 'pic-higuchi-5/results_35_higuchi.png',
                                 'pic-higuchi-5/results_39_higuchi.png',],
                                'combined/higuchi.png')
    combine_images_horizontally(['pic-hc-5/results_1_hc.png',
                                 'pic-hc-5/results_2_hc.png',
                                 'pic-hc-5/results_24_hc.png',
                                 'pic-hc-5/results_35_hc.png',
                                 'pic-hc-5/results_39_hc.png',],
                                'combined/hc.png')
    combine_images_horizontally(['pic-square-5/results_1_square.png',
                                 'pic-square-5/results_2_square.png',
                                 'pic-square-5/results_24_square.png',
                                 'pic-square-5/results_35_square.png',
                                 'pic-square-5/results_39_square.png',],
                                'combined/square.png')
    combine_images_horizontally(['pic-three-5/results_1_three.png',
                                 'pic-three-5/results_2_three.png',
                                 'pic-three-5/results_24_three.png',
                                 'pic-three-5/results_35_three.png',
                                 'pic-three-5/results_39_three.png',],
                                'combined/three.png')
    combine_images_horizontally(['pic-weibull-5/results_1_weibull.png',
                                 'pic-weibull-5/results_2_weibull.png',
                                 'pic-weibull-5/results_24_weibull.png',
                                 'pic-weibull-5/results_35_weibull.png',
                                 'pic-weibull-5/results_39_weibull.png',],
                                'combined/weibull.png')
    combine_images_horizontally(['pic-kp-5/results_1_kp.png',
                                 'pic-kp-5/results_2_kp.png',
                                 'pic-kp-5/results_24_kp.png',
                                 'pic-kp-5/results_35_kp.png',
                                 'pic-kp-5/results_39_kp.png',],
                                'combined/kp.png')
    combine_images_horizontally(['pic-our-5/results_1_our.png',
                                 'pic-our-5/results_2_our.png',
                                 'pic-our-5/results_24_our.png',
                                 'pic-our-5/results_35_our.png',
                                 'pic-our-5/results_39_our.png',],
                                'combined/our.png')

    combine_images_vertically(['combined/first.png',
                               'combined/higuchi.png',
                               'combined/hc.png',
                               'combined/square.png',
                               'combined/three.png',
                               'combined/weibull.png',
                               'combined/kp.png',
                               'combined/our.png',],
                              'combined/ultra.png')

