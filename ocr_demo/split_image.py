import os
import sys
import glob
import numpy as np
from PIL import Image

def is_separator_row(row_data, threshold=10, black_ratio=0.995):
    """
    判断一行像素是否为分隔行（几乎全是纯黑色）
    
    参数:
    row_data: 图片的一行像素数据
    threshold: 黑色的阈值（0-255），小于此值被视为黑色
    black_ratio: 判定为分隔行所需的黑色像素比例
    """
    # 将RGB转为灰度值
    if len(row_data.shape) == 2:  # 已经是灰度图
        gray_row = row_data
    else:  # RGB图像
        gray_row = np.mean(row_data, axis=1)
    
    # 计算黑色像素比例
    black_pixels = np.sum(gray_row < threshold)
    return black_pixels / len(gray_row) >= black_ratio

def find_separator_regions(img_array, threshold=10, black_ratio=0.995, min_height=3, max_height=60):
    """
    查找图片中的黑色分隔区域（纯黑色行）
    
    参数:
    img_array: 图片数组
    threshold: 黑色的阈值
    black_ratio: 判定为分隔行所需的黑色像素比例
    min_height: 分隔区域的最小高度（行数）
    max_height: 分隔区域的最大高度（行数），如果为None则不限制
    
    返回:
    分隔点的列表 [y1, y2, ...]
    """
    height = img_array.shape[0]
    separators = []
    
    # 寻找连续的黑色区域
    in_black_region = False
    black_start = 0
    
    for y in range(height):
        if is_separator_row(img_array[y], threshold, black_ratio):
            if not in_black_region:
                in_black_region = True
                black_start = y
        else:
            if in_black_region:
                in_black_region = False
                black_region_height = y - black_start
                # 检查黑色区域高度是否符合要求
                if black_region_height >= min_height and (max_height is None or black_region_height <= max_height):
                    # 找到一个分隔区域，记录其中点
                    separator_point = (black_start + y) // 2
                    separators.append(separator_point)
                    print(f"找到分隔线：位置 {separator_point}，高度 {black_region_height} 像素，黑度 {black_ratio*100:.1f}%")
    
    # 处理图片底部的黑色区域
    if in_black_region:
        black_region_height = height - black_start
        if black_region_height >= min_height and (max_height is None or black_region_height <= max_height):
            separator_point = (black_start + height) // 2
            separators.append(separator_point)
            print(f"找到分隔线：位置 {separator_point}，高度 {black_region_height} 像素，黑度 {black_ratio*100:.1f}%")
    
    return separators

def debug_image_rows(image_path, output_dir, sample_step=100):
    """
    分析图片每行的黑色像素比例，帮助调试分隔线检测
    
    参数:
    image_path: 输入图片路径
    output_dir: 输出目录
    sample_step: 采样步长，每隔多少行采样一次
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"分析图片: {image_path}, 尺寸: {width}x{height}")
            
            # 转换为numpy数组
            img_array = np.array(img)
            
            # 创建输出文件
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_analysis.txt")
            
            with open(output_path, 'w') as f:
                f.write(f"图片行分析: {image_path}\n")
                f.write(f"行号\t黑色像素比例(阈值5)\t黑色像素比例(阈值10)\t黑色像素比例(阈值15)\n")
                
                # 分析每隔sample_step行的黑色像素比例
                for y in range(0, height, sample_step):
                    if y >= height:
                        break
                    
                    row = img_array[y]
                    # 计算不同阈值下的黑色像素比例
                    if len(row.shape) == 2:  # 灰度图
                        gray_row = row
                    else:  # RGB图像
                        gray_row = np.mean(row, axis=1)
                    
                    ratio5 = np.sum(gray_row < 5) / len(gray_row)
                    ratio10 = np.sum(gray_row < 10) / len(gray_row)
                    ratio15 = np.sum(gray_row < 15) / len(gray_row)
                    
                    f.write(f"{y}\t{ratio5:.4f}\t{ratio10:.4f}\t{ratio15:.4f}\n")
                    
                    # 如果发现高黑色比例，额外记录周围行
                    if ratio10 > 0.7:
                        print(f"行 {y} 黑色比例较高: {ratio10:.4f}")
                        # 分析周围20行
                        for dy in range(-10, 11):
                            check_y = y + dy
                            if 0 <= check_y < height:
                                check_row = img_array[check_y]
                                if len(check_row.shape) == 2:
                                    check_gray = check_row
                                else:
                                    check_gray = np.mean(check_row, axis=1)
                                
                                check_ratio = np.sum(check_gray < 10) / len(check_gray)
                                f.write(f"{check_y}\t{check_ratio:.4f}\t-\t-\n")
            
            print(f"分析结果已保存到: {output_path}")
            return output_path
    
    except Exception as e:
        print(f"分析图片时出错: {str(e)}")
        return None

def split_image_by_color(image_path, output_dir, black_threshold=5, black_ratio=0.995, min_separator_height=3, max_separator_height=60):
    """
    根据纯黑色分隔行将长图分割成多张小图
    
    参数:
    image_path: 输入图片路径
    output_dir: 输出目录
    black_threshold: 黑色的阈值（0-255），小于此值被视为黑色
    black_ratio: 判定为分隔行所需的黑色像素比例
    min_separator_height: 分隔线的最小高度（像素）
    max_separator_height: 分隔线的最大高度（像素），如果为None则不限制
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开原始图片
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"处理: {image_path}")
            print(f"原图尺寸: {width}x{height} 像素")
            print(f"分隔线配置: 黑色阈值={black_threshold}, 黑色比例={black_ratio*100:.1f}%, 最小高度={min_separator_height}, 最大高度={max_separator_height or '不限'}")
            
            # 转换为numpy数组处理
            img_array = np.array(img)
            
            # 查找分隔点
            separators = find_separator_regions(img_array, black_threshold, black_ratio, min_separator_height, max_separator_height)
            separators = [0] + separators + [height]
            
            print(f"检测到 {len(separators)-1} 个区域，分隔点位置: {separators}")
            
            # 分割并保存
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            
            for i in range(len(separators) - 1):
                top = separators[i]
                bottom = separators[i + 1]
                
                # 确保区域高度大于0
                if bottom - top <= 0:
                    continue
                    
                # 裁剪图片
                chunk = img.crop((0, top, width, bottom))
                
                # 保存小图
                output_path = os.path.join(output_dir, f"{base_name}_part_{i+1}.jpg")
                chunk.save(output_path, quality=95)
                print(f"已保存: {output_path} ({chunk.width}x{chunk.height} 像素)")
                sys.stdout.flush()
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")

def split_image_by_height(image_path, output_dir, chunk_height=1000):
    """
    按固定高度将长图分割成多张小图
    
    参数:
    image_path: 输入图片路径
    output_dir: 输出目录
    chunk_height: 每张小图的高度(像素)
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开原始图片
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"处理: {image_path}")
            print(f"原图尺寸: {width}x{height} 像素")
            
            # 计算需要分割的数量
            num_chunks = (height + chunk_height - 1) // chunk_height
            print(f"将分割成 {num_chunks} 张小图")
            
            # 分割并保存
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            
            for i in range(num_chunks):
                top = i * chunk_height
                bottom = min((i + 1) * chunk_height, height)
                
                # 裁剪图片
                chunk = img.crop((0, top, width, bottom))
                
                # 保存小图
                output_path = os.path.join(output_dir, f"{base_name}_part_{i+1}.jpg")
                chunk.save(output_path, quality=95)
                print(f"已保存: {output_path} ({chunk.width}x{chunk.height} 像素)")
                sys.stdout.flush()
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")

def split_image(image_path, output_dir, chunk_height=1000, by_color=False, black_threshold=5, 
                black_ratio=0.995, min_separator_height=3, max_separator_height=60, debug=True):
    """
    将长图从上往下分割成多张小图
    
    参数:
    image_path: 输入图片路径
    output_dir: 输出目录
    chunk_height: 每张小图的高度(像素)
    by_color: 是否按颜色分隔区域分割
    black_threshold: 黑色的阈值（0-255），小于此值被视为黑色
    black_ratio: 判定为分隔行所需的黑色像素比例
    min_separator_height: 分隔线的最小高度（像素）
    max_separator_height: 分隔线的最大高度（像素），如果为None则不限制
    debug: 是否进行调试分析
    """
    # 如果开启调试，先分析图片
    if debug:
        debug_image_rows(image_path, output_dir)
    
    if by_color:
        split_image_by_color(image_path, output_dir, black_threshold, black_ratio, 
                             min_separator_height, max_separator_height)
    else:
        split_image_by_height(image_path, output_dir, chunk_height)

def process_batch(image_paths, output_dir, chunk_height=1000, by_color=False, black_threshold=5, 
                  black_ratio=0.995, min_separator_height=3, max_separator_height=60):
    """
    批量处理多个图片
    
    参数:
    image_paths: 图片路径列表或通配符
    output_dir: 输出目录
    chunk_height: 每张小图的高度(像素)
    by_color: 是否按颜色分隔区域分割
    black_threshold: 黑色的阈值（0-255），小于此值被视为黑色
    black_ratio: 判定为分隔行所需的黑色像素比例
    min_separator_height: 分隔线的最小高度（像素）
    max_separator_height: 分隔线的最大高度（像素），如果为None则不限制
    """
    # 如果输入是通配符，展开为文件列表
    if isinstance(image_paths, str) and ('*' in image_paths or '?' in image_paths):
        files = glob.glob(image_paths)
    else:
        files = image_paths if isinstance(image_paths, list) else [image_paths]
    
    # 文件类型过滤
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in files if os.path.isfile(f) and 
                   os.path.splitext(f.lower())[1] in valid_extensions]
    
    if not image_files:
        print(f"没有找到匹配的图片文件: {image_paths}")
        return
    
    print(f"找到 {len(image_files)} 个图片文件待处理")
    
    # 处理每个图片
    for image_path in image_files:
        # 为每个图片创建单独的输出目录
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        img_output_dir = os.path.join(output_dir, base_name)
        
        split_image(image_path, img_output_dir, chunk_height, by_color, black_threshold, 
                   black_ratio, min_separator_height, max_separator_height)
        print(f"完成处理: {image_path}")
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("单张图片: python split_image.py <图片路径> [输出目录] [每张小图高度/auto] [color] [黑色阈值] [黑色比例] [最小分隔线高度] [最大分隔线高度]")
        print("批量处理: python split_image.py --batch <图片路径通配符> [输出目录] [每张小图高度/auto] [color] [黑色阈值] [黑色比例] [最小分隔线高度] [最大分隔线高度]")
        print("\n参数说明:")
        print("  黑色阈值: 0-255之间的数字，小于此值的像素被视为黑色，默认5")
        print("  黑色比例: 0-1之间的小数，一行中黑色像素比例超过此值才视为分隔线，默认0.995")
        print("  最小分隔线高度: 连续黑色行的最小数量，少于此值不视为分隔线，默认3像素")
        print("  最大分隔线高度: 连续黑色行的最大数量，超过此值不视为分隔线，默认60像素")
        print("\n示例:")
        print("单张图片按固定高度分割: python split_image.py long_image.jpg output_chunks 1000")
        print("单张图片按颜色分割: python split_image.py long_image.jpg output_chunks auto color")
        print("指定分隔线高度为10-50像素: python split_image.py long_image.jpg output_chunks auto color 5 0.995 10 50")
        print("批量按固定高度分割: python split_image.py --batch \"*.jpg\" output_all 1000")
        print("批量按颜色分割: python split_image.py --batch \"datasets/*.jpg\" output_all auto color")
        sys.exit(1)
    
    # 检查是否为批量模式
    batch_mode = sys.argv[1] == '--batch'
    
    if batch_mode:
        if len(sys.argv) < 3:
            print("批量模式需要指定图片路径通配符")
            sys.exit(1)
        
        image_path_pattern = sys.argv[2]
        
        # 参数偏移量 (因为有--batch参数)
        offset = 1
    else:
        image_path = sys.argv[1]
        offset = 0
    
    # 默认输出目录为"split_output"
    output_dir = sys.argv[2 + offset] if len(sys.argv) > 2 + offset else "split_output"
    
    # 检查是否按颜色分割
    by_color = False
    arg_index = 3 + offset
    has_color = False
    
    if len(sys.argv) > arg_index and (sys.argv[arg_index].lower() == "color" or 
                                     (len(sys.argv) > arg_index + 1 and sys.argv[arg_index + 1].lower() == "color")):
        by_color = True
        if sys.argv[arg_index].lower() == "color":
            has_color = True
            arg_index += 1
        else:
            next_index = arg_index + 1
            if len(sys.argv) > next_index and sys.argv[next_index].lower() == "color":
                has_color = True
                arg_index += 2
            else:
                arg_index += 1
        
        chunk_height = 0  # 不使用固定高度
        print("将按照黑色分隔区域分割图片")
    else:
        # 默认每张小图高度为1000像素
        chunk_height = int(sys.argv[arg_index]) if len(sys.argv) > arg_index and sys.argv[arg_index].lower() != "auto" else 1000
        arg_index += 1
    
    # 解析颜色分割的参数
    black_threshold = 5  # 默认黑色阈值
    black_ratio = 0.995  # 默认黑色比例
    min_separator_height = 3  # 默认最小分隔线高度
    max_separator_height = 60  # 默认最大分隔线高度

    if by_color:
        # 黑色阈值
        if len(sys.argv) > arg_index:
            try:
                black_threshold = int(sys.argv[arg_index])
                arg_index += 1
            except ValueError:
                pass  # 不是数字，忽略
        
        # 黑色比例
        if len(sys.argv) > arg_index:
            try:
                black_ratio = float(sys.argv[arg_index])
                black_ratio = max(0, min(1, black_ratio))  # 确保在0-1之间
                arg_index += 1
            except ValueError:
                pass  # 不是数字，忽略
        
        # 最小分隔线高度
        if len(sys.argv) > arg_index:
            try:
                min_separator_height = int(sys.argv[arg_index])
                arg_index += 1
            except ValueError:
                pass  # 不是数字，忽略
        
        # 最大分隔线高度
        if len(sys.argv) > arg_index:
            try:
                max_separator_height = int(sys.argv[arg_index])
            except ValueError:
                max_separator_height = 60  # 默认值
        else:
            max_separator_height = 60  # 确保有默认值
    
    # 执行分割
    if batch_mode:
        process_batch(image_path_pattern, output_dir, chunk_height, by_color, black_threshold, 
                     black_ratio, min_separator_height, max_separator_height)
    else:
        split_image(image_path, output_dir, chunk_height, by_color, black_threshold, 
                   black_ratio, min_separator_height, max_separator_height) 
