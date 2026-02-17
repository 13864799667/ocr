import os
import sys
import glob
from ocr_to_excel import load_coordinate_file, process_images, load_coordinate_data

def process_batch(base_dir, coordinates, output_file, save_roi=True, roi_dir="roi_images", recursive=False):
    """
    批量处理图片目录中的图片文件，执行OCR识别并保存到Excel
    
    参数:
    base_dir: 包含图片的目录或父目录
    coordinates: 坐标字典
    output_file: 输出Excel文件路径
    save_roi: 是否保存ROI图像
    roi_dir: ROI图像保存目录
    recursive: 是否递归处理所有子目录
    """
    # 如果是递归模式，处理所有子目录
    if recursive:
        # 获取所有子目录
        all_dirs = [base_dir]
        for root, dirs, files in os.walk(base_dir):
            for dir_name in dirs:
                all_dirs.append(os.path.join(root, dir_name))
        
        print(f"找到 {len(all_dirs)} 个目录待处理")
        processed_count = 0
        
        # 处理每个子目录
        for dir_path in all_dirs:
            # 检查是否有图片文件
            image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                print(f"处理目录: {dir_path}")
                process_images(dir_path, coordinates, output_file, save_roi=save_roi, roi_dir=roi_dir)
                processed_count += 1
                print("-" * 40)
        
        print(f"共处理了 {processed_count} 个包含图片的目录")
        return
    
    # 以下是非递归模式的原有逻辑
    # 检查base_dir是否直接包含图片文件
    image_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:
        # 目录直接包含图片，直接处理
        print(f"在 {base_dir} 中找到 {len(image_files)} 个图片文件")
        process_images(base_dir, coordinates, output_file, save_roi=save_roi, roi_dir=roi_dir)
        return
    
    # 如果目录不直接包含图片，检查子目录
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print(f"错误: 在 {base_dir} 及其子目录中没有找到图片文件")
        return
    
    print(f"找到 {len(subdirs)} 个目录待处理")
    
    # 处理每个子目录
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        print(f"处理目录: {subdir_path}")
        process_images(subdir_path, coordinates, output_file, save_roi=save_roi, roi_dir=roi_dir)
        print("-" * 40)

def main():
    if len(sys.argv) < 2:
        print("用法: python batch_ocr_to_excel.py <图片目录> [坐标文件] [输出Excel文件] [--recursive] [--save-roi] [--roi-dir <ROI保存目录>]")
        print("示例: python batch_ocr_to_excel.py min_height_test")
        print("示例: python batch_ocr_to_excel.py min_height_test results.xlsx")
        print("示例: python batch_ocr_to_excel.py min_height_test 参考坐标区域.txt results.xlsx")
        print("示例: python batch_ocr_to_excel.py all_color_output results.xlsx --recursive --save-roi --roi-dir batch_roi")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    # 解析命令行参数
    save_roi = "--save-roi" in sys.argv
    recursive = "--recursive" in sys.argv
    roi_dir = "roi_images"  # 默认ROI目录
    
    if "--roi-dir" in sys.argv:
        roi_dir_index = sys.argv.index("--roi-dir")
        if roi_dir_index + 1 < len(sys.argv):
            roi_dir = sys.argv[roi_dir_index + 1]
    
    # 检查是否提供了坐标文件
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--') and os.path.exists(sys.argv[2]):
        # 使用外部坐标文件
        coord_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else "results.xlsx"
        coordinates = load_coordinate_file(coord_file)
    else:
        # 使用内置坐标数据
        output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else "results.xlsx"
        coordinates = load_coordinate_data()
    
    # 检查目录是否存在
    if not os.path.isdir(base_dir):
        print(f"错误: 目录 {base_dir} 不存在")
        sys.exit(1)
    
    # 批量处理图片并保存结果
    process_batch(base_dir, coordinates, output_file, save_roi, roi_dir, recursive)

if __name__ == "__main__":
    main() 