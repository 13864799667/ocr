#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR识别与数据校验集成脚本

整合OCR识别和数据校验功能，提供完整的处理流程：
1. OCR识别提取数据
2. 自动数据校验
3. 生成校验报告
4. 保存有问题的数据
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import datetime
import logging
from data_validator import DataValidator

def setup_logging(debug=False):
    """设置日志配置"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ocr_validation.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def run_ocr_process(input_dir, output_dir, excel_file, roi_expand, debug, no_split):
    """
    运行OCR识别过程
    
    参数:
    input_dir: 输入图片目录
    output_dir: 输出目录
    excel_file: Excel输出文件
    roi_expand: ROI扩展像素
    debug: 调试模式
    no_split: 跳过分割
    
    返回:
    (success, message)
    """
    logging.info("开始OCR识别过程...")
    
    # 构建OCR命令
    cmd = f'python one_key_process.py "{input_dir}"'
    cmd += f' --output_dir "{output_dir}"'
    cmd += f' --excel "{excel_file}"'
    cmd += f' --roi_expand {roi_expand}'
    
    if debug:
        cmd += ' --debug'
    if no_split:
        cmd += ' --no_split'
    
    logging.info(f"执行OCR命令: {cmd}")
    
    try:
        # 执行OCR命令
        result = subprocess.run(cmd, shell=True, check=True, 
                             capture_output=True, text=True, encoding='utf-8')
        
        logging.info("OCR识别完成")
        return True, "OCR识别成功"
        
    except subprocess.CalledProcessError as e:
        error_msg = f"OCR识别失败: {e.stderr}"
        logging.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"OCR执行异常: {str(e)}"
        logging.error(error_msg)
        return False, error_msg

def validate_ocr_results(excel_file, validation_config=None):
    """
    校验OCR识别结果
    
    参数:
    excel_file: OCR结果Excel文件
    validation_config: 校验配置文件
    
    返回:
    (success, validation_report, output_files)
    """
    logging.info("开始数据校验...")
    
    # 检查Excel文件是否存在
    if not os.path.exists(excel_file):
        error_msg = f"OCR结果文件不存在: {excel_file}"
        logging.error(error_msg)
        return False, None, None
    
    try:
        # 读取OCR结果
        df = pd.read_excel(excel_file)
        logging.info(f"读取OCR结果: {len(df)}行数据")
        
        # 创建数据校验器
        validator = DataValidator(validation_config)
        
        # 执行校验
        validation_report = validator.validate_dataframe(df)
        
        # 保存校验报告
        output_files = validator.save_validation_report(validation_report, "validation_output")
        
        logging.info("数据校验完成")
        return True, validation_report, output_files
        
    except Exception as e:
        error_msg = f"数据校验失败: {str(e)}"
        logging.error(error_msg)
        return False, None, None

def generate_summary_report(ocr_success, validation_report, output_files):
    """
    生成综合摘要报告
    
    参数:
    ocr_success: OCR是否成功
    validation_report: 校验报告
    output_files: 输出文件列表
    
    返回:
    summary_file: 摘要报告文件路径
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"ocr_validation_summary_{timestamp}.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("OCR识别与数据校验综合报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # OCR处理结果
        f.write("1. OCR识别结果:\n")
        f.write("-" * 30 + "\n")
        if ocr_success:
            f.write("✅ OCR识别成功\n")
        else:
            f.write("❌ OCR识别失败\n")
        f.write("\n")
        
        # 数据校验结果
        f.write("2. 数据校验结果:\n")
        f.write("-" * 30 + "\n")
        if validation_report:
            data_summary = validation_report['数据概况']
            f.write(f"总行数: {data_summary['总行数']}\n")
            f.write(f"正常行数: {data_summary['正常行数']}\n")
            f.write(f"警告行数: {data_summary['警告行数']}\n")
            f.write(f"错误行数: {data_summary['错误行数']}\n")
            f.write(f"错误率: {data_summary['错误率']}\n")
            f.write(f"平均完整性: {data_summary['平均完整性']}\n")
            
            # 主要错误类型
            if validation_report['字段错误统计']:
                f.write("\n主要字段错误:\n")
                for field, count in list(validation_report['字段错误统计'].items())[:5]:
                    f.write(f"  {field}: {count}次\n")
            
            if validation_report['逻辑错误统计']:
                f.write("\n主要逻辑错误:\n")
                for error_type, count in list(validation_report['逻辑错误统计'].items())[:5]:
                    f.write(f"  {error_type}: {count}次\n")
        else:
            f.write("❌ 数据校验失败\n")
        f.write("\n")
        
        # 生成的文件
        f.write("3. 生成的文件:\n")
        f.write("-" * 30 + "\n")
        if output_files:
            for file_type, file_path in output_files.items():
                if file_path:
                    f.write(f"{file_type}: {file_path}\n")
        f.write("\n")
        
        # 建议和后续步骤
        f.write("4. 建议和后续步骤:\n")
        f.write("-" * 30 + "\n")
        if validation_report:
            error_count = validation_report['数据概况']['错误行数']
            if error_count == 0:
                f.write("✅ 所有数据校验通过，可以直接使用\n")
            elif error_count <= 3:
                f.write("⚠️ 发现少量错误数据，建议人工检查和修正\n")
                f.write("📁 错误数据已保存到error_data_*.xlsx文件中\n")
            else:
                f.write("❌ 发现较多错误数据，建议:\n")
                f.write("  1. 检查原始图片质量\n")
                f.write("  2. 调整OCR参数重新识别\n")
                f.write("  3. 人工校对关键数据\n")
                f.write("📁 错误数据已保存到error_data_*.xlsx文件中\n")
        
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("报告结束\n")
    
    logging.info(f"综合报告已保存到: {summary_file}")
    return summary_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OCR识别与数据校验集成工具')
    parser.add_argument('input_dir', help='输入图片目录')
    parser.add_argument('--output_dir', default='all_output', help='OCR输出目录')
    parser.add_argument('--excel', default='results.xlsx', help='Excel输出文件')
    parser.add_argument('--roi_expand', type=int, default=15, help='ROI扩展像素数')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no_split', action='store_true', help='跳过图片分割')
    parser.add_argument('--validation_config', help='数据校验配置文件')
    parser.add_argument('--skip_ocr', action='store_true', help='跳过OCR，直接校验现有Excel文件')
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.debug)
    
    logging.info("=" * 60)
    logging.info("开始OCR识别与数据校验集成处理")
    logging.info("=" * 60)
    
    # 记录参数
    logging.info(f"输入目录: {args.input_dir}")
    logging.info(f"Excel文件: {args.excel}")
    logging.info(f"ROI扩展: {args.roi_expand}像素")
    logging.info(f"调试模式: {args.debug}")
    logging.info(f"跳过OCR: {args.skip_ocr}")
    
    ocr_success = True
    validation_report = None
    output_files = None
    
    # 步骤1: OCR识别（如果不跳过）
    if not args.skip_ocr:
        logging.info("\n步骤1: OCR识别")
        logging.info("-" * 40)
        
        ocr_success, ocr_message = run_ocr_process(
            args.input_dir, args.output_dir, args.excel, 
            args.roi_expand, args.debug, args.no_split
        )
        
        if not ocr_success:
            logging.error(f"OCR识别失败: {ocr_message}")
            print(f"❌ OCR识别失败: {ocr_message}")
            return
        
        logging.info("✅ OCR识别完成")
    else:
        logging.info("跳过OCR识别步骤")
    
    # 步骤2: 数据校验
    logging.info("\n步骤2: 数据校验")
    logging.info("-" * 40)
    
    validation_success, validation_report, output_files = validate_ocr_results(
        args.excel, args.validation_config
    )
    
    if not validation_success:
        logging.error("数据校验失败")
        print("❌ 数据校验失败")
        return
    
    # 步骤3: 生成综合报告
    logging.info("\n步骤3: 生成综合报告")
    logging.info("-" * 40)
    
    summary_file = generate_summary_report(ocr_success, validation_report, output_files)
    
    # 显示结果摘要
    print("\n" + "=" * 60)
    print("处理完成摘要")
    print("=" * 60)
    
    if validation_report:
        data_summary = validation_report['数据概况']
        print(f"📊 数据概况:")
        print(f"   总行数: {data_summary['总行数']}")
        print(f"   正常行数: {data_summary['正常行数']}")
        print(f"   警告行数: {data_summary['警告行数']}")
        print(f"   错误行数: {data_summary['错误行数']}")
        print(f"   错误率: {data_summary['错误率']}")
        print(f"   平均完整性: {data_summary['平均完整性']}")
        
        error_count = data_summary['错误行数']
        if error_count == 0:
            print("\n✅ 所有数据校验通过！")
        elif error_count <= 3:
            print(f"\n⚠️ 发现 {error_count} 行错误数据，建议人工检查")
        else:
            print(f"\n❌ 发现 {error_count} 行错误数据，需要重新处理")
    
    print(f"\n📁 生成的文件:")
    print(f"   OCR结果: {args.excel}")
    if output_files:
        for file_type, file_path in output_files.items():
            if file_path:
                print(f"   {file_type}: {file_path}")
    print(f"   综合报告: {summary_file}")
    
    print(f"\n📋 详细日志: ocr_validation.log")
    print("=" * 60)

if __name__ == "__main__":
    main() 