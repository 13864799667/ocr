#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR数据校验系统 - 专业医学标准版

基于专业身体成分及代谢指标参考范围进行数据校验
支持性别特异性、年龄调整、临床意义评估等高级功能
"""

import pandas as pd
import numpy as np
import json
import os
import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging

class DataValidator:
    """专业数据校验器 - 基于医学标准"""
    
    def __init__(self, config_file=None):
        """
        初始化数据校验器
        
        参数:
        config_file: 配置文件路径，如果为None则使用默认配置
        """
        self.validation_rules = self._load_validation_rules(config_file)
        self.validation_results = []
        self.error_data = []
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_validation_rules(self, config_file):
        """加载校验规则"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 如果没有配置文件，使用内置的专业标准
        return self._get_default_professional_rules()
    
    def _get_default_professional_rules(self):
        """获取默认的专业校验规则"""
        return {
            "基本指标": {
                "体重(kg)": {
                    "min": 20.0, "max": 300.0, "type": "float", "required": True,
                    "description": "成人体重范围，单位：千克"
                },
                "BMI": {
                    "min": 10.0, "max": 60.0, "type": "float", "required": True,
                    "description": "BMI范围，中国标准：正常18.5-24.0，WHO标准：正常18.5-25.0",
                    "warning_ranges": {
                        "underweight": {"max": 18.5, "message": "体重过低"},
                        "normal_cn": {"min": 18.5, "max": 24.0, "message": "正常范围(中国标准)"},
                        "overweight_cn": {"min": 24.0, "max": 28.0, "message": "超重(中国标准)"},
                        "obese_cn": {"min": 28.0, "message": "肥胖(中国标准)"}
                    }
                },
                "体脂率(%)": {
                    "min": 3.0, "max": 60.0, "type": "float", "required": True,
                    "description": "体脂率范围，男性正常10-25%，女性正常17-32%",
                    "gender_specific": {
                        "male": {"healthy": {"min": 10, "max": 20}, "acceptable": {"min": 20, "max": 25}},
                        "female": {"healthy": {"min": 17, "max": 24}, "acceptable": {"min": 24, "max": 30}}
                    }
                }
            },
            "逻辑关系": {
                "体重组成": {"tolerance": 0.15, "enabled": True},
                "BMI计算验证": {"height_estimates": [1.6, 1.65, 1.7, 1.75, 1.8], "tolerance": 0.25, "enabled": True}
            }
        }
    
    def _detect_gender(self, row_data):
        """
        尝试从数据中推断性别
        
        参数:
        row_data: 一行数据的字典
        
        返回:
        'male', 'female', 或 None
        """
        # 基于体脂率和骨骼肌比例的简单推断
        try:
            body_fat = float(row_data.get("体脂率(%)", 0))
            weight = float(row_data.get("体重(kg)", 0))
            muscle = float(row_data.get("骨骼肌(kg)", 0))
            
            if weight > 0 and muscle > 0:
                muscle_ratio = muscle / weight
                
                # 简单的性别推断逻辑
                if body_fat < 15 and muscle_ratio > 0.35:
                    return 'male'
                elif body_fat > 25 and muscle_ratio < 0.32:
                    return 'female'
                elif muscle_ratio > 0.38:
                    return 'male'
                elif muscle_ratio < 0.30:
                    return 'female'
        except (ValueError, TypeError):
            pass
        
        return None
    
    def validate_single_value_enhanced(self, value, field_name, rules, row_data=None):
        """
        增强版单个数值校验，支持性别特异性和专业标准
        
        参数:
        value: 待校验的值
        field_name: 字段名称
        rules: 校验规则
        row_data: 完整行数据，用于性别推断等
        
        返回:
        (is_valid, error_messages, warning_messages)
        """
        errors = []
        warnings = []
        
        # 检查是否为空值
        if pd.isna(value) or value == "" or value is None:
            if rules.get("required", False):
                errors.append(f"{field_name}: 必填字段为空")
            return len(errors) == 0, errors, warnings
        
        # 类型转换和检查
        try:
            if rules["type"] == "float":
                numeric_value = float(value)
            elif rules["type"] == "int":
                numeric_value = int(float(value))
            else:
                numeric_value = value
        except (ValueError, TypeError):
            errors.append(f"{field_name}: 无法转换为{rules['type']}类型，当前值: {value}")
            return False, errors, warnings
        
        # 基本范围检查
        if "min" in rules and numeric_value < rules["min"]:
            errors.append(f"{field_name}: 值过小 ({numeric_value} < {rules['min']})")
        
        if "max" in rules and numeric_value > rules["max"]:
            errors.append(f"{field_name}: 值过大 ({numeric_value} > {rules['max']})")
        
        # 检查负数
        if not rules.get("allow_negative", True) and numeric_value < 0:
            errors.append(f"{field_name}: 不允许负数，当前值: {numeric_value}")
        
        # 性别特异性校验
        if "gender_specific" in rules and row_data is not None:
            # 确保row_data是字典格式
            if hasattr(row_data, 'to_dict'):
                row_dict = row_data.to_dict()
            else:
                row_dict = row_data
            
            gender = self._detect_gender(row_dict)
            if gender and gender in rules["gender_specific"]:
                gender_rules = rules["gender_specific"][gender]
                self._check_gender_specific_ranges(numeric_value, field_name, gender_rules, gender, warnings)
        
        # 警告范围检查
        if "warning_ranges" in rules:
            self._check_warning_ranges(numeric_value, field_name, rules["warning_ranges"], warnings)
        
        # 百分比范围检查（相对于体重）
        if "percentage_ranges" in rules and row_data is not None:
            # 确保row_data是字典格式
            if hasattr(row_data, 'to_dict'):
                row_dict = row_data.to_dict()
            else:
                row_dict = row_data
            self._check_percentage_ranges(numeric_value, field_name, rules["percentage_ranges"], row_dict, warnings)
        
        return len(errors) == 0, errors, warnings
    
    def _check_gender_specific_ranges(self, value, field_name, gender_rules, gender, warnings):
        """检查性别特异性范围"""
        gender_label = "男性" if gender == "male" else "女性"
        
        for range_type, range_def in gender_rules.items():
            if "min" in range_def and "max" in range_def:
                if range_def["min"] <= value <= range_def["max"]:
                    warnings.append(f"{field_name}: {gender_label}{range_type}范围 ({value})")
                    break
            elif "min" in range_def and value >= range_def["min"]:
                warnings.append(f"{field_name}: {gender_label}{range_type}范围 ({value})")
                break
            elif "max" in range_def and value <= range_def["max"]:
                warnings.append(f"{field_name}: {gender_label}{range_type}范围 ({value})")
                break
    
    def _check_warning_ranges(self, value, field_name, warning_ranges, warnings):
        """检查警告范围"""
        for range_name, range_def in warning_ranges.items():
            if "min" in range_def and "max" in range_def:
                if range_def["min"] <= value <= range_def["max"]:
                    warnings.append(f"{field_name}: {range_def['message']} ({value})")
            elif "min" in range_def and value >= range_def["min"]:
                warnings.append(f"{field_name}: {range_def['message']} ({value})")
            elif "max" in range_def and value <= range_def["max"]:
                warnings.append(f"{field_name}: {range_def['message']} ({value})")
    
    def _check_percentage_ranges(self, value, field_name, percentage_ranges, row_data, warnings):
        """检查百分比范围（相对于体重）"""
        try:
            weight = float(row_data.get("体重(kg)", 0))
            if weight > 0:
                ratio = value / weight
                
                if isinstance(percentage_ranges, dict) and "min" in percentage_ranges:
                    # 单一范围
                    if not (percentage_ranges["min"] <= ratio <= percentage_ranges["max"]):
                        expected_min = weight * percentage_ranges["min"]
                        expected_max = weight * percentage_ranges["max"]
                        warnings.append(f"{field_name}: 占体重比例异常 ({ratio:.1%}，期望{percentage_ranges['min']:.1%}-{percentage_ranges['max']:.1%}，即{expected_min:.1f}-{expected_max:.1f}kg)")
                
                elif isinstance(percentage_ranges, dict) and "male" in percentage_ranges:
                    # 性别特异性范围
                    gender = self._detect_gender(row_data)
                    if gender and gender in percentage_ranges:
                        gender_range = percentage_ranges[gender]
                        if not (gender_range["min"] <= ratio <= gender_range["max"]):
                            gender_label = "男性" if gender == "male" else "女性"
                            warnings.append(f"{field_name}: {gender_label}占体重比例异常 ({ratio:.1%}，期望{gender_range['min']:.1%}-{gender_range['max']:.1%})")
        except (ValueError, TypeError):
            pass
    
    def validate_logical_relationships_enhanced(self, row_data):
        """
        增强版逻辑关系校验，支持更多专业标准
        
        参数:
        row_data: 一行数据的字典
        
        返回:
        (is_valid, error_messages, warning_messages)
        """
        errors = []
        warnings = []
        logic_rules = self.validation_rules.get("逻辑关系", {})
        
        # 1. 体重组成校验
        if logic_rules.get("体重组成", {}).get("enabled", False):
            self._validate_body_composition(row_data, logic_rules["体重组成"], errors, warnings)
        
        # 2. BMI计算验证
        if logic_rules.get("BMI计算验证", {}).get("enabled", False):
            self._validate_bmi_calculation(row_data, logic_rules["BMI计算验证"], errors, warnings)
        
        # 3. 节段肌肉对称性
        if logic_rules.get("节段肌肉对称性", {}).get("enabled", False):
            self._validate_muscle_symmetry(row_data, logic_rules["节段肌肉对称性"], errors, warnings)
        
        # 4. 体脂率与BMI相关性
        if logic_rules.get("体脂率与BMI相关性", {}).get("enabled", False):
            self._validate_fat_bmi_correlation(row_data, logic_rules["体脂率与BMI相关性"], errors, warnings)
        
        # 5. 骨骼肌率合理性
        if logic_rules.get("骨骼肌率合理性", {}).get("enabled", False):
            self._validate_muscle_ratio(row_data, logic_rules["骨骼肌率合理性"], errors, warnings)
        
        return len(errors) == 0, errors, warnings
    
    def _validate_body_composition(self, row_data, rule, errors, warnings):
        """验证体重组成"""
        try:
            weight = float(row_data.get("体重(kg)", 0))
            water = float(row_data.get("水分含量(kg)", 0))
            protein = float(row_data.get("蛋白质(kg)", 0))
            mineral = float(row_data.get("无机盐(kg)", 0))
            body_fat_rate = float(row_data.get("体脂率(%)", 0)) / 100
            
            fat_weight = weight * body_fat_rate
            total_components = water + protein + mineral + fat_weight
            
            if weight > 0 and total_components > 0:
                diff_ratio = abs(weight - total_components) / weight
                if diff_ratio > rule["tolerance"]:
                    errors.append(f"体重组成不匹配: 体重{weight}kg vs 成分总和{total_components:.1f}kg (差异{diff_ratio:.1%})")
                elif diff_ratio > rule["tolerance"] * 0.7:  # 70%阈值作为警告
                    warnings.append(f"体重组成轻微不匹配: 差异{diff_ratio:.1%}")
        except (ValueError, TypeError, KeyError):
            pass
    
    def _validate_bmi_calculation(self, row_data, rule, errors, warnings):
        """验证BMI计算"""
        try:
            weight = float(row_data.get("体重(kg)", 0))
            bmi = float(row_data.get("BMI", 0))
            
            if weight > 0 and bmi > 0:
                # 使用多个身高估算值
                height_estimates = rule.get("height_estimates", [1.7])
                min_diff = float('inf')
                best_height = None
                
                for height in height_estimates:
                    estimated_bmi = weight / (height ** 2)
                    diff_ratio = abs(bmi - estimated_bmi) / bmi
                    if diff_ratio < min_diff:
                        min_diff = diff_ratio
                        best_height = height
                
                if min_diff > rule["tolerance"]:
                    estimated_bmi = weight / (best_height ** 2)
                    errors.append(f"BMI计算可能有误: 实际BMI{bmi} vs 估算BMI{estimated_bmi:.1f} (基于身高{best_height}m，差异{min_diff:.1%})")
                elif min_diff > rule["tolerance"] * 0.7:
                    warnings.append(f"BMI计算轻微偏差: 差异{min_diff:.1%}")
        except (ValueError, TypeError, KeyError):
            pass
    
    def _validate_muscle_symmetry(self, row_data, rule, errors, warnings):
        """验证肌肉对称性"""
        for pair in rule.get("pairs", []):
            try:
                left_field, right_field = pair
                left_value = float(row_data.get(left_field, 0))
                right_value = float(row_data.get(right_field, 0))
                
                if left_value > 0 and right_value > 0:
                    diff_ratio = abs(left_value - right_value) / max(left_value, right_value)
                    if diff_ratio > rule["tolerance"]:
                        errors.append(f"肌肉对称性异常: {left_field}({left_value}kg) vs {right_field}({right_value}kg) (差异{diff_ratio:.1%})")
                    elif diff_ratio > rule["tolerance"] * 0.7:
                        warnings.append(f"肌肉轻微不对称: {left_field}({left_value}kg) vs {right_field}({right_value}kg) (差异{diff_ratio:.1%})")
            except (ValueError, TypeError, KeyError):
                pass
    
    def _validate_fat_bmi_correlation(self, row_data, rule, errors, warnings):
        """验证体脂率与BMI相关性"""
        try:
            bmi = float(row_data.get("BMI", 0))
            body_fat = float(row_data.get("体脂率(%)", 0))
            
            if bmi > 0 and body_fat > 0:
                correlation_rules = rule.get("correlation_rules", {})
                
                # 检查低BMI高体脂（隐性肥胖）
                if "low_bmi_high_fat" in correlation_rules:
                    rule_def = correlation_rules["low_bmi_high_fat"]
                    if bmi <= rule_def["bmi_max"] and body_fat >= rule_def["fat_min"]:
                        warnings.append(rule_def["message"] + f" (BMI:{bmi}, 体脂率:{body_fat}%)")
                
                # 检查高BMI低体脂（肌肉型）
                if "high_bmi_low_fat" in correlation_rules:
                    rule_def = correlation_rules["high_bmi_low_fat"]
                    if bmi >= rule_def["bmi_min"] and body_fat <= rule_def["fat_max"]:
                        warnings.append(rule_def["message"] + f" (BMI:{bmi}, 体脂率:{body_fat}%)")
        except (ValueError, TypeError, KeyError):
            pass
    
    def _validate_muscle_ratio(self, row_data, rule, errors, warnings):
        """验证骨骼肌率合理性"""
        try:
            weight = float(row_data.get("体重(kg)", 0))
            muscle = float(row_data.get("骨骼肌(kg)", 0))
            
            if weight > 0 and muscle > 0:
                muscle_ratio = muscle / weight
                gender = self._detect_gender(row_data)
                
                if gender and gender in rule.get("gender_specific", {}):
                    gender_rule = rule["gender_specific"][gender]
                    if not (gender_rule["min_ratio"] <= muscle_ratio <= gender_rule["max_ratio"]):
                        gender_label = "男性" if gender == "male" else "女性"
                        errors.append(f"骨骼肌率异常: {gender_label}骨骼肌率{muscle_ratio:.1%}，正常范围{gender_rule['min_ratio']:.1%}-{gender_rule['max_ratio']:.1%}")
        except (ValueError, TypeError, KeyError):
            pass
    
    def validate_dataframe_enhanced(self, df):
        """
        增强版DataFrame校验
        
        参数:
        df: 待校验的DataFrame
        
        返回:
        validation_report: 校验报告字典
        """
        self.validation_results = []
        self.error_data = []
        
        self.logger.info(f"开始专业数据校验，共{len(df)}行")
        
        for index, row in df.iterrows():
            row_result = {
                "行号": index + 1,
                "文件名": row.get("文件名", f"行{index+1}"),
                "字段错误": [],
                "字段警告": [],
                "逻辑错误": [],
                "逻辑警告": [],
                "完整性评分": 0,
                "缺失字段": [],
                "推断性别": None,
                "临床评估": [],
                "总体状态": "正常"
            }
            
            has_errors = False
            has_warnings = False
            
            # 推断性别
            row_dict = row.to_dict()  # 将pandas Series转换为字典
            row_result["推断性别"] = self._detect_gender(row_dict)
            
            # 1. 校验各个字段
            for category, fields in self.validation_rules.items():
                if category in ["逻辑关系", "数据质量", "年龄性别调整", "肌肉减少症筛查"]:
                    continue
                    
                for field_name, rules in fields.items():
                    value = row.get(field_name)
                    is_valid, errors, warnings = self.validate_single_value_enhanced(value, field_name, rules, row_dict)
                    
                    if not is_valid:
                        row_result["字段错误"].extend(errors)
                        has_errors = True
                    
                    if warnings:
                        row_result["字段警告"].extend(warnings)
                        has_warnings = True
            
            # 2. 校验逻辑关系
            is_logic_valid, logic_errors, logic_warnings = self.validate_logical_relationships_enhanced(row_dict)
            if not is_logic_valid:
                row_result["逻辑错误"].extend(logic_errors)
                has_errors = True
            
            if logic_warnings:
                row_result["逻辑警告"].extend(logic_warnings)
                has_warnings = True
            
            # 3. 校验数据完整性
            completeness_score, missing_fields = self.validate_data_completeness(row_dict)
            row_result["完整性评分"] = completeness_score
            row_result["缺失字段"] = missing_fields
            
            # 4. 临床评估
            clinical_assessment = self._perform_clinical_assessment(row_dict)
            row_result["临床评估"] = clinical_assessment
            
            # 5. 确定总体状态
            if has_errors:
                row_result["总体状态"] = "错误"
                # 保存有问题的数据
                error_row = row.copy()
                all_errors = row_result["字段错误"] + row_result["逻辑错误"]
                error_row["校验错误"] = "; ".join(all_errors)
                self.error_data.append(error_row)
            elif has_warnings or completeness_score < 0.8:
                row_result["总体状态"] = "警告"
            
            self.validation_results.append(row_result)
        
        # 生成校验报告
        report = self._generate_enhanced_validation_report()
        self.logger.info(f"专业校验完成，发现{len(self.error_data)}行有问题的数据")
        
        return report
    
    def _perform_clinical_assessment(self, row_data):
        """执行临床评估"""
        assessments = []
        
        try:
            # BMI评估
            bmi = float(row_data.get("BMI", 0))
            if bmi > 0:
                if bmi < 18.5:
                    assessments.append("体重过低，建议营养评估")
                elif bmi >= 28.0:
                    assessments.append("肥胖，建议减重管理")
                elif bmi >= 24.0:
                    assessments.append("超重，建议体重控制")
            
            # 体脂率评估
            body_fat = float(row_data.get("体脂率(%)", 0))
            gender = self._detect_gender(row_data)
            if body_fat > 0 and gender:
                if gender == "male" and body_fat > 25:
                    assessments.append("男性体脂率偏高，建议减脂")
                elif gender == "female" and body_fat > 32:
                    assessments.append("女性体脂率偏高，建议减脂")
                elif (gender == "male" and body_fat < 5) or (gender == "female" and body_fat < 13):
                    assessments.append("体脂率过低，可能影响健康")
            
            # 内脏脂肪评估
            visceral_fat = row_data.get("内脏脂肪等级")
            if visceral_fat and str(visceral_fat).isdigit():
                vf_level = int(visceral_fat)
                if vf_level >= 15:
                    assessments.append("内脏脂肪过高，建议医学评估")
                elif vf_level >= 10:
                    assessments.append("内脏脂肪偏高，建议关注")
            
            # 健康分数评估
            health_score = row_data.get("健康分数")
            if health_score and str(health_score).replace('.', '').isdigit():
                score = float(health_score)
                if score < 70:
                    assessments.append("健康分数偏低，建议改善生活方式")
                elif score >= 90:
                    assessments.append("健康分数优秀，保持良好状态")
        
        except (ValueError, TypeError):
            pass
        
        return assessments
    
    def validate_data_completeness(self, row_data):
        """
        校验数据完整性
        
        参数:
        row_data: 一行数据的字典
        
        返回:
        (completeness_score, missing_fields)
        """
        total_fields = 0
        missing_fields = []
        
        for category, fields in self.validation_rules.items():
            if category in ["逻辑关系", "数据质量", "年龄性别调整", "肌肉减少症筛查"]:
                continue
                
            for field_name, rules in fields.items():
                total_fields += 1
                value = row_data.get(field_name)
                
                if pd.isna(value) or value == "" or value is None:
                    if rules.get("required", False):
                        missing_fields.append(f"{field_name} (必填)")
                    else:
                        missing_fields.append(f"{field_name} (可选)")
        
        completeness_score = (total_fields - len(missing_fields)) / total_fields if total_fields > 0 else 0
        return completeness_score, missing_fields
    
    def _generate_enhanced_validation_report(self):
        """生成增强版校验报告"""
        total_rows = len(self.validation_results)
        error_rows = len([r for r in self.validation_results if r["总体状态"] == "错误"])
        warning_rows = len([r for r in self.validation_results if r["总体状态"] == "警告"])
        normal_rows = total_rows - error_rows - warning_rows
        
        # 统计错误和警告类型
        field_errors = {}
        field_warnings = {}
        logic_errors = {}
        logic_warnings = {}
        clinical_assessments = {}
        
        for result in self.validation_results:
            # 字段错误统计
            for error in result["字段错误"]:
                field_name = error.split(":")[0]
                field_errors[field_name] = field_errors.get(field_name, 0) + 1
            
            # 字段警告统计
            for warning in result["字段警告"]:
                field_name = warning.split(":")[0]
                field_warnings[field_name] = field_warnings.get(field_name, 0) + 1
            
            # 逻辑错误统计
            for error in result["逻辑错误"]:
                error_type = error.split(":")[0]
                logic_errors[error_type] = logic_errors.get(error_type, 0) + 1
            
            # 逻辑警告统计
            for warning in result["逻辑警告"]:
                warning_type = warning.split(":")[0]
                logic_warnings[warning_type] = logic_warnings.get(warning_type, 0) + 1
            
            # 临床评估统计
            for assessment in result["临床评估"]:
                clinical_assessments[assessment] = clinical_assessments.get(assessment, 0) + 1
        
        # 计算平均完整性评分
        avg_completeness = np.mean([r["完整性评分"] for r in self.validation_results])
        
        # 性别分布统计
        gender_distribution = {"male": 0, "female": 0, "unknown": 0}
        for result in self.validation_results:
            gender = result.get("推断性别", "unknown")
            if gender in gender_distribution:
                gender_distribution[gender] += 1
            else:
                gender_distribution["unknown"] += 1
        
        report = {
            "校验时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "校验版本": "专业医学标准版 v2.1",
            "数据概况": {
                "总行数": total_rows,
                "正常行数": normal_rows,
                "警告行数": warning_rows,
                "错误行数": error_rows,
                "错误率": f"{error_rows/total_rows*100:.1f}%" if total_rows > 0 else "0%",
                "平均完整性": f"{avg_completeness*100:.1f}%"
            },
            "性别分布": {
                "推断男性": gender_distribution["male"],
                "推断女性": gender_distribution["female"],
                "未知性别": gender_distribution["unknown"]
            },
            "字段错误统计": field_errors,
            "字段警告统计": field_warnings,
            "逻辑错误统计": logic_errors,
            "逻辑警告统计": logic_warnings,
            "临床评估统计": clinical_assessments,
            "详细结果": self.validation_results
        }
        
        return report
    
    def save_validation_report(self, report, output_dir="validation_output"):
        """
        保存校验报告
        
        参数:
        report: 校验报告
        output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存JSON格式的详细报告
        json_file = os.path.join(output_dir, f"validation_report_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 2. 保存Excel格式的错误数据
        error_excel = None
        if self.error_data:
            error_excel = os.path.join(output_dir, f"error_data_{timestamp}.xlsx")
            error_df = pd.DataFrame(self.error_data)
            error_df.to_excel(error_excel, index=False)
            self.logger.info(f"错误数据已保存到: {error_excel}")
        
        # 3. 保存增强版文本报告
        txt_file = os.path.join(output_dir, f"validation_summary_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("OCR数据校验报告 - 专业医学标准版\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"校验时间: {report['校验时间']}\n")
            f.write(f"校验版本: {report['校验版本']}\n\n")
            
            f.write("数据概况:\n")
            for key, value in report['数据概况'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("性别分布:\n")
            for key, value in report['性别分布'].items():
                f.write(f"  {key}: {value}人\n")
            f.write("\n")
            
            if report['字段错误统计']:
                f.write("字段错误统计 (Top 10):\n")
                sorted_errors = sorted(report['字段错误统计'].items(), key=lambda x: x[1], reverse=True)
                for field, count in sorted_errors[:10]:
                    f.write(f"  {field}: {count}次\n")
                f.write("\n")
            
            if report['字段警告统计']:
                f.write("字段警告统计 (Top 10):\n")
                sorted_warnings = sorted(report['字段警告统计'].items(), key=lambda x: x[1], reverse=True)
                for field, count in sorted_warnings[:10]:
                    f.write(f"  {field}: {count}次\n")
                f.write("\n")
            
            if report['逻辑错误统计']:
                f.write("逻辑错误统计:\n")
                for error_type, count in sorted(report['逻辑错误统计'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {error_type}: {count}次\n")
                f.write("\n")
            
            if report['临床评估统计']:
                f.write("临床评估统计:\n")
                for assessment, count in sorted(report['临床评估统计'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {assessment}: {count}次\n")
                f.write("\n")
            
            # 详细错误列表
            f.write("详细问题列表:\n")
            f.write("-" * 40 + "\n")
            for result in self.validation_results:
                if result["总体状态"] != "正常":
                    f.write(f"\n文件: {result['文件名']} (行{result['行号']})\n")
                    f.write(f"状态: {result['总体状态']}\n")
                    f.write(f"完整性: {result['完整性评分']*100:.1f}%\n")
                    if result.get("推断性别"):
                        gender_label = "男性" if result["推断性别"] == "male" else "女性"
                        f.write(f"推断性别: {gender_label}\n")
                    
                    if result["字段错误"]:
                        f.write("字段错误:\n")
                        for error in result["字段错误"]:
                            f.write(f"  ❌ {error}\n")
                    
                    if result["字段警告"]:
                        f.write("字段警告:\n")
                        for warning in result["字段警告"]:
                            f.write(f"  ⚠️ {warning}\n")
                    
                    if result["逻辑错误"]:
                        f.write("逻辑错误:\n")
                        for error in result["逻辑错误"]:
                            f.write(f"  ❌ {error}\n")
                    
                    if result["逻辑警告"]:
                        f.write("逻辑警告:\n")
                        for warning in result["逻辑警告"]:
                            f.write(f"  ⚠️ {warning}\n")
                    
                    if result["临床评估"]:
                        f.write("临床评估:\n")
                        for assessment in result["临床评估"]:
                            f.write(f"  🏥 {assessment}\n")
                    
                    if result["缺失字段"]:
                        f.write("缺失字段:\n")
                        for field in result["缺失字段"]:
                            f.write(f"  - {field}\n")
        
        self.logger.info(f"专业校验报告已保存到: {output_dir}")
        return {
            "json_report": json_file,
            "error_excel": error_excel,
            "text_summary": txt_file
        }
    
    # 保持向后兼容性的方法
    def validate_single_value(self, value, field_name, rules):
        """向后兼容的单值校验方法"""
        is_valid, errors, warnings = self.validate_single_value_enhanced(value, field_name, rules)
        return is_valid, errors
    
    def validate_logical_relationships(self, row_data):
        """向后兼容的逻辑关系校验方法"""
        is_valid, errors, warnings = self.validate_logical_relationships_enhanced(row_data)
        return is_valid, errors
    
    def validate_dataframe(self, df):
        """向后兼容的DataFrame校验方法"""
        return self.validate_dataframe_enhanced(df)
    
    def get_validation_summary(self):
        """获取校验摘要"""
        if not self.validation_results:
            return "尚未进行数据校验"
        
        total = len(self.validation_results)
        errors = len([r for r in self.validation_results if r["总体状态"] == "错误"])
        warnings = len([r for r in self.validation_results if r["总体状态"] == "警告"])
        
        return f"专业校验完成: 总计{total}行，错误{errors}行，警告{warnings}行，正常{total-errors-warnings}行"

def main():
    """主函数 - 用于测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR数据校验工具 - 专业医学标准版')
    parser.add_argument('excel_file', help='待校验的Excel文件')
    parser.add_argument('--config', help='校验规则配置文件')
    parser.add_argument('--output', default='validation_output', help='输出目录')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.excel_file):
        print(f"错误: 文件 {args.excel_file} 不存在")
        return
    
    # 创建校验器
    validator = DataValidator(args.config)
    
    # 读取数据
    try:
        df = pd.read_excel(args.excel_file)
        print(f"成功读取数据文件: {args.excel_file}")
        print(f"数据行数: {len(df)}")
        print(f"数据列数: {len(df.columns)}")
    except Exception as e:
        print(f"读取文件失败: {str(e)}")
        return
    
    # 执行校验
    print("\n开始专业数据校验...")
    report = validator.validate_dataframe_enhanced(df)
    
    # 保存报告
    output_files = validator.save_validation_report(report, args.output)
    
    # 显示摘要
    print("\n" + "="*60)
    print("专业校验摘要:")
    print("="*60)
    for key, value in report['数据概况'].items():
        print(f"{key}: {value}")
    
    print("\n性别分布:")
    for key, value in report['性别分布'].items():
        print(f"{key}: {value}")
    
    print(f"\n校验报告已保存到: {args.output}")
    print("生成的文件:")
    for file_type, file_path in output_files.items():
        if file_path:
            print(f"  {file_type}: {file_path}")

if __name__ == "__main__":
    main() 