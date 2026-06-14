-- ============================================================
-- 个人全能助手 — MySQL 初始化脚本
--
-- 使用方式：
--   mysql -u root -p < scripts/init_mysql.sql
--   密码：drp
-- ============================================================

-- 创建数据库
CREATE DATABASE IF NOT EXISTS `agent_memory`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE `agent_memory`;

-- ==================== 用户基础信息表 ====================
CREATE TABLE IF NOT EXISTS `user_profiles` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `user_id` VARCHAR(64) NOT NULL UNIQUE COMMENT '用户标识',
    `name` VARCHAR(128) COMMENT '用户名称',
    `personal_info` JSON COMMENT '脱敏个人信息',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX `idx_user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户基础信息';

-- ==================== 用户偏好表 ====================
CREATE TABLE IF NOT EXISTS `user_preferences` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `user_id` VARCHAR(64) NOT NULL COMMENT '用户标识',
    `category` VARCHAR(64) NOT NULL COMMENT '偏好类别：food/travel/language/display/...',
    `pref_key` VARCHAR(128) NOT NULL COMMENT '偏好键',
    `pref_value` TEXT COMMENT '偏好值',
    `confidence` FLOAT DEFAULT 0.5 COMMENT '置信度 0-1',
    `mention_count` INT DEFAULT 1 COMMENT '被提及次数',
    `source` VARCHAR(32) DEFAULT 'inferred' COMMENT '来源：inferred/explicit',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY `uk_user_cat_key` (`user_id`, `category`, `pref_key`),
    INDEX `idx_user_category` (`user_id`, `category`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户偏好';

-- ==================== 用户行为日志表 ====================
CREATE TABLE IF NOT EXISTS `user_behavior_log` (
    `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `user_id` VARCHAR(64) NOT NULL COMMENT '用户标识',
    `action_type` VARCHAR(64) NOT NULL COMMENT '行为类型：search/travel/recipe/cli/...',
    `action_detail` JSON COMMENT '行为详情',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX `idx_user_time` (`user_id`, `created_at`),
    INDEX `idx_action` (`action_type`, `created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户行为日志';
