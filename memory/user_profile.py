"""
用户画像记忆 — MySQL 数据库

存储结构化用户信息：
  - 偏好设置（饮食、出行、语言等）
  - 行为习惯
  - 常用功能
  - 个人信息（脱敏）

判断何时更新：
  - 同一偏好被提及 ≥ threshold 次 → 确认并写入
  - 用户明确设置 → 立即更新
  - 对话中提取到新个人信息 → 待确认后更新
"""
import json
import time
from typing import Optional
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

from config import MYSQL_CONFIG


class UserProfileDB:
    """
    基于 MySQL 的用户画像存储

    表结构：
    - user_profiles: 用户基础信息
    - user_preferences: 键值对偏好
    - user_behavior_log: 行为日志（用于分析习惯）
    """

    def __init__(self):
        self.config = MYSQL_CONFIG
        self._ensure_database()
        self._ensure_tables()

    @contextmanager
    def _get_conn(self):
        """获取数据库连接（上下文管理器）"""
        conn = pymysql.connect(
            host=self.config["host"],
            port=self.config["port"],
            user=self.config["user"],
            password=self.config["password"],
            database=self.config["database"],
            charset=self.config.get("charset", "utf8mb4"),
            cursorclass=DictCursor,
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_database(self):
        """确保数据库存在"""
        conn = pymysql.connect(
            host=self.config["host"],
            port=self.config["port"],
            user=self.config["user"],
            password=self.config["password"],
            charset=self.config.get("charset", "utf8mb4"),
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.config['database']}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            conn.commit()
        finally:
            conn.close()

    def _ensure_tables(self):
        """确保表结构存在"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # 用户基础信息表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id VARCHAR(64) NOT NULL UNIQUE,
                        name VARCHAR(128),
                        personal_info JSON COMMENT '脱敏个人信息',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_user_id (user_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

                # 用户偏好表（键值对）
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id VARCHAR(64) NOT NULL,
                        category VARCHAR(64) NOT NULL COMMENT 'food/travel/language/display/...',
                        pref_key VARCHAR(128) NOT NULL,
                        pref_value TEXT,
                        confidence FLOAT DEFAULT 0.5 COMMENT '置信度 0-1',
                        mention_count INT DEFAULT 1 COMMENT '被提及次数',
                        source VARCHAR(32) DEFAULT 'inferred' COMMENT 'inferred/explicit',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        UNIQUE KEY uk_user_cat_key (user_id, category, pref_key),
                        INDEX idx_user_category (user_id, category)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

                # 行为日志表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_behavior_log (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        user_id VARCHAR(64) NOT NULL,
                        action_type VARCHAR(64) NOT NULL COMMENT 'search/travel/recipe/cli/...',
                        action_detail JSON COMMENT '行为详情',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_user_time (user_id, created_at),
                        INDEX idx_action (action_type, created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

    # ==================== 用户基础信息 ====================

    def get_profile(self, user_id: str) -> Optional[dict]:
        """获取用户完整画像"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_profiles WHERE user_id = %s",
                    (user_id,),
                )
                profile = cur.fetchone()

                cur.execute(
                    "SELECT category, pref_key, pref_value, confidence, mention_count, source "
                    "FROM user_preferences WHERE user_id = %s",
                    (user_id,),
                )
                prefs = cur.fetchall()

        if not profile and not prefs:
            return None

        # 组装画像
        result = {
            "user_id": user_id,
            "name": None,
            "personal_info": {},
            "preferences": {},
            "updated_at": None,
        }

        if profile:
            result["name"] = profile.get("name")
            result["personal_info"] = self._safe_json_load(profile.get("personal_info"), {})
            result["updated_at"] = str(profile.get("updated_at", ""))

        # 按类别组织偏好
        for p in prefs:
            cat = p["category"]
            if cat not in result["preferences"]:
                result["preferences"][cat] = {}
            result["preferences"][cat][p["pref_key"]] = {
                "value": p["pref_value"],
                "confidence": p["confidence"],
                "mention_count": p["mention_count"],
                "source": p["source"],
            }

        return result

    def upsert_profile(self, user_id: str, name: str = None, personal_info: dict = None) -> bool:
        """创建或更新用户基础信息"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO user_profiles (user_id, name, personal_info)
                       VALUES (%s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                           name = COALESCE(VALUES(name), name),
                           personal_info = COALESCE(VALUES(personal_info), personal_info)""",
                    (user_id, name, json.dumps(personal_info, ensure_ascii=False) if personal_info else None),
                )
        return True

    # ==================== 偏好管理 ====================

    def get_preference(self, user_id: str, category: str, key: str) -> Optional[str]:
        """获取单个偏好值"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT pref_value, confidence FROM user_preferences
                       WHERE user_id = %s AND category = %s AND pref_key = %s""",
                    (user_id, category, key),
                )
                row = cur.fetchone()
        return row["pref_value"] if row else None

    def upsert_preference(
        self,
        user_id: str,
        category: str,
        key: str,
        value: str,
        confidence: float = 0.5,
        source: str = "inferred",
    ) -> bool:
        """
        更新偏好（存在则增加 mention_count 和置信度，不存在则插入）

        更新逻辑：
        - source=explicit（用户明确设置）→ confidence 直接设为 1.0
        - source=inferred（推断）→ 每次提及 confidence += 0.15，mention_count += 1
        """
        if source == "explicit":
            confidence = 1.0

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # 检查是否已存在
                cur.execute(
                    """SELECT id, confidence, mention_count FROM user_preferences
                       WHERE user_id = %s AND category = %s AND pref_key = %s""",
                    (user_id, category, key),
                )
                existing = cur.fetchone()

                if existing:
                    new_count = existing["mention_count"] + 1
                    new_confidence = min(1.0, existing["confidence"] + (0.15 if source == "inferred" else 0.5))
                    if source == "explicit":
                        new_confidence = 1.0
                    cur.execute(
                        """UPDATE user_preferences
                           SET pref_value = %s, confidence = %s, mention_count = %s,
                               source = %s, updated_at = NOW()
                           WHERE id = %s""",
                        (value, new_confidence, new_count, source, existing["id"]),
                    )
                else:
                    cur.execute(
                        """INSERT INTO user_preferences
                           (user_id, category, pref_key, pref_value, confidence, mention_count, source)
                           VALUES (%s, %s, %s, %s, %s, 1, %s)""",
                        (user_id, category, key, value, confidence, source),
                    )
        return True

    def should_update_profile(self, user_id: str, category: str, key: str, value: str) -> bool:
        """
        判断是否应该更新用户画像

        规则：
        - 被提及 ≥ 3 次（mention_count >= 3）→ 确认偏好
        - 用户明确设置 → 立即确认
        - 新偏好首次出现 → 暂存但标记为低置信度
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT mention_count, confidence, source FROM user_preferences
                       WHERE user_id = %s AND category = %s AND pref_key = %s""",
                    (user_id, category, key),
                )
                row = cur.fetchone()

        if not row:
            return True  # 新偏好，应该记录

        if row["source"] == "explicit":
            return False  # 已确认，无需更新

        if row["mention_count"] >= 3:
            return False  # 已充分确认

        if row["confidence"] >= 0.8:
            return False  # 高置信度，无需更新

        return True  # 需要继续积累

    # ==================== 行为日志 ====================

    def log_behavior(self, user_id: str, action_type: str, detail: dict = None) -> bool:
        """记录用户行为"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO user_behavior_log (user_id, action_type, action_detail)
                       VALUES (%s, %s, %s)""",
                    (user_id, action_type, json.dumps(detail, ensure_ascii=False) if detail else None),
                )
        return True

    def get_frequent_actions(self, user_id: str, top_n: int = 5) -> list[str]:
        """获取用户最常用功能"""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT action_type, COUNT(*) as cnt FROM user_behavior_log
                       WHERE user_id = %s
                       GROUP BY action_type ORDER BY cnt DESC LIMIT %s""",
                    (user_id, top_n),
                )
                return [row["action_type"] for row in cur.fetchall()]

    # ==================== 工具方法 ====================

    @staticmethod
    def _safe_json_load(data, default=None):
        if data is None:
            return default
        if isinstance(data, (dict, list)):
            return data
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return default
